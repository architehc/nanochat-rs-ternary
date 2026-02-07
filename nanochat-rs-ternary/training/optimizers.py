"""
optimizers.py — Muon + Lion optimizers for nanochat-rs Ternary training

Muon (for 2D+ linear weight matrices):
  Nesterov momentum + Newton-Schulz orthogonalization of gradient.
  NS iteration uses quintic polynomial: X_{k+1} = a*X + (b*A + c*A^2) @ X
  where A = X @ X^T, with coefficients (3.4445, -4.7750, 2.0315).
  Only apply orthogonalization to 2D+ params; 1D params get plain momentum.

  Reference: KellerJordan/Muon (https://github.com/KellerJordan/Muon)

Lion (for 1D params: norms, mHC, embeddings):
  Sign-based SGD: update = sign(beta1*momentum + (1-beta1)*grad)
  Then momentum = beta2*momentum + (1-beta2)*grad
  Weight decay applied multiplicatively before update.

  Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms"
             (arXiv 2302.06675)

No external dependencies beyond PyTorch.
"""

import torch
from torch.optim import Optimizer


# =============================================================================
# Newton-Schulz orthogonalization
# =============================================================================

def newton_schulz_orthogonalize(G: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """Orthogonalize gradient matrix G via Newton-Schulz iteration.

    Computes the zeroth power / polar factor of G, i.e., the nearest
    orthogonal matrix UV^T from the SVD G = U S V^T.

    Uses a quintic polynomial NS iteration whose coefficients (3.4445,
    -4.7750, 2.0315) are tuned to maximize convergence speed. This is the
    same iteration used in the reference Muon optimizer.

    For tall matrices (rows > cols), we transpose, iterate, and transpose back
    so the inner loop always works with the smaller dimension in rows.

    The initial X is normalized by its Frobenius norm so that all singular
    values are in (0, 1], which is required for the quintic iteration to
    converge.

    After iteration, the result is scaled by sqrt(max(1, rows/cols)) to
    preserve the aspect-ratio energy.

    Args:
        G: 2D gradient tensor [rows, cols].
        ns_steps: Number of Newton-Schulz iterations (5 is typically sufficient).

    Returns:
        Orthogonalized gradient, same shape as G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.float()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Normalize so spectral norm is at most 1 (Frobenius norm bound)
    norms = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norms + 1e-7)

    # Quintic Newton-Schulz iteration
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT

    # Scale by aspect ratio to preserve energy
    rows, cols = G.shape[-2], G.shape[-1]
    X = X * max(1, rows / cols) ** 0.5

    return X


# =============================================================================
# Muon Optimizer
# =============================================================================

class Muon(Optimizer):
    """Muon optimizer: Nesterov momentum with Newton-Schulz orthogonalization.

    For 2D+ parameters: momentum buffer is updated, Nesterov interpolation is
    computed, then orthogonalized via Newton-Schulz before applying the update.
    For 1D parameters: plain momentum update (no orthogonalization).

    The momentum update uses EMA-style interpolation:
      momentum = beta * momentum + (1 - beta) * grad
    Nesterov look-ahead:
      update = beta * momentum + (1 - beta) * grad

    Weight decay is applied multiplicatively: p *= (1 - lr * wd).

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.02).
        momentum: Momentum factor beta (default: 0.95).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        weight_decay: Weight decay (default: 0.0).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get or initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']

                # EMA momentum update: buf = beta * buf + (1 - beta) * grad
                buf.lerp_(grad, 1.0 - beta)

                if p.dim() >= 2:
                    # Nesterov look-ahead: update = beta * buf + (1 - beta) * grad
                    # Use a temporary to avoid modifying grad in-place
                    update = torch.lerp(grad, buf, beta)

                    # Reshape to 2D for orthogonalization, then restore
                    orig_shape = update.shape
                    update_2d = update.reshape(orig_shape[0], -1)
                    update_orth = newton_schulz_orthogonalize(update_2d, ns_steps)
                    update = update_orth.reshape(orig_shape)
                else:
                    # 1D params: plain momentum update (no orthogonalization)
                    update = buf.clone()

                # Weight decay (multiplicative)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # Apply update
                p.data.add_(update, alpha=-lr)

        return loss


# =============================================================================
# Lion Optimizer
# =============================================================================

class Lion(Optimizer):
    """Lion optimizer: sign-based update with EMA momentum.

    Update rule:
      1. Weight decay: p = p * (1 - lr * weight_decay)
      2. Update direction: update = sign(beta1 * m + (1 - beta1) * grad)
      3. Apply: p = p - lr * update
      4. Momentum update: m = beta2 * m + (1 - beta2) * grad

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-4).
        betas: Coefficients for computing update direction and momentum
               (default: (0.9, 0.99)).
        weight_decay: Weight decay (L2 penalty) (default: 0.0).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get or initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                m = state['exp_avg']

                # 1. Weight decay (multiplicative, applied before update)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # 2. Compute update direction: sign(interpolation of m and grad)
                update = (beta1 * m + (1.0 - beta1) * grad).sign_()

                # 3. Apply update
                p.data.add_(update, alpha=-lr)

                # 4. Update momentum (after the parameter update, using original grad)
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
