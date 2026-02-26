# nano-500m-wave-haar Architecture

> A ~280M parameter transformer language model trained on Rust source code,
> featuring **Haar wavelet attention**, **mHC-lite doubly stochastic residuals**,
> and **ternary quantization-aware training (QAT)**.

## High-Level Forward Pass

```mermaid
flowchart TD
    subgraph INPUT["**Input Pipeline**"]
        tokens["Token IDs<br/>[batch=2, seq=512]<br/>BPE vocab=4096"]
    end

    subgraph EMBED["**Embedding Layer** (FP32, not quantized)"]
        emb["Token Embedding<br/>4096 × 768<br/>(weight-tied with LM head)"]
    end

    subgraph MHC_EXPAND["**mHC-lite Expand** (2-stream duplication)"]
        expand["cat([x, x], dim=-1)<br/>[B, 512, 768] → [B, 512, 1536]<br/><i>Creates 2 parallel information streams<br/>that mix via doubly stochastic matrices</i>"]
    end

    subgraph BLOCKS["**16 Transformer Blocks**<br/>(8 standard attention + 8 Haar wavefield, interleaved)"]
        direction TB
        block_loop["Each block: x_exp [B, 512, 1536] → x_exp [B, 512, 1536]<br/>See detailed block diagram below"]
    end

    subgraph MHC_COLLAPSE["**mHC-lite Collapse** (stream merging)"]
        collapse["avg(stream₀, stream₁)<br/>[B, 512, 1536] → [B, 512, 768]<br/><i>Combines learned stream representations</i>"]
    end

    subgraph FINAL["**Output Head**"]
        final_norm["RMSNorm<br/>(final layer normalization)"]
        lm_head["LM Head (weight-tied Embedding.T)<br/>768 → 4096 logits"]
        mtp["MTP Heads ×3<br/>Predict next 1, 2, 3 tokens<br/><i>+20% data efficiency</i>"]
    end

    subgraph LOSS["**Loss Computation**"]
        ce["Cross-Entropy Loss<br/>label smoothing ε=0.1"]
        mtp_loss["MTP Loss<br/>weight=0.2, geometric decay"]
        total_loss["Total Loss = CE + 0.2 × MTP"]
    end

    tokens --> emb
    emb -->|"[B, 512, 768]"| expand
    expand -->|"[B, 512, 1536]"| block_loop
    block_loop -->|"[B, 512, 1536]"| collapse
    collapse -->|"[B, 512, 768]"| final_norm
    final_norm --> lm_head
    final_norm --> mtp
    lm_head --> ce
    mtp --> mtp_loss
    ce --> total_loss
    mtp_loss --> total_loss

    style INPUT fill:#f0f0f0,stroke:#333
    style EMBED fill:#e8f4fd,stroke:#2196F3
    style MHC_EXPAND fill:#fff3e0,stroke:#FF9800
    style BLOCKS fill:#e8f5e9,stroke:#4CAF50
    style MHC_COLLAPSE fill:#fff3e0,stroke:#FF9800
    style FINAL fill:#f3e5f5,stroke:#9C27B0
    style LOSS fill:#ffebee,stroke:#f44336
```

## Single Transformer Block (Detailed)

Each of the 16 blocks follows this structure. The key innovation is that
**residual connections flow through mHC doubly stochastic mixing matrices**
instead of simple addition, preventing signal amplification across depth.

```mermaid
flowchart TD
    x_in["x_exp input<br/>[B, 512, 1536]<br/><i>2-stream expanded representation</i>"]

    subgraph ATTN_SUB["**Attention Sub-Layer**"]
        direction TB

        subgraph ATTN_MHC_PRE["mHC Prepare (stream → single)"]
            attn_prepare["w = sigmoid(pre_logits + pre_bias)<br/>out = w₀·stream₀ + w₁·stream₁<br/>[B, 512, 1536] → [B, 512, 768]<br/><i>Learned weighted mix of streams</i>"]
        end

        attn_norm["RMSNorm<br/><i>Pre-norm stabilization</i>"]

        attn_choice{{"Layer type?"}}

        subgraph STD_ATTN["**Standard GQA Attention**<br/>(layers 0,2,4,6,8,10,12,14)"]
            std_qkv["Q = Wq·x (12 heads × 64d)<br/>K = Wk·x (4 KV heads × 64d)<br/>V = Wv·x (4 KV heads × 64d)<br/><i>Grouped Query Attention: 3 Q heads share 1 KV head</i>"]
            std_rope["RoPE: rotate Q,K by position<br/><i>Relative position encoding via<br/>complex-valued rotation</i>"]
            std_sdpa["Scaled Dot-Product Attention<br/>scores = Q·Kᵀ / √64<br/>+ causal mask (upper triangle = -∞)<br/>attn = softmax(scores) · V"]
            std_out["Output projection: Wo · attn<br/>[B, 512, 768]"]

            std_qkv --> std_rope --> std_sdpa --> std_out
        end

        subgraph WAVE_ATTN["**Haar Wavefield Attention**<br/>(layers 1,3,5,7,9,11,13,15)"]
            wave_scatter["Scatter: project tokens onto<br/>continuous wave field via<br/>bilinear interpolation<br/>[B, 512, 768] → [B, 12, 256, 64]<br/><i>256-point field per head</i>"]
            wave_haar["Haar DWT (6 levels)<br/>Decompose field into<br/>wavelet scale coefficients"]
            wave_scale["Scale-selective filtering:<br/>filtered = haar_coeffs ⊙ haar(field)<br/><i>12×256 learned coefficients control<br/>which scales (coarse↔fine) to amplify</i>"]
            wave_ihaar["Inverse Haar DWT<br/>Reconstruct filtered field"]
            wave_couple["Head coupling:<br/>softmax mixing across 12 heads<br/><i>Heads share information</i>"]
            wave_gather["Gather: interpolate back<br/>to token positions<br/>[B, 12, 256, 64] → [B, 512, 768]"]
            wave_gate["Content gate:<br/>gate = sigmoid(Wg·x)<br/>out = gate ⊙ gathered<br/><i>Learned relevance weighting</i>"]
            wave_out["Output projection: Wo · gated<br/>[B, 512, 768]"]

            wave_scatter --> wave_haar --> wave_scale --> wave_ihaar
            wave_ihaar --> wave_couple --> wave_gather --> wave_gate --> wave_out
        end

        subgraph ATTN_MHC_POST["mHC Apply (residual + stream mix)"]
            attn_apply["Add residual to each stream:<br/>s₀' = s₀ + h_post₀ · attn_out<br/>s₁' = s₁ + h_post₁ · attn_out<br/><br/>Doubly stochastic mixing:<br/>α = sigmoid(alpha_logit)<br/>out₀ = α·s₀' + (1-α)·s₁'<br/>out₁ = (1-α)·s₀' + α·s₁'<br/>[B, 512, 1536]"]
        end

        attn_prepare --> attn_norm --> attn_choice
        attn_choice -->|"Even layer"| std_qkv
        attn_choice -->|"Odd layer"| wave_scatter
        std_out --> attn_apply
        wave_out --> attn_apply
    end

    subgraph FFN_SUB["**FFN Sub-Layer**"]
        direction TB

        subgraph FFN_MHC_PRE["mHC Prepare (stream → single)"]
            ffn_prepare["w = sigmoid(pre_logits + pre_bias)<br/>out = w₀·stream₀ + w₁·stream₁<br/>[B, 512, 1536] → [B, 512, 768]"]
        end

        ffn_norm["RMSNorm"]

        subgraph SWIGLU["**SwiGLU Feed-Forward Network**"]
            ffn_gate["gate = silu(W_gate · x)<br/><i>768 → 2048, swish activation</i>"]
            ffn_up["up = W_up · x<br/><i>768 → 2048, linear</i>"]
            ffn_mul["hidden = gate ⊙ up<br/><i>Element-wise gating</i>"]
            ffn_down["out = W_down · hidden<br/><i>2048 → 768</i>"]

            ffn_gate --> ffn_mul
            ffn_up --> ffn_mul
            ffn_mul --> ffn_down
        end

        subgraph FFN_MHC_POST["mHC Apply (residual + stream mix)"]
            ffn_apply["Same doubly stochastic residual<br/>as attention sub-layer above<br/>(separate learned α, pre, post params)"]
        end

        ffn_prepare --> ffn_norm --> ffn_gate
        ffn_norm --> ffn_up
        ffn_down --> ffn_apply
    end

    x_out["x_exp output<br/>[B, 512, 1536]"]

    x_in --> attn_prepare
    x_in -.->|"skip connection<br/>(through mHC)"| attn_apply
    attn_apply --> ffn_prepare
    attn_apply -.->|"skip connection<br/>(through mHC)"| ffn_apply
    ffn_apply --> x_out

    style ATTN_SUB fill:#e8f5e9,stroke:#4CAF50
    style FFN_SUB fill:#e8f5e9,stroke:#4CAF50
    style STD_ATTN fill:#e3f2fd,stroke:#2196F3
    style WAVE_ATTN fill:#fce4ec,stroke:#E91E63
    style SWIGLU fill:#f3e5f5,stroke:#9C27B0
    style ATTN_MHC_PRE fill:#fff3e0,stroke:#FF9800
    style ATTN_MHC_POST fill:#fff3e0,stroke:#FF9800
    style FFN_MHC_PRE fill:#fff3e0,stroke:#FF9800
    style FFN_MHC_POST fill:#fff3e0,stroke:#FF9800
```

## mHC-lite N=2: Doubly Stochastic Residual Connections

Traditional transformers use simple addition for residuals (`x = x + layer(x)`),
which can cause **signal amplification** — after 64 layers, the residual stream's
amplitude can grow 3000×. mHC-lite constrains the mixing matrix to be
**doubly stochastic** (all rows and columns sum to 1), guaranteeing bounded signal flow.

```mermaid
flowchart LR
    subgraph EXPAND["**Expand** (model entry)"]
        exp_in["x [B, S, 768]"]
        exp_op["Duplicate:<br/>cat([x, x])"]
        exp_out["x_exp [B, S, 1536]<br/>stream₀ | stream₁"]
        exp_in --> exp_op --> exp_out
    end

    subgraph PREPARE["**Prepare** (per sub-layer)"]
        prep_split["Split into<br/>s₀ [B,S,768]<br/>s₁ [B,S,768]"]
        prep_w["w = sigmoid(<br/>pre_logits + pre_bias)<br/>∈ (0,1)²"]
        prep_mix["output = w₀·s₀ + w₁·s₁<br/>[B, S, 768]"]
        prep_split --> prep_mix
        prep_w --> prep_mix
    end

    subgraph LAYER["**Transformer Layer**<br/>(Attention or FFN)"]
        layer_op["layer_out<br/>[B, S, 768]"]
    end

    subgraph APPLY["**Apply** (per sub-layer)"]
        app_post["h_post = 2·sigmoid(<br/>post_logits + post_bias)<br/>∈ (0,2)²<br/><i>Controls residual strength</i>"]
        app_res["s₀' = s₀ + h_post₀ · layer_out<br/>s₁' = s₁ + h_post₁ · layer_out"]
        app_alpha["α = sigmoid(alpha_logit)<br/>∈ (0,1)"]
        app_mix["H_res (doubly stochastic):<br/>┌         ┐ ┌   ┐   ┌   ┐<br/>│  α  1-α │·│s₀'│ = │o₀ │<br/>│ 1-α  α  │·│s₁'│ = │o₁ │<br/>└         ┘ └   ┘   └   ┘<br/><i>Rows sum to 1, cols sum to 1</i><br/><i>Guarantees ‖output‖ ≤ ‖input‖</i>"]
        app_cat["cat([o₀, o₁])<br/>[B, S, 1536]"]

        app_post --> app_res
        app_alpha --> app_mix
        app_res --> app_mix --> app_cat
    end

    subgraph COLLAPSE["**Collapse** (model exit)"]
        col_avg["avg(stream₀, stream₁)<br/>[B, S, 1536] → [B, S, 768]"]
    end

    exp_out --> prep_split
    prep_mix --> layer_op
    layer_op --> app_res
    app_cat --> col_avg

    style EXPAND fill:#fff3e0,stroke:#FF9800
    style PREPARE fill:#fff8e1,stroke:#FFC107
    style LAYER fill:#e8f5e9,stroke:#4CAF50
    style APPLY fill:#fff3e0,stroke:#FF9800
    style COLLAPSE fill:#fff3e0,stroke:#FF9800
```

## Haar Wavefield Attention (Detail)

Instead of computing pairwise token interactions (O(n²) attention),
wavefield attention projects tokens onto a **continuous wave field**,
applies **Haar wavelet filtering** in the transform domain, then gathers
results back. This gives each head **scale-selective** filtering — it can
independently amplify or suppress coarse (global) vs fine (local) features.

```mermaid
flowchart TD
    input["Input x [B, 512, 768]"]

    subgraph SCATTER["**1. Scatter** — tokens → wave field"]
        s_proj["Linear projection<br/>768 → 12 heads × 64d"]
        s_interp["Bilinear interpolation<br/>512 token positions mapped to<br/>256-point continuous field<br/><i>pos = linspace(0, 255, 512)</i>"]
        s_field["Wave fields<br/>[B, 12, 256, 64]<br/><i>12 heads, 256 field points, 64 features</i>"]
        s_proj --> s_interp --> s_field
    end

    subgraph HAAR["**2. Haar Wavelet Transform** (6 levels)"]
        direction TB
        h_fwd["Forward Haar DWT<br/><i>Decompose into scale bands:</i><br/>Level 1: 128 detail coefficients (finest)<br/>Level 2: 64 detail coefficients<br/>Level 3: 32 detail coefficients<br/>Level 4: 16 detail coefficients<br/>Level 5: 8 detail coefficients<br/>Level 6: 4 detail + 4 approx (coarsest)"]
        h_coeffs["Learned scale coefficients<br/>kernel_haar_coeffs [12, 256]<br/><i>Per-head, per-coefficient weight<br/>Controls which scales matter</i>"]
        h_filter["Element-wise multiply<br/>filtered = coeffs ⊙ haar(field)<br/><i>Amplify important scales,<br/>suppress irrelevant ones</i>"]
        h_inv["Inverse Haar DWT<br/><i>Reconstruct filtered field</i><br/>[B, 12, 256, 64]"]

        h_fwd --> h_filter
        h_coeffs --> h_filter
        h_filter --> h_inv
    end

    subgraph COUPLING["**3. Head Coupling**"]
        coup["Softmax mixing matrix [12×12]<br/>Each head receives weighted sum<br/>of all heads' outputs<br/><i>Cross-head information sharing</i>"]
    end

    subgraph GATHER["**4. Gather** — wave field → tokens"]
        g_interp["Bilinear interpolation<br/>256-point field → 512 token values"]
        g_gate["Content gate:<br/>gate = sigmoid(W_gate · x)<br/>gated = gate ⊙ gathered<br/><i>Per-element relevance mask</i>"]
        g_proj["Output projection<br/>12×64 → 768"]
        g_interp --> g_gate --> g_proj
    end

    output["Output [B, 512, 768]"]

    input --> s_proj
    s_field --> h_fwd
    h_inv --> coup --> g_interp
    g_proj --> output

    style SCATTER fill:#e3f2fd,stroke:#2196F3
    style HAAR fill:#fce4ec,stroke:#E91E63
    style COUPLING fill:#f3e5f5,stroke:#9C27B0
    style GATHER fill:#e8f5e9,stroke:#4CAF50
```

## Training Configuration

```mermaid
flowchart LR
    subgraph DATA["**Dataset**"]
        dataset["Rust source code<br/>~36M tokens (144MB)<br/>BPE tokenizer vocab=4096"]
    end

    subgraph QUANT["**Quantization-Aware Training**"]
        bitlinear["BitLinearSTE<br/><i>Forward: quantize weights to<br/>ternary {-1, 0, +1} via absmean<br/>+ activations to INT8 per-token</i><br/><br/><i>Backward: straight-through estimator<br/>(gradients pass through quantization)</i><br/><br/>Group size: 128"]
    end

    subgraph OPT["**Dual Optimizer**"]
        muon["**Muon** (lr=0.015)<br/>For: all linear weight matrices<br/><i>Newton-Schulz orthogonalization (5 iter)<br/>Momentum β=0.95<br/>Better for high-rank 2D updates</i>"]
        lion["**Lion** (lr=0.0001)<br/>For: embeddings, norms, mHC params<br/><i>Sign-based update<br/>β₁=0.9, β₂=0.99<br/>Better for small/1D params</i>"]
    end

    subgraph SCHED["**WSD LR Schedule**"]
        warmup["Warmup<br/>steps 0→500<br/>linear 0 → lr"]
        stable["Stable<br/>steps 500→24000<br/>constant lr"]
        decay["Cosine Decay<br/>steps 24000→30000<br/>lr → 0.1×lr"]
        warmup --> stable --> decay
    end

    subgraph AUX["**Auxiliary Objectives**"]
        mtp_obj["Multi-Token Prediction<br/>3 future tokens, weight=0.2<br/><i>Denser gradients per sample<br/>~20% better data efficiency</i>"]
        label_smooth["Label Smoothing ε=0.1<br/><i>Prevents overconfident predictions<br/>Regularizes softmax outputs</i>"]
    end

    DATA --> QUANT
    QUANT --> muon
    QUANT --> lion
    muon --> SCHED
    lion --> SCHED
    DATA --> AUX

    style DATA fill:#f0f0f0,stroke:#333
    style QUANT fill:#e0f7fa,stroke:#00BCD4
    style OPT fill:#e8f5e9,stroke:#4CAF50
    style SCHED fill:#fff3e0,stroke:#FF9800
    style AUX fill:#f3e5f5,stroke:#9C27B0
```

## Parameter Breakdown

```mermaid
pie title "~280M Parameters by Component"
    "Attention projections (Wq,Wk,Wv,Wo)" : 113
    "FFN projections (gate,up,down)" : 96
    "Wavefield projections + coeffs" : 40
    "Token embedding (tied)" : 3
    "MTP heads ×3" : 9
    "RMSNorm (32 layers)" : 0.05
    "mHC params (160 total)" : 0.001
```

## Key Innovations

| Component | What it does | Why it matters |
|-----------|-------------|----------------|
| **mHC-lite** | Doubly stochastic residual mixing via BvN decomposition | Prevents signal amplification across 16 layers (traditional residuals can grow 3000× at depth 64) |
| **Haar Wavefield** | Wavelet-domain filtering instead of O(n²) attention | Scale-selective: independently control coarse (global context) vs fine (local syntax) features |
| **BitLinear STE** | Ternary weight quantization during training | Model learns to work with {-1, 0, +1} weights, enabling 16× compression at inference |
| **Muon + Lion** | Orthogonalized updates for matrices, sign-based for scalars | Better optimization landscape navigation for each parameter type |
| **Multi-Token Prediction** | Predict next 1, 2, 3 tokens simultaneously | 20% more learning signal per training sample |
| **Head Coupling** | Softmax mixing matrix across wavefield heads | Heads specialize on different frequency bands then share findings |
