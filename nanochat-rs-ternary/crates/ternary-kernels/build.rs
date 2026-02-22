use std::process::Command;

fn main() {
    // Declare has_cuda as a valid cfg so #[cfg(has_cuda)] doesn't warn.
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    // Allow overriding the target architecture via TERNARY_MARCH env var.
    // Default: "native" (best perf on build machine, NOT portable).
    // For portable builds: set TERNARY_MARCH=x86-64 (or x86-64-v3 for AVX2 baseline).
    // The C code uses runtime CPUID dispatch for SIMD kernels, so -march=x86-64
    // is safe — it just may produce slower non-kernel helper code.
    let march = std::env::var("TERNARY_MARCH").unwrap_or_else(|_| "native".to_string());
    let march_flag = format!("-march={}", march);
    println!("cargo:rerun-if-env-changed=TERNARY_MARCH");

    // CPU kernels (v3.5.0) — All SIMD kernels consolidated in ternary_gemv.c.
    // Supports x86_64 (AVX-512/AVX2/SSSE3) and AArch64 (NEON).
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let mut build = cc::Build::new();
    build
        .file("csrc/ternary_gemv.c")
        .flag("-O3")
        .flag("-fno-strict-aliasing")
        .flag("-DNDEBUG")
        // Rename main() so it doesn't conflict with Rust binary
        .flag("-Dmain=ternary_gemv_main")
        // Suppress warnings from battle-tested C code
        .flag("-w");

    if target_arch == "aarch64" {
        // ARM64: NEON is baseline, no special -march needed.
        // Function-level target attributes are not needed since NEON is always available.
    } else {
        // x86_64: Use -march flag for non-SIMD helper code.
        // SIMD kernels use function-level target attributes for runtime dispatch.
        build.flag(&march_flag);
    }

    build.compile("ternary_gemv");

    println!("cargo:rerun-if-changed=csrc/ternary_gemv.c");
    println!("cargo:rerun-if-changed=csrc/ternary_gemv.h");
    println!("cargo:rerun-if-changed=csrc/ternary_dp4a.cu");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // GPU kernel (optional — gate on CUDA toolkit presence)
    let cuda_path = if std::env::var("CUDA_PATH").is_ok() {
        std::env::var("CUDA_PATH").unwrap()
    } else if std::path::Path::new("/usr/local/cuda").exists() {
        "/usr/local/cuda".to_string()
    } else {
        return; // No CUDA, skip GPU kernel
    };

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    if !std::path::Path::new(&nvcc).exists() {
        println!(
            "cargo:warning=CUDA detected but nvcc not found at {}; skipping CUDA kernel build",
            nvcc
        );
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let obj_path = format!("{}/ternary_dp4a.o", out_dir);
    let lib_path = format!("{}/libternary_dp4a.a", out_dir);

    // Compile CUDA kernel to object file
    let cuda_arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    let status = match Command::new(&nvcc)
        .args([
            "-c",
            "csrc/ternary_dp4a.cu",
            "-o",
            &obj_path,
            "-O3",
            &format!("--gpu-architecture={}", cuda_arch),
            "-Xcompiler",
            "-fPIC",
            "-DNDEBUG",
        ])
        .status()
    {
        Ok(status) => status,
        Err(err) => {
            println!(
                "cargo:warning=Failed to run nvcc at {}: {}; skipping CUDA kernel build",
                nvcc, err
            );
            return;
        }
    };

    if !status.success() {
        println!("cargo:warning=nvcc compilation of ternary_dp4a.cu failed; skipping CUDA kernel build");
        return;
    }

    // Archive into static library
    let status = match Command::new("ar").args(["rcs", &lib_path, &obj_path]).status() {
        Ok(status) => status,
        Err(err) => {
            println!(
                "cargo:warning=Failed to run ar for CUDA static library: {}; skipping CUDA kernel build",
                err
            );
            return;
        }
    };

    if !status.success() {
        println!("cargo:warning=ar failed to create libternary_dp4a.a; skipping CUDA kernel build");
        return;
    }

    // Use a custom cfg flag instead of feature="cuda" to avoid conflicting
    // with Cargo's feature system. Rust source gates on:
    //   #[cfg(any(feature = "cuda", has_cuda))]
    println!("cargo:rustc-cfg=has_cuda");

    // Link the CUDA library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=ternary_dp4a");

    // Link CUDA runtime
    let cuda_lib_dir = format!("{}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
