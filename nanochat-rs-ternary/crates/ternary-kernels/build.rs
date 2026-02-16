use std::process::Command;

fn main() {
    // Declare has_cuda as a valid cfg so #[cfg(has_cuda)] doesn't warn.
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    // AVX2 kernel — compiled separately with AVX2+FMA target.
    // NOTE: -march=native means the resulting binary is NOT portable to machines
    // with a different (older) CPU microarchitecture. For portable builds,
    // replace with explicit -mavx2 -mfma or use function multi-versioning.
    cc::Build::new()
        .file("csrc/ternary_gemv_avx2.c")
        .flag("-O3")
        .flag("-march=native")
        .flag("-fno-strict-aliasing")
        .flag("-DNDEBUG")
        .flag("-w")
        .compile("ternary_gemv_avx2");

    // CPU kernels — compile with native arch detection.
    // NOTE: -march=native means the resulting binary is NOT portable to machines
    // with a different (older) CPU microarchitecture. The C code uses runtime
    // CPUID dispatch so it will not emit illegal instructions, but the compiler
    // may still use native-only optimizations in non-kernel helper functions.
    // For portable builds, replace with -march=x86-64 (baseline) and rely
    // solely on the runtime dispatch + function-level target attributes.
    cc::Build::new()
        .file("csrc/ternary_gemv.c")
        .flag("-O3")
        .flag("-march=native")
        .flag("-fno-strict-aliasing")
        .flag("-DNDEBUG")
        // Rename main() so it doesn't conflict with Rust binary
        .flag("-Dmain=ternary_gemv_main")
        // Suppress warnings from battle-tested C code
        .flag("-w")
        .compile("ternary_gemv");

    println!("cargo:rerun-if-changed=csrc/ternary_gemv.c");
    println!("cargo:rerun-if-changed=csrc/ternary_gemv.h");
    println!("cargo:rerun-if-changed=csrc/ternary_gemv_avx2.c");
    println!("cargo:rerun-if-changed=csrc/ternary_gemv_avx2.h");
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

    // Use a custom cfg flag instead of feature="cuda" to avoid conflicting
    // with Cargo's feature system. Rust source gates on:
    //   #[cfg(any(feature = "cuda", has_cuda))]
    println!("cargo:rustc-cfg=has_cuda");

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let obj_path = format!("{}/ternary_dp4a.o", out_dir);
    let lib_path = format!("{}/libternary_dp4a.a", out_dir);

    // Compile CUDA kernel to object file
    let cuda_arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    let status = Command::new(&nvcc)
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
        .expect("Failed to run nvcc");

    if !status.success() {
        panic!("nvcc compilation of ternary_dp4a.cu failed");
    }

    // Archive into static library
    let status = Command::new("ar")
        .args(["rcs", &lib_path, &obj_path])
        .status()
        .expect("Failed to run ar");

    if !status.success() {
        panic!("ar failed to create libternary_dp4a.a");
    }

    // Link the CUDA library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=ternary_dp4a");

    // Link CUDA runtime
    let cuda_lib_dir = format!("{}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
