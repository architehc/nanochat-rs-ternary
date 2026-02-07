fn main() {
    // AVX2 kernel — compiled separately with AVX2+FMA target
    cc::Build::new()
        .file("csrc/ternary_gemv_avx2.c")
        .flag("-O3")
        .flag("-march=native")
        .flag("-fno-strict-aliasing")
        .flag("-DNDEBUG")
        .flag("-w")
        .compile("ternary_gemv_avx2");

    // CPU kernels — compile with native arch detection
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

    // GPU kernel (optional — gate on CUDA toolkit presence)
    if std::env::var("CUDA_PATH").is_ok() || std::path::Path::new("/usr/local/cuda").exists() {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
}
