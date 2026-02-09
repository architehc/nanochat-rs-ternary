fn main() {
    // Only build CUDA kernels if cuda feature is enabled
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=cuda/sigmoid.cu");

        // Check for CUDA toolkit
        let cuda_path = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        let nvcc = format!("{}/bin/nvcc", cuda_path);

        // Check if nvcc exists
        if !std::path::Path::new(&nvcc).exists() {
            println!("cargo:warning=CUDA feature enabled but nvcc not found at {}", nvcc);
            println!("cargo:warning=Skipping CUDA kernel compilation");
            return;
        }

        // Compile CUDA kernel
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let cuda_file = "cuda/sigmoid.cu";
        let output_lib = format!("{}/libcuda_sigmoid.a", out_dir);

        // Compile to object file
        let obj_file = format!("{}/sigmoid.o", out_dir);
        let status = std::process::Command::new(&nvcc)
            .args(&[
                "-c",
                cuda_file,
                "-o",
                &obj_file,
                "--compiler-options",
                "-fPIC",
                "-O3",
                "--use_fast_math",
                "-arch=sm_89", // RTX 4090 (Ada Lovelace)
            ])
            .status()
            .expect("Failed to run nvcc");

        if !status.success() {
            panic!("nvcc compilation failed");
        }

        // Create static library
        let status = std::process::Command::new("ar")
            .args(&["rcs", &output_lib, &obj_file])
            .status()
            .expect("Failed to run ar");

        if !status.success() {
            panic!("ar failed to create static library");
        }

        // Link the library
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=cuda_sigmoid");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
}
