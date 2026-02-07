fn main() {
    // Declare the has_numa cfg for check-cfg validation
    println!("cargo::rustc-check-cfg=cfg(has_numa)");

    // Link libnuma on Linux if available
    #[cfg(target_os = "linux")]
    {
        // Check if libnuma shared library exists
        let numa_paths = [
            "/usr/lib/x86_64-linux-gnu/libnuma.so",
            "/usr/lib64/libnuma.so",
            "/usr/lib/libnuma.so",
        ];

        let has_numa = numa_paths.iter().any(|p| std::path::Path::new(p).exists());

        if has_numa {
            println!("cargo:rustc-link-lib=numa");
            println!("cargo:rustc-cfg=has_numa");
        }
    }
}
