fn main() {
    // cuBLASLt comes from the CUDA toolkit (CUDA_HOME, or a conda-style env).
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = std::env::var(var) {
            for sub in ["lib", "lib64", "targets/x86_64-linux/lib", "targets/sbsa-linux/lib"] {
                println!("cargo:rustc-link-search=native={root}/{sub}");
            }
        }
    }
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
