fn main() {
    #[cfg(feature = "cuda")]
    {
        // 1. CMake build: compile cuda/*.cu + cpp/*.cpp -> libmolrs_core_cuda.a
        let dst = cmake::build(".");

        // 2. Link
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-lib=static=molrs_core_cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=stdc++");

        // 3. bindgen: include/molrs_core.h -> $OUT_DIR/bindings.rs
        let bindings = bindgen::Builder::default()
            .header("include/molrs_core.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("bindgen failed to generate bindings");

        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out.join("bindings.rs"))
            .expect("failed to write bindings");
    }
}
