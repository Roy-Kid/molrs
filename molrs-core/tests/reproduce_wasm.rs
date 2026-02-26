#![cfg(all(feature = "zarr", feature = "slow-tests"))]

use molrs::io::zarr::ZarrStoreReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::WritableStorageTraits;
use zarrs::storage::store::MemoryStore;

fn visit_dirs(dir: &Path, cb: &mut dyn FnMut(&Path)) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(&path);
            }
        }
    }
    Ok(())
}

#[test]
fn test_reproduce_wasm_loading() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let zarr_root = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("molrs-md/tip3p_traj.zarr");

    println!("Simulating generic loading from: {:?}", zarr_root);

    let store = Arc::new(MemoryStore::new());

    // Simulate what standard generic loading + WASM adapter does:
    // 1. Traverse directory
    // 2. Read file bytes
    // 3. Insert into MemoryStore with keys relative to root

    let root_path = zarr_root.clone();
    visit_dirs(&zarr_root, &mut |path| {
        let relative_path = path.strip_prefix(&root_path).unwrap();
        // WASM adapter bug fix: Zarrs StoreKey must NOT start with /
        let key_str = relative_path.to_string_lossy().to_string();

        let content = std::fs::read(path).expect("failed to read file");

        if key_str.contains("frame/atoms/x") {
            println!("Skipping {}", key_str);
            return;
        }
        // println!("Storing key: {}", key_str);

        // Zarrs 0.16+ StoreKey handling
        let key = zarrs::storage::StoreKey::new(&key_str).expect("invalid key");
        store
            .set(&key, content.into())
            .expect("failed to set store value");
    })
    .expect("failed to visit dir");

    // Now try to open with ZarrStoreReader
    println!("Store populated. Opening ZarrStoreReader...");

    let reader = ZarrStoreReader::open_store(store as ReadableWritableListableStorage)
        .expect("Failed to open Zarr trajectory from MemoryStore");

    let n_frames = reader.len();
    println!("Number of frames: {}", n_frames);
    assert!(n_frames > 0, "Trajectory should have format frames");

    // Read first frame
    let frame0 = reader.read_frame(0).expect("Failed to read frame 0");

    if let Some(atoms) = frame0.get("atoms") {
        let n_atoms = atoms.nrows().unwrap_or(0);
        println!("Number of atoms: {}", n_atoms);
        assert!(n_atoms > 0, "Frame should have atoms");
    } else {
        panic!("No atoms block found in frame 0");
    }
}
