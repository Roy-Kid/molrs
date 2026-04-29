//! Integration tests for the Gaussian Cube reader.
//!
//! Fixtures live under `tests-data/cube/` and exercise the format's quirks:
//!
//!   cube/valtest.cube      – 5×1×1 hand-crafted threshold test (5 atoms)
//!   cube/grid20.cube       – 20³, 16 atoms, **Bohr** units
//!   cube/grid20ang.cube    – 20³, 16 atoms, **Å** units (negative N1)
//!   cube/grid20mo6-8.cube  – 20³, 7 atoms, 3 MOs (negative natoms = -7)
//!   cube/grid25mo.cube     – 25³, 7 atoms, 1 MO (negative natoms)
//!
//! The reader normalises every frame's atoms and simbox to **Å**: positive
//! N1 means file is in Bohr → multiply by `BOHR_TO_ANG`. The grid block is
//! `frame.get("grid")` with `set_shape([nx, ny, nz])`.

use molrs_io::cube::read_cube;

fn cube_path(name: &str) -> std::path::PathBuf {
    crate::test_data::get_test_data_path(&format!("cube/{}", name))
}

fn all_cube_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("cube");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read cube test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No cube files in tests-data/cube/");
    paths
}

// ---------------------------------------------------------------------------
// Generic structural checks — every fixture must satisfy these
// ---------------------------------------------------------------------------

/// Every fixture must produce a frame with atoms, a 3-D grid block, and a
/// simbox in Å.
#[test]
fn test_all_cube_files_parse() {
    for path in all_cube_files() {
        let frame = read_cube(&path).unwrap_or_else(|e| panic!("{:?}: {}", path, e));

        let atoms = frame
            .get("atoms")
            .unwrap_or_else(|| panic!("{:?}: missing atoms block", path));
        assert!(
            atoms.nrows().unwrap_or(0) > 0,
            "{:?}: atoms block is empty",
            path
        );

        let grid = frame
            .get("grid")
            .unwrap_or_else(|| panic!("{:?}: missing grid block", path));
        let shape = grid.shape();
        assert_eq!(shape.len(), 3, "{:?}: grid not 3-D", path);
        assert!(
            shape.iter().all(|&n| n > 0),
            "{:?}: grid has zero dim",
            path
        );

        assert!(
            frame.simbox.is_some(),
            "{:?}: simbox must be present (always Å)",
            path
        );
    }
}

// ---------------------------------------------------------------------------
// Per-fixture assertions
// ---------------------------------------------------------------------------

/// `valtest.cube`: header declares 1×1×5 voxels with 2 atoms. Density
/// sequence is the canonical `[-1e2, -1e-2, 0, 1e-2, 1e2]` sentinel,
/// stored row-major (ix outermost) so for a 1×1×5 grid the data slice
/// is exactly the file order.
#[test]
fn test_valtest_density_sequence() {
    let frame = read_cube(cube_path("valtest.cube")).expect("read valtest.cube");
    let grid = frame.get("grid").expect("grid block");
    assert_eq!(grid.shape(), vec![1, 1, 5], "valtest is 1×1×5");

    let density = grid
        .get_float("density")
        .expect("density column")
        .iter()
        .copied()
        .collect::<Vec<f64>>();
    let expected = [-1e2_f64, -1e-2, 0.0, 1e-2, 1e2];
    for (i, (got, &want)) in density.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "valtest density[{}] = {} (want {})",
            i,
            got,
            want
        );
    }

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 2, "valtest has 2 atoms");
}

/// `grid20.cube` (Bohr): the cube spec encodes Bohr by a positive first
/// voxel-count integer. Verify the reader interprets file lengths as Bohr
/// and converts the simbox to Å on the way out (extents ≈ 12.2 × 12.9 ×
/// 12.9, see fixture header).
#[test]
fn test_grid20_bohr_simbox_in_angstrom() {
    let frame = read_cube(cube_path("grid20.cube")).expect("read grid20.cube");
    assert_eq!(
        frame.get("grid").unwrap().shape(),
        vec![20, 20, 20],
        "20³ grid"
    );
    assert_eq!(frame.get("atoms").unwrap().nrows().unwrap(), 16);

    let lens = frame.simbox.as_ref().expect("simbox").lengths();
    // Header voxel vec ≈ (1.15, 1.22, 1.20) Bohr × 20 ≈ (23.0, 24.3, 24.0)
    // Bohr → ≈ (12.2, 12.9, 12.7) Å.
    let expected = [12.2_f64, 12.9, 12.7];
    for i in 0..3 {
        assert!(
            (lens[i] - expected[i]).abs() < 0.5,
            "axis {}: Bohr→Å length {:.3} far from {:.3}",
            i,
            lens[i],
            expected[i]
        );
    }
}

/// `grid20ang.cube` (Å, encoded by negative first voxel count): the
/// reader keeps the simbox at the file values without a Bohr→Å rescale.
#[test]
fn test_grid20ang_no_unit_rescale() {
    let frame = read_cube(cube_path("grid20ang.cube")).expect("read grid20ang.cube");
    assert_eq!(
        frame.get("grid").unwrap().shape(),
        vec![20, 20, 20],
        "20³ grid"
    );
    assert_eq!(frame.get("atoms").unwrap().nrows().unwrap(), 16);

    let lens = frame.simbox.as_ref().expect("simbox").lengths();
    // Same numerical voxel vectors but interpreted as Å → 20× ≈ (23, 24.3, 24).
    let expected = [23.0_f64, 24.3, 24.0];
    for i in 0..3 {
        assert!(
            (lens[i] - expected[i]).abs() < 1.0,
            "axis {}: Å length {:.3} far from {:.3}",
            i,
            lens[i],
            expected[i]
        );
    }
}

/// `grid20mo6-8.cube`: negative-natoms multi-orbital file. Three orbitals
/// (6, 7, 8) → three columns `mo_6`, `mo_7`, `mo_8`, each 20³ values.
#[test]
fn test_grid20mo_three_orbital_columns() {
    let frame = read_cube(cube_path("grid20mo6-8.cube")).expect("read grid20mo6-8.cube");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 7, "7 atoms");

    let grid = frame.get("grid").expect("grid block");
    assert_eq!(grid.shape(), vec![20, 20, 20], "20³");

    for idx in [6_usize, 7, 8] {
        let key = format!("mo_{}", idx);
        let col = grid
            .get_float(&key)
            .unwrap_or_else(|| panic!("missing column {}", key));
        assert_eq!(col.len(), 20 * 20 * 20, "{} length", key);
    }
}

/// `grid25mo.cube`: single-orbital negative-natoms file, 25³.
#[test]
fn test_grid25mo_single_orbital() {
    let frame = read_cube(cube_path("grid25mo.cube")).expect("read grid25mo.cube");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 7, "7 atoms");

    let grid = frame.get("grid").expect("grid block");
    assert_eq!(grid.shape(), vec![25, 25, 25], "25³");

    // Find the single mo_* column. The orbital index varies per fixture;
    // we don't pin it here because the fixture's MO id is implementation
    // detail of the producing toolchain.
    let mo_keys: Vec<String> = grid
        .keys()
        .filter(|k| k.starts_with("mo_"))
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        mo_keys.len(),
        1,
        "exactly one mo_* column, got {:?}",
        mo_keys
    );
    let col = grid.get_float(&mo_keys[0]).expect("mo column");
    assert_eq!(col.len(), 25 * 25 * 25);
}
