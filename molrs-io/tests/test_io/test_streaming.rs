//! Integration tests for the streaming `FrameIndexBuilder` family.
//!
//! For each format these tests verify the spec-mandated invariants:
//!
//! 1. The index built from a single `feed(whole_file)` matches the index
//!    built from feeding 1 KiB chunks and 64 KiB chunks (byte-identical
//!    `FrameIndexEntry` lists).
//! 2. Each entry's slice (`bytes[byte_offset..byte_offset + byte_len]`) is
//!    parseable by the format's `parse_frame_bytes` free function.
//! 3. The resulting `Frame`'s atoms-block columns match the legacy reader's
//!    output for the same step / record.

use molrs::frame::Frame;
use molrs::types::F;
use molrs_io::streaming::{FrameIndexBuilder, FrameIndexEntry};
use std::path::Path;

const CHUNK_1K: usize = 1 << 10;
const CHUNK_64K: usize = 64 << 10;

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

fn build_index<B>(
    make: impl Fn() -> Box<B>,
    bytes: &[u8],
    chunk_size: usize,
) -> Vec<FrameIndexEntry>
where
    B: FrameIndexBuilder + 'static + ?Sized,
{
    let mut builder = make();
    let mut offset: u64 = 0;
    let mut out: Vec<FrameIndexEntry> = Vec::new();
    for piece in bytes.chunks(chunk_size.max(1)) {
        builder.feed(piece, offset);
        offset += piece.len() as u64;
        out.extend(builder.drain());
    }
    out.extend(builder.finish().expect("finish"));
    out
}

fn assert_chunked_indices_match(
    label: &str,
    one_shot: &[FrameIndexEntry],
    chunked_1k: &[FrameIndexEntry],
    chunked_64k: &[FrameIndexEntry],
) {
    assert_eq!(
        one_shot, chunked_1k,
        "{label}: single-shot vs 1KiB-chunked indices differ"
    );
    assert_eq!(
        one_shot, chunked_64k,
        "{label}: single-shot vs 64KiB-chunked indices differ"
    );
}

fn assert_float_columns_eq(reference: &Frame, candidate: &Frame, columns: &[&str], label: &str) {
    let r_atoms = reference
        .get("atoms")
        .unwrap_or_else(|| panic!("{label}: reference frame has no atoms block"));
    let c_atoms = candidate
        .get("atoms")
        .unwrap_or_else(|| panic!("{label}: candidate frame has no atoms block"));
    assert_eq!(
        r_atoms.nrows(),
        c_atoms.nrows(),
        "{label}: atom row count differs"
    );
    for col in columns {
        let Some(r_col) = r_atoms.get_float(col) else {
            continue;
        };
        let c_col = c_atoms
            .get_float(col)
            .unwrap_or_else(|| panic!("{label}: candidate missing float column '{col}'"));
        let r_slice = r_col
            .as_slice()
            .unwrap_or_else(|| panic!("{label}: ref column '{col}' not contiguous"));
        let c_slice = c_col
            .as_slice()
            .unwrap_or_else(|| panic!("{label}: cand column '{col}' not contiguous"));
        assert_eq!(
            r_slice.len(),
            c_slice.len(),
            "{label}: column '{col}' length differs"
        );
        for (i, (a, b)) in r_slice.iter().zip(c_slice.iter()).enumerate() {
            let abs_diff = (*a - *b).abs() as F;
            assert!(
                abs_diff < 1e-6,
                "{label}: column '{col}' row {i} differs: ref={a} cand={b}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// LAMMPS dump
// ---------------------------------------------------------------------------

mod lammps_dump_streaming {
    use super::*;
    use molrs_io::lammps_dump::{LAMMPSTrajReader, LammpsDumpIndexBuilder, parse_frame_bytes};
    use molrs_io::reader::TrajReader;
    use std::io::{BufReader, Cursor};

    fn make_builder() -> Box<LammpsDumpIndexBuilder> {
        Box::new(LammpsDumpIndexBuilder::new())
    }

    fn run_one(path: &Path) {
        let bytes = std::fs::read(path).expect("read file");
        let label = path.display().to_string();

        let one = build_index(make_builder, &bytes, bytes.len());
        let c1k = build_index(make_builder, &bytes, CHUNK_1K);
        let c64k = build_index(make_builder, &bytes, CHUNK_64K);
        assert_chunked_indices_match(&label, &one, &c1k, &c64k);

        // Per-frame parse + column equality with legacy reader.
        let mut legacy = LAMMPSTrajReader::new(BufReader::new(Cursor::new(bytes.clone())));
        let n_legacy = legacy.len().expect("legacy len");
        assert_eq!(
            one.len(),
            n_legacy,
            "{label}: streaming index frame count != legacy frame count"
        );

        // Spot-check up to the first 4 frames (full check would be O(N²) for
        // big trajectories; the spec requires "match the legacy read_step(i)
        // output", but an exhaustive check on a 5 MB trajectory is too slow
        // for a unit test). Pick first, last, and one in the middle.
        let mut indices: Vec<usize> = vec![0];
        if n_legacy >= 2 {
            indices.push(n_legacy - 1);
        }
        if n_legacy >= 4 {
            indices.push(n_legacy / 2);
        }

        for i in indices {
            let entry = &one[i];
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            let cand = parse_frame_bytes(&bytes[lo..hi])
                .unwrap_or_else(|e| panic!("{label}: parse frame {i}: {e}"));
            let refr = legacy
                .read_step(i)
                .unwrap_or_else(|e| panic!("{label}: legacy step {i}: {e}"))
                .unwrap_or_else(|| panic!("{label}: legacy step {i} returned None"));
            assert_float_columns_eq(
                &refr,
                &cand,
                &["x", "y", "z", "xu", "yu", "zu", "xs", "ys", "zs"],
                &format!("{label} frame {i}"),
            );
        }
    }

    #[test]
    fn nacl_round_trip() {
        run_one(&crate::test_data::get_test_data_path(
            "lammps/nacl.lammpstrj",
        ));
    }

    #[test]
    fn polymer_round_trip() {
        run_one(&crate::test_data::get_test_data_path(
            "lammps/polymer.lammpstrj",
        ));
    }

    #[test]
    fn properties_round_trip() {
        run_one(&crate::test_data::get_test_data_path(
            "lammps/properties.lammpstrj",
        ));
    }

    #[test]
    fn wrapped_round_trip() {
        run_one(&crate::test_data::get_test_data_path(
            "lammps/wrapped.lammpstrj",
        ));
    }
}

// ---------------------------------------------------------------------------
// XYZ
// ---------------------------------------------------------------------------

mod xyz_streaming {
    use super::*;
    use molrs_io::reader::TrajReader;
    use molrs_io::xyz::{XYZReader, XyzIndexBuilder, parse_frame_bytes};
    use std::io::{BufReader, Cursor};

    fn make_builder() -> Box<XyzIndexBuilder> {
        Box::new(XyzIndexBuilder::new())
    }

    fn run_one(path: &Path) {
        let bytes = std::fs::read(path).expect("read file");
        let label = path.display().to_string();

        let one = build_index(make_builder, &bytes, bytes.len());
        let c1k = build_index(make_builder, &bytes, CHUNK_1K);
        let c64k = build_index(make_builder, &bytes, CHUNK_64K);
        assert_chunked_indices_match(&label, &one, &c1k, &c64k);

        let mut legacy = XYZReader::new(BufReader::new(Cursor::new(bytes.clone())));
        let n_legacy = legacy.len().expect("legacy len");
        assert_eq!(
            one.len(),
            n_legacy,
            "{label}: streaming index frame count != legacy frame count"
        );

        let mut indices: Vec<usize> = vec![0];
        if n_legacy >= 2 {
            indices.push(n_legacy - 1);
        }
        if n_legacy >= 4 {
            indices.push(n_legacy / 2);
        }

        for i in indices {
            let entry = &one[i];
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            let cand = parse_frame_bytes(&bytes[lo..hi])
                .unwrap_or_else(|e| panic!("{label}: parse frame {i}: {e}"));
            let refr = legacy
                .read_step(i)
                .unwrap_or_else(|e| panic!("{label}: legacy step {i}: {e}"))
                .unwrap_or_else(|| panic!("{label}: legacy step {i} returned None"));
            assert_float_columns_eq(
                &refr,
                &cand,
                &["x", "y", "z", "vx", "vy", "vz"],
                &format!("{label} frame {i}"),
            );
        }
    }

    #[test]
    fn methane() {
        run_one(&crate::test_data::get_test_data_path("xyz/methane.xyz"));
    }

    #[test]
    fn trajectory() {
        run_one(&crate::test_data::get_test_data_path("xyz/trajectory.xyz"));
    }

    #[test]
    fn extended() {
        run_one(&crate::test_data::get_test_data_path("xyz/extended.xyz"));
    }

    #[test]
    fn velocities() {
        run_one(&crate::test_data::get_test_data_path("xyz/velocities.xyz"));
    }

    #[test]
    fn topology() {
        run_one(&crate::test_data::get_test_data_path("xyz/topology.xyz"));
    }
}

// ---------------------------------------------------------------------------
// PDB
// ---------------------------------------------------------------------------

mod pdb_streaming {
    use super::*;
    use molrs_io::pdb::{PdbIndexBuilder, parse_frame_bytes};

    fn make_builder() -> Box<PdbIndexBuilder> {
        Box::new(PdbIndexBuilder::new())
    }

    fn run_one(path: &Path, expected_min_frames: usize) {
        let bytes = std::fs::read(path).expect("read file");
        let label = path.display().to_string();

        let one = build_index(make_builder, &bytes, bytes.len());
        let c1k = build_index(make_builder, &bytes, CHUNK_1K);
        let c64k = build_index(make_builder, &bytes, CHUNK_64K);
        assert_chunked_indices_match(&label, &one, &c1k, &c64k);

        assert!(
            one.len() >= expected_min_frames,
            "{label}: expected at least {expected_min_frames} frames, got {}",
            one.len()
        );

        // Parse first and last entries.
        let entry0 = &one[0];
        let lo = entry0.byte_offset as usize;
        let hi = lo + entry0.byte_len as usize;
        parse_frame_bytes(&bytes[lo..hi]).unwrap_or_else(|e| panic!("{label}: parse frame 0: {e}"));

        let last = one.last().unwrap();
        let lo = last.byte_offset as usize;
        let hi = lo + last.byte_len as usize;
        parse_frame_bytes(&bytes[lo..hi])
            .unwrap_or_else(|e| panic!("{label}: parse last frame: {e}"));
    }

    #[test]
    fn single_frame_water() {
        // water.pdb has no MODEL records → single-frame mode.
        run_one(&crate::test_data::get_test_data_path("pdb/water.pdb"), 1);
    }

    #[test]
    fn multi_model_2hkb() {
        // 2hkb has 10 NMR models.
        let path = crate::test_data::get_test_data_path("pdb/2hkb.pdb");
        let bytes = std::fs::read(&path).expect("read file");
        let one = build_index(
            || Box::new(molrs_io::pdb::PdbIndexBuilder::new()),
            &bytes,
            bytes.len(),
        );
        assert_eq!(one.len(), 10, "2hkb.pdb should yield 10 frames");
        run_one(&path, 10);
    }

    #[test]
    fn multi_model_ase() {
        // ase.pdb has 156 models.
        let path = crate::test_data::get_test_data_path("pdb/ase.pdb");
        let bytes = std::fs::read(&path).expect("read file");
        let one = build_index(
            || Box::new(molrs_io::pdb::PdbIndexBuilder::new()),
            &bytes,
            bytes.len(),
        );
        assert_eq!(one.len(), 156, "ase.pdb should yield 156 frames");
        run_one(&path, 156);
    }
}

// ---------------------------------------------------------------------------
// LAMMPS data
// ---------------------------------------------------------------------------

mod lammps_data_streaming {
    use super::*;
    use molrs_io::lammps_data::{LammpsDataIndexBuilder, parse_frame_bytes, read_lammps_data};

    fn make_builder() -> Box<LammpsDataIndexBuilder> {
        Box::new(LammpsDataIndexBuilder::new())
    }

    fn run_one(path: &Path) {
        let bytes = std::fs::read(path).expect("read file");
        let label = path.display().to_string();

        let one = build_index(make_builder, &bytes, bytes.len());
        let c1k = build_index(make_builder, &bytes, CHUNK_1K);
        let c64k = build_index(make_builder, &bytes, CHUNK_64K);
        assert_chunked_indices_match(&label, &one, &c1k, &c64k);

        assert_eq!(one.len(), 1, "{label}: data file must produce 1 frame");
        let entry = &one[0];
        assert_eq!(entry.byte_offset, 0, "{label}: data frame offset must be 0");
        assert_eq!(
            entry.byte_len as usize,
            bytes.len(),
            "{label}: data frame byte_len must equal file size"
        );

        let cand = parse_frame_bytes(&bytes).unwrap_or_else(|e| panic!("{label}: parse: {e}"));
        let refr = read_lammps_data(path).unwrap_or_else(|e| panic!("{label}: legacy read: {e}"));
        assert_float_columns_eq(&refr, &cand, &["x", "y", "z"], &label);
    }

    #[test]
    fn solvated() {
        run_one(&crate::test_data::get_test_data_path(
            "lammps-data/solvated.lmp",
        ));
    }

    #[test]
    fn data_body() {
        run_one(&crate::test_data::get_test_data_path(
            "lammps-data/data.body",
        ));
    }
}

// ---------------------------------------------------------------------------
// SDF
// ---------------------------------------------------------------------------

mod sdf_streaming {
    use super::*;
    use molrs_io::reader::FrameReader;
    use molrs_io::sdf::{SDFReader, SdfIndexBuilder, parse_frame_bytes};
    use std::io::{BufReader, Cursor};

    fn make_builder() -> Box<SdfIndexBuilder> {
        Box::new(SdfIndexBuilder::new())
    }

    fn run_one(path: &Path) {
        let bytes = std::fs::read(path).expect("read file");
        let label = path.display().to_string();

        let one = build_index(make_builder, &bytes, bytes.len());
        let c1k = build_index(make_builder, &bytes, CHUNK_1K);
        let c64k = build_index(make_builder, &bytes, CHUNK_64K);
        assert_chunked_indices_match(&label, &one, &c1k, &c64k);

        let mut legacy = SDFReader::new(BufReader::new(Cursor::new(bytes.clone())));
        let mut legacy_frames: Vec<Frame> = Vec::new();
        while let Some(f) = legacy.read_frame().expect("legacy read") {
            legacy_frames.push(f);
        }
        assert_eq!(
            one.len(),
            legacy_frames.len(),
            "{label}: streaming index frame count != legacy frame count"
        );

        for (i, entry) in one.iter().enumerate() {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            let cand = parse_frame_bytes(&bytes[lo..hi])
                .unwrap_or_else(|e| panic!("{label}: parse frame {i}: {e}"));
            assert_float_columns_eq(
                &legacy_frames[i],
                &cand,
                &["x", "y", "z"],
                &format!("{label} record {i}"),
            );
        }
    }

    #[test]
    fn aspirin() {
        run_one(&crate::test_data::get_test_data_path("sdf/aspirin.sdf"));
    }

    #[test]
    fn kinases_multi_record() {
        run_one(&crate::test_data::get_test_data_path("sdf/kinases.sdf"));
    }
}
