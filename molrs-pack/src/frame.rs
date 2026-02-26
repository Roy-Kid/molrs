//! Helpers for converting `molrs_core::Frame` to packing inputs.

use molrs::core::types::F;
use std::str::FromStr;

/// Extract atom positions, VdW radii, and element symbols from a `molrs::Frame`.
///
/// Reads the `"atoms"` block, expecting `"x"`, `"y"`, `"z"` (f32) and `"element"` (String)
/// columns. VdW radii are looked up from `molrs::Element::vdw_radius()`.
/// Unknown elements fall back to 1.5 Å with symbol `"X"`.
///
/// # Panics
/// Panics if the frame has no `"atoms"` block or no `"x"` / `"y"` / `"z"` columns.
pub fn frame_to_coords(frame: &molrs::Frame) -> (Vec<[F; 3]>, Vec<F>) {
    let (positions, radii, _) = frame_to_coords_and_elements(frame);
    (positions, radii)
}

/// Like [`frame_to_coords`] but also returns element symbols.
pub fn frame_to_coords_and_elements(frame: &molrs::Frame) -> (Vec<[F; 3]>, Vec<F>, Vec<String>) {
    let atoms = frame.get("atoms").expect("frame has no 'atoms' block");

    let x = atoms.get_f32("x").expect("atoms block has no 'x' column");
    let y = atoms.get_f32("y").expect("atoms block has no 'y' column");
    let z = atoms.get_f32("z").expect("atoms block has no 'z' column");

    let n = x.len();

    let positions: Vec<[F; 3]> = x
        .iter()
        .zip(y.iter())
        .zip(z.iter())
        .map(|((&xi, &yi), &zi)| [xi as F, yi as F, zi as F])
        .collect();

    let (radii, elements): (Vec<F>, Vec<String>) = if let Some(elems) = atoms.get_string("element")
    {
        elems
            .iter()
            .map(|sym| {
                let s = sym.trim();
                let r = molrs::Element::from_str(s)
                    .map(|e| e.vdw_radius() as F)
                    .unwrap_or(1.5);
                (r, s.to_string())
            })
            .unzip()
    } else {
        (vec![1.5; n], vec!["X".to_string(); n])
    };

    (positions, radii, elements)
}
