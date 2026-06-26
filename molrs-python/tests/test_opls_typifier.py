"""Python-binding coverage for the native OPLS-AA typifier.

Mirrors the MMFFTypifier binding: ``molrs.OplsTypifier`` surfaces the validated
Rust ``OplsTypifier`` (embedded canonical OPLS-AA), so molpy can delegate instead
of carrying its own Python OPLS typifier. The typing/assignment logic itself is
exercised in Rust (``ff::typifier::opls``); these tests only assert the PyO3
surface.
"""

import math

import numpy as np
import pytest

import molrs


def _ethane() -> "molrs.Atomistic":
    """Ethane (C2H6) with explicit hydrogens and a plausible geometry."""
    mol = molrs.Atomistic()
    c1 = mol.add_atom("C", 0.0, 0.0, 0.0)
    c2 = mol.add_atom("C", 1.54, 0.0, 0.0)
    hpos = [
        (c1, (-0.36, 1.03, 0.0)),
        (c1, (-0.36, -0.51, 0.89)),
        (c1, (-0.36, -0.51, -0.89)),
        (c2, (1.90, 1.03, 0.0)),
        (c2, (1.90, -0.51, 0.89)),
        (c2, (1.90, -0.51, -0.89)),
    ]
    for c, (x, y, z) in hpos:
        h = mol.add_atom("H", x, y, z)
        mol.add_bond(c, h)
    mol.add_bond(c1, c2)
    return mol


def test_opls_typifier_is_exposed():
    """molrs.OplsTypifier exists and constructs from embedded OPLS-AA."""
    assert "OplsTypifier" in dir(molrs)
    typifier = molrs.OplsTypifier()
    assert typifier is not None


def test_typify_assigns_atom_types():
    """typify() returns a Frame whose atoms block carries assigned types."""
    typifier = molrs.OplsTypifier()
    frame = typifier.typify(_ethane())
    atoms = frame["atoms"]
    assert atoms.nrows == 8
    types = atoms["type"]
    # Every atom typed (no empty / null type label).
    assert all(str(t) != "" for t in types)


def test_typify_full_and_build():
    """typify_full() adds bonded blocks; build() yields finite energy."""
    typifier = molrs.OplsTypifier()
    mol = _ethane()
    full = typifier.typify_full(mol)
    assert full["bonds"].nrows > 0

    pots = typifier.build(mol)
    coords = molrs.extract_coords(typifier.typify(mol))
    energy, forces = pots.calc_energy_forces(coords)
    assert math.isfinite(energy)
    assert np.isfinite(np.asarray(forces)).all()


def test_from_xml_str_constructs():
    """from_xml_str builds a typifier from OPLS-AA XML text."""
    # The embedded canonical set is also reachable via the reader; round-trip a
    # minimal well-formed OPLS-AA forcefield document.
    xml = (
        "<ForceField><AtomTypes>"
        '<Type name="opls_135" class="CT" element="C" mass="12.011"/>'
        "</AtomTypes></ForceField>"
    )
    typifier = molrs.OplsTypifier.from_xml_str(xml)
    assert typifier is not None


def test_invalid_xml_raises_not_panics():
    """Malformed input raises a Python exception rather than aborting."""
    with pytest.raises((ValueError, RuntimeError)):
        molrs.OplsTypifier.from_xml_str("not valid xml <<<")
