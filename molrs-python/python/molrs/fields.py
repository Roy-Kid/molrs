"""Canonical field definitions and I/O boundary formatters.

Translates format-native column names (``resid``, ``q``, ``symbol``) into
project-wide canonical names (``res_id``, ``charge``, ``element``) at the
I/O boundary.  Each format defines its own :class:`FieldFormatter` subclass
that the :mod:`molrs.io` facade applies automatically.

Architecture::

    FieldSpec                        — canonical field definition (key, dtype)
        ↓
    FieldFormatter                   — per-format mapping registry
        ↓                              canonicalize() / localize() on Block / Frame
    GroFieldFormatter(FieldFormatter) — GRO-specific mappings
    PdbFieldFormatter(FieldFormatter) — PDB-specific mappings
    ...

Usage (internal to molrs.io)::

    from molrs.fields import GroFieldFormatter

    fmt = GroFieldFormatter()
    frame = read_gro_native("system.gro")   # format-native names
    fmt.canonicalize_frame(frame)           # → canonical names
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# ===================================================================
#                       FieldSpec
# ===================================================================


@dataclass(frozen=True)
class FieldSpec:
    """Specification of a canonical field in the internal data model.

    Attributes:
        key: Canonical field name used internally.
        dtype: NumPy dtype for this field's array.
        doc: Human-readable description with units.
    """

    key: str
    dtype: np.dtype
    doc: str


# ===================================================================
#               Canonical Atom Fields
# ===================================================================

ATOM_ID = FieldSpec("id", np.dtype(np.int64), "atom ID (1-indexed)")
ATOM_TYPE = FieldSpec("type", np.dtype("U64"), "force field type label")
CHARGE = FieldSpec("charge", np.dtype(np.float64), "partial charge (e)")
MASS = FieldSpec("mass", np.dtype(np.float64), "atomic mass (amu)")
MOL_ID = FieldSpec("mol_id", np.dtype(np.int64), "molecule ID (1-indexed)")
ELEMENT = FieldSpec("element", np.dtype("U4"), "element symbol")
ATOM_NAME = FieldSpec("name", np.dtype("U8"), "atom name")
POS_X = FieldSpec("x", np.dtype(np.float64), "x coordinate (Angstrom)")
POS_Y = FieldSpec("y", np.dtype(np.float64), "y coordinate (Angstrom)")
POS_Z = FieldSpec("z", np.dtype(np.float64), "z coordinate (Angstrom)")
VEL_X = FieldSpec("vx", np.dtype(np.float64), "x velocity (Angstrom/fs)")
VEL_Y = FieldSpec("vy", np.dtype(np.float64), "y velocity (Angstrom/fs)")
VEL_Z = FieldSpec("vz", np.dtype(np.float64), "z velocity (Angstrom/fs)")
RES_ID = FieldSpec("res_id", np.dtype(np.int64), "residue ID")
RES_NAME = FieldSpec("res_name", np.dtype("U8"), "residue name")

# ===================================================================
#               Canonical Topology Fields
# ===================================================================

BOND_TYPE = FieldSpec("type", np.dtype("U64"), "bond type label")
BOND_ATOMI = FieldSpec("atomi", np.dtype(np.int64), "first atom index (0-indexed)")
BOND_ATOMJ = FieldSpec("atomj", np.dtype(np.int64), "second atom index (0-indexed)")

ANGLE_TYPE = FieldSpec("type", np.dtype("U64"), "angle type label")
ANGLE_ATOMI = FieldSpec("atomi", np.dtype(np.int64), "first atom index (0-indexed)")
ANGLE_ATOMJ = FieldSpec("atomj", np.dtype(np.int64), "vertex atom index (0-indexed)")
ANGLE_ATOMK = FieldSpec("atomk", np.dtype(np.int64), "third atom index (0-indexed)")

DIHEDRAL_TYPE = FieldSpec("type", np.dtype("U64"), "dihedral type label")
DIHEDRAL_ATOMI = FieldSpec("atomi", np.dtype(np.int64), "first atom index (0-indexed)")
DIHEDRAL_ATOMJ = FieldSpec("atomj", np.dtype(np.int64), "second atom index (0-indexed)")
DIHEDRAL_ATOMK = FieldSpec("atomk", np.dtype(np.int64), "third atom index (0-indexed)")
DIHEDRAL_ATOML = FieldSpec("atoml", np.dtype(np.int64), "fourth atom index (0-indexed)")


# ===================================================================
#                       FieldFormatter
# ===================================================================


class FieldFormatter:
    """Translates between format-specific and canonical field names.

    Subclasses define ``_field_formatters`` as a class-level registry mapping
    format-native column names to :class:`FieldSpec` objects.  Registrations
    are isolated per subclass via ``__init_subclass__``.

    Example::

        class LammpsFieldFormatter(FieldFormatter):
            _field_formatters = {
                "q":   CHARGE,   # LAMMPS "q" → canonical "charge"
                "mol": MOL_ID,   # LAMMPS "mol" → canonical "mol_id"
            }
    """

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._field_formatters = dict(cls._field_formatters)

    @classmethod
    def register_field(cls, format_key: str, spec: FieldSpec) -> None:
        """Register a field mapping at runtime."""
        cls._field_formatters[format_key] = spec

    # ── Block-level translation ──────────────────────────────────

    def canonicalize(self, block: "Block") -> "Block":
        """Reader exit: rename format-specific keys to canonical (in-place)."""
        for fmt_key, spec in self._field_formatters.items():
            if fmt_key in block and spec.key not in block:
                arr = block.view(fmt_key).copy()
                block.remove(fmt_key)
                block.insert(spec.key, arr)
        return block

    def localize(self, block: "Block") -> "Block":
        """Writer entry: rename canonical keys to format-specific (in-place)."""
        for fmt_key, spec in self._field_formatters.items():
            if spec.key in block and fmt_key not in block:
                arr = block.view(spec.key).copy()
                block.remove(spec.key)
                block.insert(fmt_key, arr)
        return block

    # ── Frame-level convenience ──────────────────────────────────

    def canonicalize_frame(self, frame: "Frame") -> "Frame":
        """Canonicalize all blocks in a Frame (in-place)."""
        for key in list(frame.keys()):
            self.canonicalize(frame[key])
        return frame

    def localize_frame(self, frame: "Frame") -> "Frame":
        """Localize all blocks in a Frame (in-place)."""
        for key in list(frame.keys()):
            self.localize(frame[key])
        return frame


# ===================================================================
#               Per-format Formatters
# ===================================================================


class GroFieldFormatter(FieldFormatter):
    """GRO format ↔ canonical field name translation.

    Maps the column names produced by molrs's native GRO reader to the
    project canonical names.
    """

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "resid": RES_ID,
        "resname": RES_NAME,
        "atom_name": ATOM_NAME,
        "atom_id": ATOM_ID,
    }


class PdbFieldFormatter(FieldFormatter):
    """PDB format ↔ canonical field name translation."""

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "resid": RES_ID,
        "resname": RES_NAME,
        "symbol": ELEMENT,
    }


class LammpsFieldFormatter(FieldFormatter):
    """LAMMPS data format ↔ canonical field name translation."""

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "q": CHARGE,
        "mol": MOL_ID,
    }


class XyzFieldFormatter(FieldFormatter):
    """XYZ format ↔ canonical field name translation."""

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "symbol": ELEMENT,
    }
