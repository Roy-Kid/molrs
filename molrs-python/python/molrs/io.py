"""Python-shaped I/O facade for molrs.

This module exposes the Rust readers/writers under names that mirror the
``molpy.io`` API so molrs can serve as a drop-in replacement in tests
and downstream code.

Every reader applies a :class:`FieldFormatter <molrs.fields.FieldFormatter>`
to translate format-native column names (``resid``, ``q``, ``symbol``) into
project-wide canonical names (``res_id``, ``charge``, ``element``).  Writers
apply the reverse translation before delegating to the native backend.

For access to raw format-native output, use the top-level ``molrs.read_*``
functions directly.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

from .fields import (
    GroFieldFormatter,
    LammpsFieldFormatter,
    PdbFieldFormatter,
    XyzFieldFormatter,
)
from .molrs import (
    read_gro as _read_gro,
    read_lammps as _read_lammps,
    read_pdb as _read_pdb,
    read_xyz as _read_xyz,
    write_gro as _write_gro,
    write_lammps as _write_lammps,
    write_pdb as _write_pdb,
    write_xyz as _write_xyz,
)

_gro_fmt = GroFieldFormatter()
_pdb_fmt = PdbFieldFormatter()
_lammps_fmt = LammpsFieldFormatter()
_xyz_fmt = XyzFieldFormatter()


def read_lammps_data(
    file: str | PathLike[str],
    atom_style: str | None = None,
    frame: Any = None,
) -> Any:
    """Read a LAMMPS data file. molpy-compatible signature.

    Args:
        file: Path to the LAMMPS data file.
        atom_style: Accepted for API parity with molpy; the Rust reader
            auto-detects the style from the file (column count and the
            optional ``Atoms # <style>`` comment).
        frame: Reserved for API parity. molrs always returns a new
            ``Frame``; passing an existing frame is not supported.

    Returns:
        A molrs ``Frame`` with canonical field names.
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_lammps_data does not accept an existing frame; "
            "it always returns a new Frame."
        )
    result = _read_lammps(str(file))
    _lammps_fmt.canonicalize_frame(result)
    return result


def read_pdb(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read a PDB file. molpy-compatible signature.

    Returns a molrs ``Frame`` with canonical field names
    (``element`` instead of ``symbol``, ``res_id`` instead of ``resid``).
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_pdb does not accept an existing frame."
        )
    result = _read_pdb(str(file))
    _pdb_fmt.canonicalize_frame(result)
    return result


def read_xyz(file: str | PathLike[str], frame: Any = None) -> Any:
    """Read an XYZ file. molpy-compatible signature.

    Returns a molrs ``Frame`` with canonical field names
    (``element`` instead of ``symbol``).
    """
    if frame is not None:
        raise NotImplementedError(
            "molrs.io.read_xyz does not accept an existing frame."
        )
    result = _read_xyz(str(file))
    _xyz_fmt.canonicalize_frame(result)
    return result


def read_gro(file: str | PathLike[str]) -> list[Any]:
    """Read all frames from a GROMACS GRO file.

    Args:
        file: Path to a ``.gro`` file (single- or multi-frame).

    Returns:
        List of molrs ``Frame`` objects with canonical field names
        (``res_id``, ``res_name``, ``name``, ``id``).

    Raises:
        OSError: If the file cannot be opened or parsed.
    """
    frames = _read_gro(str(file))
    for f in frames:
        _gro_fmt.canonicalize_frame(f)
    return frames


def write_lammps_data(
    file: str | PathLike[str],
    frame: Any,
    atom_style: str | None = None,
) -> None:
    """Write a LAMMPS data file. molpy-compatible signature.

    ``atom_style`` is accepted for API parity but the writer derives the
    style from the columns present in ``frame['atoms']``.

    The *frame* is localised (canonical → format-native column names)
    before writing.
    """
    _lammps_fmt.localize_frame(frame)
    _write_lammps(str(file), frame)


def write_pdb(file: str | PathLike[str], frame: Any) -> None:
    """Write a PDB file.  Localises *frame* in-place before writing."""
    _pdb_fmt.localize_frame(frame)
    _write_pdb(str(file), frame)


def write_xyz(file: str | PathLike[str], frame: Any) -> None:
    """Write an XYZ file.  Localises *frame* in-place before writing."""
    _xyz_fmt.localize_frame(frame)
    _write_xyz(str(file), frame)


def write_gro(file: str | PathLike[str], frame: Any) -> None:
    """Write a single Frame to a GROMACS GRO file.

    Localises *frame* in-place (``res_id`` → ``resid``,
    ``name`` → ``atom_name``, ``id`` → ``atom_id``) before writing.
    """
    _gro_fmt.localize_frame(frame)
    _write_gro(str(file), frame)


__all__ = [
    "read_lammps_data",
    "read_pdb",
    "read_xyz",
    "read_gro",
    "write_lammps_data",
    "write_pdb",
    "write_xyz",
    "write_gro",
]
