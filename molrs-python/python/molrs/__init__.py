"""molrs — Rust-backed molecular simulation primitives.

Top-level re-exports cover core data structures, I/O, regions, embedding,
force fields, and the SMILES front-end. Analysis classes live under the
:mod:`molrs.compute` package, namespaced by domain:

* :mod:`molrs.compute.density` — RDF, GaussianDensity, LocalDensity
* :mod:`molrs.compute.order` — Steinhardt, Nematic, Hexatic, SolidLiquid
* :mod:`molrs.compute.environment` — BondOrder
* :mod:`molrs.compute.pmft` — PMFTXY
* :mod:`molrs.compute.diffraction` — StaticStructureFactorDebye
* :mod:`molrs.compute.cluster` — Cluster, ClusterProperties, gyration / inertia / COM
* :mod:`molrs.compute.msd` — MSD
* :mod:`molrs.compute.ml` — PCA, K-means, descriptor rows
"""

from .molrs import (
    # SimBox + neighbors
    Box,
    LinkedCell,
    NeighborQuery,
    NeighborList,
    # Block + Frame
    Block,
    Frame,
    Trajectory,
    MolRec,
    Observables,
    ScalarObservable,
    VectorObservable,
    # I/O
    read_pdb,
    read_xyz,
    read_xyz_trajectory,
    read_lammps,
    read_lammps_traj,
    LAMMPSTrajReader,
    read_gro,
    read_chgcar_file,
    read_cube_file,
    write_cube_file,
    write_pdb,
    write_xyz,
    write_gro,
    write_lammps,
    write_lammps_traj,
    parse_smiles,
    SmilesIR,
    # Regions
    Sphere,
    HollowSphere,
    Region,
    # Molecular graph
    Atomistic,
    # Embed
    EmbedOptions,
    EmbedReport,
    EmbedResult,
    StageReport,
    generate_3d,
    # Force field
    MMFFTypifier,
    ForceField,
    Potentials,
    read_forcefield_xml,
    extract_coords,
)

from . import io  # molpy-compatible I/O facade (read_lammps_data, …)
from . import compute  # analysis subpackage — molrs.compute.{density,order,…}
from . import signal
from . import validate
from . import dielectric

__all__ = [
    "io",
    "compute",
    "signal",
    "validate",
    "dielectric",
    "Box",
    "LinkedCell",
    "NeighborQuery",
    "NeighborList",
    "Block",
    "Frame",
    "Trajectory",
    "MolRec",
    "Observables",
    "ScalarObservable",
    "VectorObservable",
    "read_pdb",
    "read_xyz",
    "read_xyz_trajectory",
    "read_lammps",
    "read_lammps_traj",
    "LAMMPSTrajReader",
    "read_gro",
    "read_chgcar_file",
    "read_cube_file",
    "write_cube_file",
    "write_pdb",
    "write_xyz",
    "write_gro",
    "write_lammps",
    "write_lammps_traj",
    "parse_smiles",
    "SmilesIR",
    "Sphere",
    "HollowSphere",
    "Region",
    "Atomistic",
    "EmbedOptions",
    "EmbedReport",
    "EmbedResult",
    "StageReport",
    "generate_3d",
    "MMFFTypifier",
    "ForceField",
    "Potentials",
    "read_forcefield_xml",
    "extract_coords",
]
