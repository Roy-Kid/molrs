"""molrs — Rust-backed molecular simulation primitives.

Top-level re-exports cover core data structures, I/O, regions, conformer generation,
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
    # Public exceptions
    BlockDtypeError,
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
    read_pdb_trajectory,
    read_xyz,
    read_xyz_trajectory,
    XYZTrajReader,
    read_lammps,
    read_lammps_traj,
    LAMMPSTrajReader,
    read_dcd,
    DCDTrajReader,
    read_gro,
    read_chgcar_file,
    read_cube_file,
    write_cube_file,
    write_pdb,
    write_pdb_trajectory,
    write_xyz,
    write_gro,
    write_lammps,
    write_lammps_traj,
    write_dcd,
    parse_smiles,
    SmilesIR,
    # Regions
    Sphere,
    HollowSphere,
    Region,
    # Molecular graph hierarchy
    Graph,
    Atomistic,
    CoarseGrain,
    # Systems (module-level free functions over a graph)
    translate,
    rotate,
    perceive_aromaticity,
    add_hydrogens,
    find_rings,
    compute_gasteiger_charges,
    # Conformer generation
    Conformer,
    ConformerReport,
    ConformerStageReport,
    # Force field
    MMFFTypifier,
    ForceField,
    Potentials,
    OptReport,
    LBFGS,
    read_forcefield_xml,
    read_forcefield_xml_str,
    extract_coords,
    build_mmff_potentials,
    # Field-name convention submodule
    keys,
    # Signal processing (low-level FFT helpers)
    signal_acf_fft,
    signal_apply_window,
    signal_frequency_grid,
)

# Rich Python Frame/Block layer (pandas-style API; CSV engine in Rust on the
# core Block). These subclass the bare PyO3 cores and SHADOW the top-level
# ``molrs.Block`` / ``molrs.Frame`` as the canonical types — every public API
# (io readers, etc.) yields these. The shadow is safe now that molpy re-exports
# them instead of subclassing the bare core (chain spec 04). Internal modules
# that need the raw cores import them from ``.molrs`` directly.
from . import frame  # noqa: F401
from .frame import Block, Frame

from . import io  # molpy-compatible I/O facade (read_lammps_data, …)
from . import compute  # analysis subpackage — molrs.compute.{density,order,…}
from . import signal
from . import validate
from . import dielectric
from . import transport

__all__ = [
    "io",
    "compute",
    "signal",
    "validate",
    "dielectric",
    "transport",
    "BlockDtypeError",
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
    "read_pdb_trajectory",
    "read_xyz",
    "read_xyz_trajectory",
    "XYZTrajReader",
    "read_lammps",
    "read_lammps_traj",
    "LAMMPSTrajReader",
    "read_dcd",
    "DCDTrajReader",
    "read_gro",
    "read_chgcar_file",
    "read_cube_file",
    "write_cube_file",
    "write_pdb",
    "write_pdb_trajectory",
    "write_xyz",
    "write_gro",
    "write_lammps",
    "write_lammps_traj",
    "write_dcd",
    "parse_smiles",
    "SmilesIR",
    "Sphere",
    "HollowSphere",
    "Region",
    "Graph",
    "Atomistic",
    "CoarseGrain",
    "translate",
    "rotate",
    "perceive_aromaticity",
    "add_hydrogens",
    "find_rings",
    "compute_gasteiger_charges",
    "keys",
    "Conformer",
    "ConformerReport",
    "ConformerStageReport",
    "MMFFTypifier",
    "ForceField",
    "Potentials",
    "OptReport",
    "LBFGS",
    "build_mmff_potentials",
    "read_forcefield_xml",
    "read_forcefield_xml_str",
    "extract_coords",
    "signal_acf_fft",
    "signal_apply_window",
    "signal_frequency_grid",
]
