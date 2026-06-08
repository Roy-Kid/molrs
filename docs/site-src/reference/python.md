# Python Reference

This page is rendered from the installed `molrs` Python module by
`mkdocstrings-python`. The type stub in `molrs-python/python/molrs/molrs.pyi`
is the committed companion artifact that keeps signatures visible to static
tools and the docs build.

## Core Model

::: molrs.Box

::: molrs.Block

::: molrs.Frame

## Topology and SMILES

::: molrs.Atomistic

::: molrs.SmilesIR

::: molrs.parse_smiles

## I/O

::: molrs.read_pdb

::: molrs.read_xyz

::: molrs.read_xyz_trajectory

::: molrs.read_lammps

::: molrs.read_lammps_traj

::: molrs.LAMMPSTrajReader

::: molrs.read_chgcar_file

::: molrs.read_cube_file

::: molrs.write_cube_file

::: molrs.write_pdb

::: molrs.write_xyz

::: molrs.write_lammps

::: molrs.write_lammps_traj

## Regions and Neighbor Search

::: molrs.Sphere

::: molrs.HollowSphere

::: molrs.Region

::: molrs.LinkedCell

::: molrs.NeighborQuery

::: molrs.NeighborList

## 3D Conformer Generation

::: molrs.Conformer

::: molrs.ConformerStageReport

::: molrs.ConformerReport

## Force Fields

::: molrs.ForceField

::: molrs.MMFFTypifier

::: molrs.Potentials

::: molrs.read_forcefield_xml

::: molrs.extract_coords

## MolRec

::: molrs.Trajectory

::: molrs.MolRec

::: molrs.Observables

::: molrs.ScalarObservable

::: molrs.VectorObservable

## Analysis

Analysis classes live under the `molrs.compute` subpackage, organized by
domain. The layout mirrors freud and the underlying Rust crate
(`molrs_compute::{density, order, environment, …}`).

### `molrs.compute.density`

::: molrs.compute.density.RDF

::: molrs.compute.density.RDFResult

::: molrs.compute.density.GaussianDensity

::: molrs.compute.density.LocalDensity

### `molrs.compute.order`

::: molrs.compute.order.Steinhardt

::: molrs.compute.order.Nematic

::: molrs.compute.order.Hexatic

::: molrs.compute.order.SolidLiquid

### `molrs.compute.environment`

::: molrs.compute.environment.BondOrder

### `molrs.compute.pmft`

::: molrs.compute.pmft.PMFTXY

### `molrs.compute.diffraction`

::: molrs.compute.diffraction.StaticStructureFactorDebye

### `molrs.compute.cluster`

::: molrs.compute.cluster.Cluster

::: molrs.compute.cluster.ClusterResult

::: molrs.compute.cluster.ClusterCenters

::: molrs.compute.cluster.ClusterCentersResult

::: molrs.compute.cluster.ClusterProperties

::: molrs.compute.cluster.CenterOfMass

::: molrs.compute.cluster.CenterOfMassResult

::: molrs.compute.cluster.GyrationTensor

::: molrs.compute.cluster.InertiaTensor

::: molrs.compute.cluster.RadiusOfGyration

### `molrs.compute.msd`

::: molrs.compute.msd.MSD

::: molrs.compute.msd.MSDResult

::: molrs.compute.msd.MSDTimeSeries

### `molrs.compute.ml`

::: molrs.compute.ml.DescriptorRow

::: molrs.compute.ml.Pca2

::: molrs.compute.ml.PcaResult

::: molrs.compute.ml.KMeans

::: molrs.compute.ml.KMeansResult

## Transport

Electrolyte transport kernels (ports of the *tame* recipes). See the
[Transport Kernels](../guides/transport.md) guide for signatures, units, and
worked examples.

### `molrs.transport`

::: molrs.transport.Onsager

::: molrs.transport.Jacf

::: molrs.transport.Persist
