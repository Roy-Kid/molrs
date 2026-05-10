# Python Reference

This page is rendered from the installed `molrs` Python module by
`mkdocstrings-python`. The type stub in `molrs-python/python/molrs/molrs.pyi`
is the committed companion artifact that keeps signatures visible to static
tools and the docs build.

## Core Model

::: molrs.Box

::: molrs.Block

::: molrs.Grid

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

## 3D Embedding

::: molrs.EmbedOptions

::: molrs.StageReport

::: molrs.EmbedReport

::: molrs.EmbedResult

::: molrs.generate_3d

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

::: molrs.GridObservable

## Analysis

::: molrs.RDF

::: molrs.RDFResult

::: molrs.MSD

::: molrs.MSDResult

::: molrs.MSDTimeSeries

::: molrs.Cluster

::: molrs.ClusterResult

::: molrs.ClusterCenters

::: molrs.ClusterCentersResult

::: molrs.CenterOfMass

::: molrs.CenterOfMassResult

::: molrs.GyrationTensor

::: molrs.InertiaTensor

::: molrs.RadiusOfGyration

::: molrs.DescriptorRow

::: molrs.Pca2

::: molrs.PcaResult

::: molrs.KMeans

::: molrs.KMeansResult
