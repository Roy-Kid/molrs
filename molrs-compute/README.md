# molcrafts-molrs-compute

Trajectory analysis for molrs: RDF, MSD, clustering, gyration / inertia tensors,
PCA, k-means — built around a single unified `Compute` trait.

## Core ideas

1. **`Compute`** — every analysis implements this trait. `&self` is an
   immutable parameter bag; two `compute` calls with identical `frames` +
   `args` always produce identical output. No hidden state, no interior
   mutability.

   ```rust
   pub trait Compute {
       type Args<'a>;
       type Output: ComputeResult + Clone + Send + Sync + 'static;
       fn compute<'a, FA: FrameAccess + 'a>(
           &self,
           frames: &[&'a FA],
           args: Self::Args<'a>,
       ) -> Result<Self::Output, ComputeError>;
   }
   ```

   A single frame is a length-1 slice. A trajectory is a longer slice. A
   dataset is still just a slice. Each impl decides how to interpret it.

2. **`ComputeResult`** — every output implements it; accumulating outputs
   override `finalize` to normalize. Callers should invoke `finalize` after
   `compute` to obtain the user-facing final form.

3. **`DescriptorRow`** — flatten any Compute output into `&[F]` rows so PCA
   and k-means can consume arbitrary upstream per-frame outputs as a matrix.

## Stateless `Compute` — orchestrate from the caller

Every analysis kernel in this crate is stateless. DAG orchestration
(topological order, diamond reuse, external input validation) belongs to the
caller. On the Python side, `molpy.compute.Workflow` does exactly that — it
composes `Compute` nodes via Python's stdlib `graphlib`, calls each Rust
kernel directly, and never touches any Rust-side Graph.

See the **[MolPy Workflow tutorial](https://molcrafts.github.io/molpy/tutorials/workflow/)**
for end-to-end examples (NeighborList → RDF, diamond reuse, cycle detection).

No `Accumulator`, no `Reducer`. No hidden state in any `Compute`. No free
functions pretending not to be Compute nodes.

## Usage

### Direct call — single analysis

```rust
use molrs::Frame;
use molrs::neighbors::{NeighborList, NeighborQuery};
use molrs_compute::RDF;

let rdf = RDF::new(40, 5.0, 0.0)?;
let result = rdf.compute(&[&frame], &vec![nlist])?;
println!("{:?}", result.rdf);
```

### Chaining — caller wires outputs to inputs

```rust
use molrs_compute::{Cluster, CenterOfMass, RadiusOfGyration};

let clusters = Cluster::new(1).compute(&frames, &nlists)?;
let com = CenterOfMass::new().compute(&frames, &clusters)?;
let rg = RadiusOfGyration::new().compute(&frames, (&clusters, &com))?;
```

## Available analyses

| Module | Args | Output |
|--------|------|--------|
| `rdf` | `&Vec<NeighborList>` | `RDFResult` |
| `msd` | `()` | `MSDTimeSeries` |
| `cluster` | `&Vec<NeighborList>` | `Vec<ClusterResult>` |
| `cluster_centers` | `&Vec<ClusterResult>` | `Vec<ClusterCentersResult>` |
| `center_of_mass` | `&Vec<ClusterResult>` | `Vec<COMResult>` |
| `radius_of_gyration` | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<RgResult>` |
| `inertia_tensor` | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<InertiaTensorResult>` |
| `gyration_tensor` | `(&Vec<ClusterResult>, &Vec<ClusterCentersResult>)` | `Vec<GyrationTensorResult>` |
| `pca` | `&Vec<T: DescriptorRow>` | `PcaResult` |
| `kmeans` | `&PcaResult` | `KMeansResult` |

## Testing

```
cargo test -p molcrafts-molrs-compute
cargo bench -p molcrafts-molrs-compute
```

## Benchmarks

Criterion benches are organized one file per compute kernel. Each kernel file
holds two sweep functions pulling their axes from `helpers.rs`:

- **`<kernel>/size_sweep/{N}`** — `N` ∈ `{100, 500, 2 000, 10 000, 50 000}` at
  constant liquid density (ρ = 0.03 Å⁻³), 4 frames per point. Exposes
  single-threaded per-particle scaling.
- **`<kernel>/frame_sweep/{n_frames}`** — `N = 5 000`, `n_frames` ∈
  `{1, 2, 4, 8, 16, 32, 64}`. Exposes rayon parallel scaling across frames.

Layout:

```
benches/
├── benchmarks.rs            # criterion_main!
├── helpers.rs               # shared sweep axes + fixture builders
├── rdf.rs                   # rdf/size_sweep, rdf/frame_sweep
├── cluster.rs               # cluster/…
├── cluster_centers.rs
├── center_of_mass.rs
├── gyration_tensor.rs
├── inertia_tensor.rs
├── radius_of_gyration.rs
└── msd.rs
```

Narrow to a single kernel or axis:

```
cargo bench -p molcrafts-molrs-compute -- rdf
cargo bench -p molcrafts-molrs-compute -- cluster/frame_sweep
cargo bench -p molcrafts-molrs-compute -- gyration_tensor/size_sweep/50000
```
