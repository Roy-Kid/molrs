use super::buffer::DeviceBuffer;
use super::device::CUDAStream;

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

/// Atom-index topology passed to GPU kernels at call time.
pub enum Topology {
    /// Bond / pair interactions: two atom indices per interaction.
    Pair {
        atom_i: DeviceBuffer<i32>,
        atom_j: DeviceBuffer<i32>,
        n_items: i32,
        n_atoms: i32,
    },
    /// Angle interactions: three atom indices per interaction.
    Triplet {
        atom_i: DeviceBuffer<i32>,
        atom_j: DeviceBuffer<i32>,
        atom_k: DeviceBuffer<i32>,
        n_items: i32,
        n_atoms: i32,
    },
}

// ---------------------------------------------------------------------------
// GPUPotential trait
// ---------------------------------------------------------------------------

/// GPU potential trait — parameterised by physics, topology passed at call time.
pub trait GPUPotential {
    /// Compute total potential energy (returns f64 on host).
    fn energy(&self, positions: &DeviceBuffer<f32>, topo: &Topology, stream: &CUDAStream) -> f64;

    /// Accumulate gradient into `grad` using atomic adds on device.
    fn gradient(
        &self,
        positions: &DeviceBuffer<f32>,
        grad: &mut DeviceBuffer<f32>,
        topo: &Topology,
        stream: &CUDAStream,
    );
}

// ---------------------------------------------------------------------------
// GPUPotentials collection
// ---------------------------------------------------------------------------

/// Collection of GPU potentials — mirrors CPU `Potentials`.
pub struct GPUPotentials {
    inner: Vec<(Box<dyn GPUPotential>, Topology)>,
}

impl GPUPotentials {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn push(&mut self, pot: Box<dyn GPUPotential>, topo: Topology) {
        self.inner.push((pot, topo));
    }

    pub fn energy(&self, positions: &DeviceBuffer<f32>, stream: &CUDAStream) -> f64 {
        self.inner
            .iter()
            .map(|(p, t)| p.energy(positions, t, stream))
            .sum()
    }

    pub fn gradient(
        &self,
        positions: &DeviceBuffer<f32>,
        grad: &mut DeviceBuffer<f32>,
        stream: &CUDAStream,
    ) {
        grad.zero();
        for (p, t) in &self.inner {
            p.gradient(positions, grad, t, stream);
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for GPUPotentials {
    fn default() -> Self {
        Self::new()
    }
}
