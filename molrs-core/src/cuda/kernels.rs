use crate::core::forcefield::Params;
use crate::core::frame::Frame;

use super::bindings;
use super::buffer::DeviceBuffer;
use super::device::{CUDADevice, CUDAStream};
use super::potential::{GPUPotential, Topology};

// ---------------------------------------------------------------------------
// BondHarmonicGPU
// ---------------------------------------------------------------------------

pub struct BondHarmonicGPU {
    k: DeviceBuffer<f32>,
    r0: DeviceBuffer<f32>,
}

impl BondHarmonicGPU {
    pub fn new(k: DeviceBuffer<f32>, r0: DeviceBuffer<f32>) -> Self {
        Self { k, r0 }
    }
}

impl GPUPotential for BondHarmonicGPU {
    fn energy(&self, positions: &DeviceBuffer<f32>, topo: &Topology, stream: &CUDAStream) -> f64 {
        let Topology::Pair {
            atom_i,
            atom_j,
            n_items,
            n_atoms,
        } = topo
        else {
            panic!("BondHarmonicGPU requires Pair topology");
        };
        unsafe {
            bindings::molrs_bond_harmonic_energy(
                positions.as_ptr(),
                atom_i.as_ptr(),
                atom_j.as_ptr(),
                self.k.as_ptr(),
                self.r0.as_ptr(),
                *n_items,
                *n_atoms,
                stream.as_ptr(),
            )
        }
    }

    fn gradient(
        &self,
        positions: &DeviceBuffer<f32>,
        grad: &mut DeviceBuffer<f32>,
        topo: &Topology,
        stream: &CUDAStream,
    ) {
        let Topology::Pair {
            atom_i,
            atom_j,
            n_items,
            n_atoms,
        } = topo
        else {
            panic!("BondHarmonicGPU requires Pair topology");
        };
        unsafe {
            bindings::molrs_bond_harmonic_gradient(
                positions.as_ptr(),
                grad.as_mut_ptr(),
                atom_i.as_ptr(),
                atom_j.as_ptr(),
                self.k.as_ptr(),
                self.r0.as_ptr(),
                *n_items,
                *n_atoms,
                stream.as_ptr(),
            );
        }
    }
}

/// Constructor for BondHarmonicGPU — mirrors `bond_harmonic_ctor` on CPU.
pub fn bond_harmonic_gpu_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
    _device: &CUDADevice,
) -> Result<(Box<dyn GPUPotential>, Topology), String> {
    let type_map: std::collections::HashMap<&str, &Params> = type_params.iter().copied().collect();

    let mut h_atom_i = Vec::new();
    let mut h_atom_j = Vec::new();
    let mut h_k = Vec::new();
    let mut h_r0 = Vec::new();

    if let Some(block) = frame.get("bonds") {
        if let (Some(i_col), Some(j_col), Some(type_col)) = (
            block.get_u32("i"),
            block.get_u32("j"),
            block.get_string("type"),
        ) {
            for idx in 0..i_col.len() {
                let label = &type_col[idx];
                let params = type_map
                    .get(label.as_str())
                    .ok_or_else(|| format!("bond type '{}' not found", label))?;
                let k_val = params
                    .get("k")
                    .ok_or_else(|| format!("missing 'k' for bond type '{}'", label))?;
                let r0_val = params
                    .get("r0")
                    .ok_or_else(|| format!("missing 'r0' for bond type '{}'", label))?;

                h_atom_i.push(i_col[idx] as i32);
                h_atom_j.push(j_col[idx] as i32);
                h_k.push(*k_val as f32);
                h_r0.push(*r0_val as f32);
            }
        }
    }

    let n = h_atom_i.len();
    let max_idx = h_atom_i
        .iter()
        .chain(h_atom_j.iter())
        .copied()
        .max()
        .unwrap_or(0);

    let topo = Topology::Pair {
        atom_i: DeviceBuffer::from_host(&h_atom_i),
        atom_j: DeviceBuffer::from_host(&h_atom_j),
        n_items: n as i32,
        n_atoms: max_idx + 1,
    };

    let pot = Box::new(BondHarmonicGPU::new(
        DeviceBuffer::from_host(&h_k),
        DeviceBuffer::from_host(&h_r0),
    ));

    Ok((pot, topo))
}

// ---------------------------------------------------------------------------
// AngleHarmonicGPU
// ---------------------------------------------------------------------------

pub struct AngleHarmonicGPU {
    k_spring: DeviceBuffer<f32>,
    theta0: DeviceBuffer<f32>,
}

impl AngleHarmonicGPU {
    pub fn new(k_spring: DeviceBuffer<f32>, theta0: DeviceBuffer<f32>) -> Self {
        Self { k_spring, theta0 }
    }
}

impl GPUPotential for AngleHarmonicGPU {
    fn energy(&self, positions: &DeviceBuffer<f32>, topo: &Topology, stream: &CUDAStream) -> f64 {
        let Topology::Triplet {
            atom_i,
            atom_j,
            atom_k,
            n_items,
            n_atoms,
        } = topo
        else {
            panic!("AngleHarmonicGPU requires Triplet topology");
        };
        unsafe {
            bindings::molrs_angle_harmonic_energy(
                positions.as_ptr(),
                atom_i.as_ptr(),
                atom_j.as_ptr(),
                atom_k.as_ptr(),
                self.k_spring.as_ptr(),
                self.theta0.as_ptr(),
                *n_items,
                *n_atoms,
                stream.as_ptr(),
            )
        }
    }

    fn gradient(
        &self,
        positions: &DeviceBuffer<f32>,
        grad: &mut DeviceBuffer<f32>,
        topo: &Topology,
        stream: &CUDAStream,
    ) {
        let Topology::Triplet {
            atom_i,
            atom_j,
            atom_k,
            n_items,
            n_atoms,
        } = topo
        else {
            panic!("AngleHarmonicGPU requires Triplet topology");
        };
        unsafe {
            bindings::molrs_angle_harmonic_gradient(
                positions.as_ptr(),
                grad.as_mut_ptr(),
                atom_i.as_ptr(),
                atom_j.as_ptr(),
                atom_k.as_ptr(),
                self.k_spring.as_ptr(),
                self.theta0.as_ptr(),
                *n_items,
                *n_atoms,
                stream.as_ptr(),
            );
        }
    }
}

/// Constructor for AngleHarmonicGPU — mirrors `angle_harmonic_ctor` on CPU.
pub fn angle_harmonic_gpu_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
    _device: &CUDADevice,
) -> Result<(Box<dyn GPUPotential>, Topology), String> {
    let type_map: std::collections::HashMap<&str, &Params> = type_params.iter().copied().collect();

    let mut h_atom_i = Vec::new();
    let mut h_atom_j = Vec::new();
    let mut h_atom_k = Vec::new();
    let mut h_k = Vec::new();
    let mut h_theta0 = Vec::new();

    if let Some(block) = frame.get("angles") {
        if let (Some(i_col), Some(j_col), Some(k_col), Some(type_col)) = (
            block.get_u32("i"),
            block.get_u32("j"),
            block.get_u32("k"),
            block.get_string("type"),
        ) {
            for idx in 0..i_col.len() {
                let label = &type_col[idx];
                let params = type_map
                    .get(label.as_str())
                    .ok_or_else(|| format!("angle type '{}' not found", label))?;
                let k_val = params
                    .get("k")
                    .ok_or_else(|| format!("missing 'k' for angle type '{}'", label))?;
                let theta0_val = params
                    .get("theta0")
                    .ok_or_else(|| format!("missing 'theta0' for angle type '{}'", label))?;

                h_atom_i.push(i_col[idx] as i32);
                h_atom_j.push(j_col[idx] as i32);
                h_atom_k.push(k_col[idx] as i32);
                h_k.push(*k_val as f32);
                h_theta0.push(*theta0_val as f32);
            }
        }
    }

    let n = h_atom_i.len();
    let max_idx = h_atom_i
        .iter()
        .chain(h_atom_j.iter())
        .chain(h_atom_k.iter())
        .copied()
        .max()
        .unwrap_or(0);

    let topo = Topology::Triplet {
        atom_i: DeviceBuffer::from_host(&h_atom_i),
        atom_j: DeviceBuffer::from_host(&h_atom_j),
        atom_k: DeviceBuffer::from_host(&h_atom_k),
        n_items: n as i32,
        n_atoms: max_idx + 1,
    };

    let pot = Box::new(AngleHarmonicGPU::new(
        DeviceBuffer::from_host(&h_k),
        DeviceBuffer::from_host(&h_theta0),
    ));

    Ok((pot, topo))
}

// ---------------------------------------------------------------------------
// PairLJ126GPU
// ---------------------------------------------------------------------------

pub struct PairLJ126GPU {
    epsilon: DeviceBuffer<f32>,
    sigma: DeviceBuffer<f32>,
}

impl PairLJ126GPU {
    pub fn new(epsilon: DeviceBuffer<f32>, sigma: DeviceBuffer<f32>) -> Self {
        Self { epsilon, sigma }
    }
}

impl GPUPotential for PairLJ126GPU {
    fn energy(&self, positions: &DeviceBuffer<f32>, topo: &Topology, stream: &CUDAStream) -> f64 {
        let Topology::Pair {
            atom_i,
            atom_j,
            n_items,
            n_atoms,
        } = topo
        else {
            panic!("PairLJ126GPU requires Pair topology");
        };
        unsafe {
            bindings::molrs_pair_lj126_energy(
                positions.as_ptr(),
                atom_i.as_ptr(),
                atom_j.as_ptr(),
                self.epsilon.as_ptr(),
                self.sigma.as_ptr(),
                *n_items,
                *n_atoms,
                stream.as_ptr(),
            )
        }
    }

    fn gradient(
        &self,
        positions: &DeviceBuffer<f32>,
        grad: &mut DeviceBuffer<f32>,
        topo: &Topology,
        stream: &CUDAStream,
    ) {
        let Topology::Pair {
            atom_i,
            atom_j,
            n_items,
            n_atoms,
        } = topo
        else {
            panic!("PairLJ126GPU requires Pair topology");
        };
        unsafe {
            bindings::molrs_pair_lj126_gradient(
                positions.as_ptr(),
                grad.as_mut_ptr(),
                atom_i.as_ptr(),
                atom_j.as_ptr(),
                self.epsilon.as_ptr(),
                self.sigma.as_ptr(),
                *n_items,
                *n_atoms,
                stream.as_ptr(),
            );
        }
    }
}

/// Constructor for PairLJ126GPU — mirrors `pair_lj126_ctor` on CPU.
pub fn pair_lj126_gpu_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
    _device: &CUDADevice,
) -> Result<(Box<dyn GPUPotential>, Topology), String> {
    let type_map: std::collections::HashMap<&str, &Params> = type_params.iter().copied().collect();

    let mut h_atom_i = Vec::new();
    let mut h_atom_j = Vec::new();
    let mut h_epsilon = Vec::new();
    let mut h_sigma = Vec::new();

    if let Some(block) = frame.get("pairs") {
        if let (Some(i_col), Some(j_col), Some(type_col)) = (
            block.get_u32("i"),
            block.get_u32("j"),
            block.get_string("type"),
        ) {
            for idx in 0..i_col.len() {
                let label = &type_col[idx];
                let params = type_map
                    .get(label.as_str())
                    .ok_or_else(|| format!("pair type '{}' not found", label))?;
                let eps = params
                    .get("epsilon")
                    .ok_or_else(|| format!("missing 'epsilon' for pair type '{}'", label))?;
                let sig = params
                    .get("sigma")
                    .ok_or_else(|| format!("missing 'sigma' for pair type '{}'", label))?;

                h_atom_i.push(i_col[idx] as i32);
                h_atom_j.push(j_col[idx] as i32);
                h_epsilon.push(*eps as f32);
                h_sigma.push(*sig as f32);
            }
        }
    }

    let n = h_atom_i.len();
    let max_idx = h_atom_i
        .iter()
        .chain(h_atom_j.iter())
        .copied()
        .max()
        .unwrap_or(0);

    let topo = Topology::Pair {
        atom_i: DeviceBuffer::from_host(&h_atom_i),
        atom_j: DeviceBuffer::from_host(&h_atom_j),
        n_items: n as i32,
        n_atoms: max_idx + 1,
    };

    let pot = Box::new(PairLJ126GPU::new(
        DeviceBuffer::from_host(&h_epsilon),
        DeviceBuffer::from_host(&h_sigma),
    ));

    Ok((pot, topo))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::potential::Potential;

    const TOL: f64 = 1e-3;

    fn init_device() -> (CUDADevice, CUDAStream) {
        let device = CUDADevice::new(0).expect("CUDA device 0");
        let stream = CUDAStream::default_stream();
        (device, stream)
    }

    // --- BondHarmonicGPU tests ---

    #[test]
    fn test_bond_harmonic_gpu_energy() {
        let (_dev, stream) = init_device();
        let coords = vec![0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];

        let gpu = BondHarmonicGPU::new(
            DeviceBuffer::from_host(&[300.0f32]),
            DeviceBuffer::from_host(&[1.5f32]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            n_items: 1,
            n_atoms: 2,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let gpu_e = gpu.energy(&d_pos, &topo, &stream);

        let cpu = crate::core::potential_kernels::BondHarmonic::new(
            vec![0],
            vec![1],
            vec![300.0],
            vec![1.5],
        );
        let cpu_e = cpu.energy(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);

        assert!((gpu_e - cpu_e).abs() < TOL, "GPU={gpu_e}, CPU={cpu_e}");
    }

    #[test]
    fn test_bond_harmonic_gpu_gradient() {
        let (_dev, stream) = init_device();
        let coords = vec![0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];

        let gpu = BondHarmonicGPU::new(
            DeviceBuffer::from_host(&[300.0f32]),
            DeviceBuffer::from_host(&[1.5f32]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            n_items: 1,
            n_atoms: 2,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let mut d_grad = DeviceBuffer::<f32>::alloc(6);
        d_grad.zero();
        gpu.gradient(&d_pos, &mut d_grad, &topo, &stream);
        let gpu_g = d_grad.to_host();

        let cpu = crate::core::potential_kernels::BondHarmonic::new(
            vec![0],
            vec![1],
            vec![300.0],
            vec![1.5],
        );
        let mut cpu_g = vec![0.0f64; 6];
        cpu.gradient(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0], &mut cpu_g);

        for i in 0..6 {
            assert!(
                (gpu_g[i] as f64 - cpu_g[i]).abs() < TOL,
                "grad[{i}]: GPU={}, CPU={}",
                gpu_g[i],
                cpu_g[i],
            );
        }
    }

    #[test]
    fn test_bond_harmonic_gpu_multiple_bonds() {
        let (_dev, stream) = init_device();
        let coords = vec![0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0, 3.5, 0.0, 0.0];

        let gpu = BondHarmonicGPU::new(
            DeviceBuffer::from_host(&[300.0f32, 200.0]),
            DeviceBuffer::from_host(&[1.5f32, 1.0]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32, 1]),
            atom_j: DeviceBuffer::from_host(&[1i32, 2]),
            n_items: 2,
            n_atoms: 3,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let gpu_e = gpu.energy(&d_pos, &topo, &stream);

        let cpu = crate::core::potential_kernels::BondHarmonic::new(
            vec![0, 1],
            vec![1, 2],
            vec![300.0, 200.0],
            vec![1.5, 1.0],
        );
        let cpu_e = cpu.energy(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.5, 0.0, 0.0]);

        assert!((gpu_e - cpu_e).abs() < TOL, "GPU={gpu_e}, CPU={cpu_e}");
    }

    // --- AngleHarmonicGPU tests ---

    #[test]
    fn test_angle_harmonic_gpu_energy_at_equilibrium() {
        let (_dev, stream) = init_device();
        let theta0 = std::f64::consts::FRAC_PI_2 as f32;
        let coords = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let gpu = AngleHarmonicGPU::new(
            DeviceBuffer::from_host(&[50.0f32]),
            DeviceBuffer::from_host(&[theta0]),
        );
        let topo = Topology::Triplet {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            atom_k: DeviceBuffer::from_host(&[2i32]),
            n_items: 1,
            n_atoms: 3,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let gpu_e = gpu.energy(&d_pos, &topo, &stream);

        assert!(gpu_e.abs() < TOL, "expected ~0, got {gpu_e}");
    }

    #[test]
    fn test_angle_harmonic_gpu_energy() {
        let (_dev, stream) = init_device();
        let theta0 = std::f64::consts::FRAC_PI_2;
        let coords = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];

        let gpu = AngleHarmonicGPU::new(
            DeviceBuffer::from_host(&[50.0f32]),
            DeviceBuffer::from_host(&[theta0 as f32]),
        );
        let topo = Topology::Triplet {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            atom_k: DeviceBuffer::from_host(&[2i32]),
            n_items: 1,
            n_atoms: 3,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let gpu_e = gpu.energy(&d_pos, &topo, &stream);

        let cpu = crate::core::potential_kernels::AngleHarmonic::new(
            vec![0],
            vec![1],
            vec![2],
            vec![50.0],
            vec![theta0],
        );
        let cpu_e = cpu.energy(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);

        assert!((gpu_e - cpu_e).abs() < TOL, "GPU={gpu_e}, CPU={cpu_e}");
    }

    #[test]
    fn test_angle_harmonic_gpu_gradient() {
        let (_dev, stream) = init_device();
        let theta0 = 100.0_f64.to_radians();
        let coords = vec![1.0f32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.1];

        let gpu = AngleHarmonicGPU::new(
            DeviceBuffer::from_host(&[60.0f32]),
            DeviceBuffer::from_host(&[theta0 as f32]),
        );
        let topo = Topology::Triplet {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            atom_k: DeviceBuffer::from_host(&[2i32]),
            n_items: 1,
            n_atoms: 3,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let mut d_grad = DeviceBuffer::<f32>::alloc(9);
        d_grad.zero();
        gpu.gradient(&d_pos, &mut d_grad, &topo, &stream);
        let gpu_g = d_grad.to_host();

        let cpu = crate::core::potential_kernels::AngleHarmonic::new(
            vec![0],
            vec![1],
            vec![2],
            vec![60.0],
            vec![theta0],
        );
        let mut cpu_g = vec![0.0f64; 9];
        cpu.gradient(&[1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.1], &mut cpu_g);

        for i in 0..9 {
            assert!(
                (gpu_g[i] as f64 - cpu_g[i]).abs() < TOL,
                "grad[{i}]: GPU={}, CPU={}",
                gpu_g[i],
                cpu_g[i],
            );
        }
    }

    #[test]
    fn test_angle_harmonic_gpu_translational_invariance() {
        let (_dev, stream) = init_device();
        let theta0 = 100.0_f64.to_radians();
        let coords = vec![1.0f32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.1];

        let gpu = AngleHarmonicGPU::new(
            DeviceBuffer::from_host(&[60.0f32]),
            DeviceBuffer::from_host(&[theta0 as f32]),
        );
        let topo = Topology::Triplet {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            atom_k: DeviceBuffer::from_host(&[2i32]),
            n_items: 1,
            n_atoms: 3,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let mut d_grad = DeviceBuffer::<f32>::alloc(9);
        d_grad.zero();
        gpu.gradient(&d_pos, &mut d_grad, &topo, &stream);
        let g = d_grad.to_host();

        for dim in 0..3 {
            let sum = g[dim] + g[3 + dim] + g[6 + dim];
            assert!((sum as f64).abs() < 1e-4, "dim={dim}: sum={sum}");
        }
    }

    // --- PairLJ126GPU tests ---

    #[test]
    fn test_pair_lj126_gpu_energy() {
        let (_dev, stream) = init_device();
        let r_min = 2.0_f64.powf(1.0 / 6.0) as f32;
        let coords = vec![0.0f32, 0.0, 0.0, r_min, 0.0, 0.0];

        let gpu = PairLJ126GPU::new(
            DeviceBuffer::from_host(&[0.5f32]),
            DeviceBuffer::from_host(&[1.0f32]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            n_items: 1,
            n_atoms: 2,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let gpu_e = gpu.energy(&d_pos, &topo, &stream);

        let cpu =
            crate::core::potential_kernels::PairLJ126::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let cpu_e = cpu.energy(&[0.0, 0.0, 0.0, r_min as f64, 0.0, 0.0]);

        assert!((gpu_e - cpu_e).abs() < TOL, "GPU={gpu_e}, CPU={cpu_e}");
    }

    #[test]
    fn test_pair_lj126_gpu_energy_at_sigma() {
        let (_dev, stream) = init_device();
        let coords = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];

        let gpu = PairLJ126GPU::new(
            DeviceBuffer::from_host(&[1.0f32]),
            DeviceBuffer::from_host(&[1.0f32]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            n_items: 1,
            n_atoms: 2,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let gpu_e = gpu.energy(&d_pos, &topo, &stream);

        assert!(gpu_e.abs() < TOL, "expected ~0 at r=sigma, got {gpu_e}");
    }

    #[test]
    fn test_pair_lj126_gpu_gradient() {
        let (_dev, stream) = init_device();
        let coords = vec![0.0f32, 0.0, 0.0, 1.5, 0.3, 0.1];

        let gpu = PairLJ126GPU::new(
            DeviceBuffer::from_host(&[0.5f32]),
            DeviceBuffer::from_host(&[1.0f32]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            n_items: 1,
            n_atoms: 2,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let mut d_grad = DeviceBuffer::<f32>::alloc(6);
        d_grad.zero();
        gpu.gradient(&d_pos, &mut d_grad, &topo, &stream);
        let gpu_g = d_grad.to_host();

        let cpu =
            crate::core::potential_kernels::PairLJ126::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let mut cpu_g = vec![0.0f64; 6];
        cpu.gradient(&[0.0, 0.0, 0.0, 1.5, 0.3, 0.1], &mut cpu_g);

        for i in 0..6 {
            assert!(
                (gpu_g[i] as f64 - cpu_g[i]).abs() < TOL,
                "grad[{i}]: GPU={}, CPU={}",
                gpu_g[i],
                cpu_g[i],
            );
        }
    }

    #[test]
    fn test_pair_lj126_gpu_newton_third() {
        let (_dev, stream) = init_device();
        let coords = vec![0.0f32, 0.0, 0.0, 1.5, 0.3, 0.1];

        let gpu = PairLJ126GPU::new(
            DeviceBuffer::from_host(&[0.5f32]),
            DeviceBuffer::from_host(&[1.0f32]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32]),
            atom_j: DeviceBuffer::from_host(&[1i32]),
            n_items: 1,
            n_atoms: 2,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let mut d_grad = DeviceBuffer::<f32>::alloc(6);
        d_grad.zero();
        gpu.gradient(&d_pos, &mut d_grad, &topo, &stream);
        let g = d_grad.to_host();

        for dim in 0..3 {
            let sum = g[dim] + g[3 + dim];
            assert!((sum as f64).abs() < 1e-5, "dim={dim}: sum={sum}");
        }
    }

    #[test]
    fn test_pair_lj126_gpu_multiple_pairs() {
        let (_dev, stream) = init_device();
        let coords = vec![0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0];

        let gpu = PairLJ126GPU::new(
            DeviceBuffer::from_host(&[1.0f32, 0.5]),
            DeviceBuffer::from_host(&[1.0f32, 1.5]),
        );
        let topo = Topology::Pair {
            atom_i: DeviceBuffer::from_host(&[0i32, 0]),
            atom_j: DeviceBuffer::from_host(&[1i32, 2]),
            n_items: 2,
            n_atoms: 3,
        };
        let d_pos = DeviceBuffer::from_host(&coords);
        let gpu_e = gpu.energy(&d_pos, &topo, &stream);

        let cpu = crate::core::potential_kernels::PairLJ126::new(
            vec![0, 0],
            vec![1, 2],
            vec![1.0, 0.5],
            vec![1.0, 1.5],
        );
        let cpu_e = cpu.energy(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0]);

        assert!((gpu_e - cpu_e).abs() < TOL, "GPU={gpu_e}, CPU={cpu_e}");
    }
}
