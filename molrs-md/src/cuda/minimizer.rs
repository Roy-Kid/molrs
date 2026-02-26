use molrs::core::types::F;
use molrs::cuda::buffer::DeviceBuffer;
use molrs::cuda::device::{CUDADevice, CUDAStream};
use molrs::cuda::potential::GPUPotentials;
use molrs::ffi as bindings;

use crate::backend::MinimizerConfig;
use crate::error::MDError;
use crate::md::MinState;

/// CUDA minimizer — steepest descent on GPU.
pub struct CUDAMinimizer {
    potentials: GPUPotentials,
    positions: DeviceBuffer<f32>,
    gradients: DeviceBuffer<f32>,
    _device: CUDADevice,
    stream: CUDAStream,
    config: MinimizerConfig,
    n_atoms: usize,
}

impl CUDAMinimizer {
    pub(crate) fn new(
        potentials: GPUPotentials,
        device: CUDADevice,
        config: MinimizerConfig,
    ) -> Self {
        Self {
            potentials,
            positions: DeviceBuffer::alloc(0),
            gradients: DeviceBuffer::alloc(0),
            _device: device,
            stream: CUDAStream::default_stream(),
            config,
            n_atoms: 0,
        }
    }

    pub fn load_coordinates(&mut self, coords: &[F]) {
        assert!(
            coords.len() % 3 == 0,
            "coordinates length must be multiple of 3"
        );
        self.n_atoms = coords.len() / 3;
        let n = coords.len();

        // Convert F → f32 for GPU
        let coords_f32: Vec<f32> = coords.iter().map(|&x| x as f32).collect();
        self.positions = DeviceBuffer::from_host(&coords_f32);
        self.gradients = DeviceBuffer::alloc(n);
    }

    pub fn run(&mut self, max_steps: usize) -> Result<MinState, MDError> {
        let steps = if max_steps > 0 {
            max_steps
        } else {
            self.config.max_iter
        };
        let tol = self.config.force_tol as f32;
        let n = self.n_atoms * 3;

        let mut trial_positions = DeviceBuffer::<f32>::alloc(n);
        let mut converged = false;
        let mut step_count = 0;
        let mut n_evals: usize = 0;
        let mut prev_alpha: f32 = 0.1;

        for _ in 0..steps {
            step_count += 1;

            // Compute gradient on GPU
            self.potentials
                .gradient(&self.positions, &mut self.gradients, &self.stream);
            n_evals += 1;

            // Check convergence: |grad|_max < tolerance
            let grad_max = unsafe {
                bindings::molrs_grad_max_norm(
                    self.gradients.as_ptr(),
                    n as i32,
                    self.stream.as_ptr(),
                )
            };
            if grad_max < tol {
                converged = true;
                break;
            }

            // Energy at current position
            let energy_current = self.potentials.energy(&self.positions, &self.stream);
            n_evals += 1;

            // Download gradient for line search step computation
            let grad_host = self.gradients.to_host();
            let grad_sq: f64 = grad_host.iter().map(|&g| (g as f64) * (g as f64)).sum();

            // Armijo backtracking line search
            let mut alpha = (2.0_f32 * prev_alpha).min(1.0 / grad_max);
            let c: f64 = 1e-4;
            let rho: f32 = 0.5;

            let pos_host = self.positions.to_host();
            let mut accepted = false;

            for _ls in 0..40 {
                // Compute trial positions on host, upload
                let trial_host: Vec<f32> = pos_host
                    .iter()
                    .zip(grad_host.iter())
                    .map(|(&p, &g)| p - alpha * g)
                    .collect();
                trial_positions = DeviceBuffer::from_host(&trial_host);

                let energy_trial = self.potentials.energy(&trial_positions, &self.stream);
                n_evals += 1;
                if energy_trial <= energy_current - c * (alpha as f64) * grad_sq {
                    accepted = true;
                    break;
                }
                alpha *= rho;
            }

            if accepted {
                prev_alpha = alpha;
                // Update positions
                let new_pos: Vec<f32> = pos_host
                    .iter()
                    .zip(grad_host.iter())
                    .map(|(&p, &g)| p - alpha * g)
                    .collect();
                self.positions = DeviceBuffer::from_host(&new_pos);
            }
        }

        let energy_f64 = self.potentials.energy(&self.positions, &self.stream);

        Ok(MinState {
            energy: energy_f64 as F,
            n_steps: step_count,
            n_evals,
            converged,
        })
    }
}
