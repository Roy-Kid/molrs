use std::collections::HashMap;

use molrs::core::frame::Frame;
use molrs::core::potential::extract_coords;
use molrs::core::region::SimBox;
use molrs::core::types::F;
use ndarray::Array1;

use crate::error::MDError;

/// Mutable simulation state passed to fixes at each stage.
pub struct MDState {
    /// Positions (3N flat: x0,y0,z0, x1,y1,z1, ...)
    pub x: Vec<F>,
    /// Velocities (3N)
    pub v: Vec<F>,
    /// Forces = -gradient (3N)
    pub f: Vec<F>,
    /// Masses (N)
    pub mass: Vec<F>,
    /// Precomputed 1/mass (N)
    pub inv_mass: Vec<F>,
    /// Number of atoms
    pub n_atoms: usize,
    /// Degrees of freedom (default 3N)
    pub n_dof: usize,
    /// Timestep
    pub dt: F,
    /// Current step number
    pub step: usize,
    /// Simulation box (optional)
    pub simbox: Option<SimBox>,
    /// Potential energy (set by engine after force computation)
    pub pe: F,
    /// Kinetic energy
    pub ke: F,
    /// Named scalar values (for fixes to communicate)
    pub scalars: HashMap<String, F>,
    /// Named per-atom arrays (for fixes to communicate)
    pub per_atom: HashMap<String, Vec<F>>,
}

impl MDState {
    /// Build an MDState from a Frame.
    ///
    /// Extracts positions from atoms block. Velocities are taken from vx/vy/vz
    /// columns if present, otherwise initialized to zero. Masses from "mass"
    /// column if present, otherwise default to 1.0.
    pub fn from_frame(frame: &Frame, dt: F) -> Result<Self, MDError> {
        let x = extract_coords(frame).map_err(MDError::ForcefieldError)?;
        let n_atoms = x.len() / 3;

        let atoms = frame
            .get("atoms")
            .ok_or_else(|| MDError::ForcefieldError("Frame has no \"atoms\" block".into()))?;

        // Velocities: try vx/vy/vz columns, default to zero
        let v = if let (Some(vx), Some(vy), Some(vz)) = (
            atoms.get_f64("vx"),
            atoms.get_f64("vy"),
            atoms.get_f64("vz"),
        ) {
            let mut v = vec![0.0 as F; 3 * n_atoms];
            for i in 0..n_atoms {
                v[3 * i] = vx[i] as F;
                v[3 * i + 1] = vy[i] as F;
                v[3 * i + 2] = vz[i] as F;
            }
            v
        } else {
            vec![0.0; 3 * n_atoms]
        };

        // Masses: try "mass" column, default to 1.0
        let mass: Vec<F> = if let Some(m) = atoms.get_f64("mass") {
            m.iter().map(|&v| v as F).collect()
        } else {
            vec![1.0; n_atoms]
        };

        let inv_mass: Vec<F> = mass.iter().map(|m| 1.0 / m).collect();
        let n_dof = 3 * n_atoms;

        let mut state = MDState {
            x,
            v,
            f: vec![0.0; 3 * n_atoms],
            mass,
            inv_mass,
            n_atoms,
            n_dof,
            dt,
            step: 0,
            simbox: frame.simbox.clone(),
            pe: 0.0,
            ke: 0.0,
            scalars: HashMap::new(),
            per_atom: HashMap::new(),
        };
        state.compute_ke();
        Ok(state)
    }

    /// Compute kinetic energy from velocities and masses: KE = 0.5 * sum(m_i * v_i^2)
    pub fn compute_ke(&mut self) {
        let mut ke: F = 0.0;
        for i in 0..self.n_atoms {
            let vx = self.v[3 * i];
            let vy = self.v[3 * i + 1];
            let vz = self.v[3 * i + 2];
            ke += self.mass[i] * (vx * vx + vy * vy + vz * vz);
        }
        self.ke = 0.5 * ke;
    }

    /// Temperature from kinetic energy: T = 2*KE / (n_dof * kB).
    /// Uses kB = 1.0 (reduced units).
    pub fn temperature(&self) -> F {
        if self.n_dof == 0 {
            return 0.0;
        }
        2.0 * self.ke / self.n_dof as F
    }

    /// Build an output Frame from this state, using the input frame as template.
    pub fn to_frame(&self, template: &Frame) -> Frame {
        let mut out = template.clone();
        let n = self.n_atoms;

        if let Some(atoms) = out.get_mut("atoms") {
            // Positions
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            let mut zs = Vec::with_capacity(n);
            for i in 0..n {
                xs.push(self.x[3 * i]);
                ys.push(self.x[3 * i + 1]);
                zs.push(self.x[3 * i + 2]);
            }
            atoms.insert("x", Array1::from_vec(xs).into_dyn()).ok();
            atoms.insert("y", Array1::from_vec(ys).into_dyn()).ok();
            atoms.insert("z", Array1::from_vec(zs).into_dyn()).ok();

            // Velocities
            let mut vxs = Vec::with_capacity(n);
            let mut vys = Vec::with_capacity(n);
            let mut vzs = Vec::with_capacity(n);
            for i in 0..n {
                vxs.push(self.v[3 * i]);
                vys.push(self.v[3 * i + 1]);
                vzs.push(self.v[3 * i + 2]);
            }
            atoms.insert("vx", Array1::from_vec(vxs).into_dyn()).ok();
            atoms.insert("vy", Array1::from_vec(vys).into_dyn()).ok();
            atoms.insert("vz", Array1::from_vec(vzs).into_dyn()).ok();
        }

        out.meta.insert("pe".into(), self.pe.to_string());
        out.meta.insert("ke".into(), self.ke.to_string());
        out.meta
            .insert("total_energy".into(), (self.pe + self.ke).to_string());
        out.meta.insert("step".into(), self.step.to_string());

        out
    }
}
