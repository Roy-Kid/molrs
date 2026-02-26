use std::collections::HashMap;
use std::path::{Path, PathBuf};

use molrs::core::frame::Frame;
use molrs::core::types::F;
use molrs::io::zarr::{StoreData, ZarrStoreWriter};

use crate::error::MDError;
use crate::run::dump::Dump;
use crate::run::state::MDState;

const DEFAULT_TIME_CHUNK_SIZE: u64 = u32::MAX as u64;

fn normalize_chunk_size(chunk_size: u64) -> u64 {
    if chunk_size == 0 {
        DEFAULT_TIME_CHUNK_SIZE
    } else {
        chunk_size
    }
}

/// DumpZarr writes trajectory data to a Zarr store during dynamics.
pub struct DumpZarr {
    path: PathBuf,
    every: usize,
    positions: bool,
    velocities: bool,
    forces: bool,
    scalars: Vec<String>,
    box_h: bool,
    chunk_size: u64,
    writer: Option<ZarrStoreWriter>,
}

/// Builder for DumpZarr configuration.
pub struct DumpZarrBuilder {
    path: PathBuf,
    every: usize,
    positions: bool,
    velocities: bool,
    forces: bool,
    scalars: Vec<String>,
    box_h: bool,
    chunk_size: u64,
}

impl DumpZarrBuilder {
    pub fn every(mut self, n: usize) -> Self {
        self.every = n;
        self
    }

    pub fn with_positions(mut self) -> Self {
        self.positions = true;
        self
    }

    pub fn with_velocities(mut self) -> Self {
        self.velocities = true;
        self
    }

    pub fn with_forces(mut self) -> Self {
        self.forces = true;
        self
    }

    pub fn with_scalars(mut self, names: &[&str]) -> Self {
        self.scalars = names.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_box_h(mut self) -> Self {
        self.box_h = true;
        self
    }

    pub fn chunk_size(mut self, c: u64) -> Self {
        self.chunk_size = normalize_chunk_size(c);
        self
    }

    pub fn build(self) -> DumpZarr {
        DumpZarr {
            path: self.path,
            every: self.every,
            positions: self.positions,
            velocities: self.velocities,
            forces: self.forces,
            scalars: self.scalars,
            box_h: self.box_h,
            chunk_size: self.chunk_size,
            writer: None,
        }
    }
}

impl DumpZarr {
    /// Create a DumpZarr builder targeting a Zarr store at the given path.
    pub fn zarr(path: impl AsRef<Path>) -> DumpZarrBuilder {
        DumpZarrBuilder {
            path: path.as_ref().to_path_buf(),
            every: 1,
            positions: false,
            velocities: false,
            forces: false,
            scalars: Vec::new(),
            box_h: false,
            chunk_size: DEFAULT_TIME_CHUNK_SIZE,
        }
    }
}

impl Dump for DumpZarr {
    fn name(&self) -> &str {
        "dump/zarr"
    }

    fn every(&self) -> usize {
        self.every
    }

    fn setup(&mut self, frame: &Frame, _state: &MDState) -> Result<(), MDError> {
        let mut builder = ZarrStoreWriter::builder(&self.path);
        if self.positions {
            builder = builder.with_positions();
        }
        if self.velocities {
            builder = builder.with_velocities();
        }
        if self.forces {
            builder = builder.with_forces();
        }
        if !self.scalars.is_empty() {
            let refs: Vec<&str> = self.scalars.iter().map(|s| s.as_str()).collect();
            builder = builder.with_scalars(&refs);
        }
        if self.box_h {
            builder = builder.with_box_h();
        }
        builder = builder.chunk_size(self.chunk_size);

        let writer = builder.create(frame).map_err(|e| MDError::DumpError {
            dump: self.name().into(),
            msg: e.to_string(),
        })?;

        self.writer = Some(writer);
        Ok(())
    }

    fn write(&mut self, state: &MDState) -> Result<(), MDError> {
        let name = self.name().to_owned();
        let writer = self.writer.as_mut().ok_or_else(|| MDError::DumpError {
            dump: name.clone(),
            msg: "writer not initialized (setup not called)".into(),
        })?;

        let step = state.step as i64;
        let time = state.step as f64 * state.dt as f64;

        // Convert F → f32 for trajectory storage
        let pos_f32: Vec<f32>;
        let positions = if self.positions {
            pos_f32 = state.x.iter().map(|&v| v as f32).collect();
            Some(pos_f32.as_slice())
        } else {
            None
        };

        let vel_f32: Vec<f32>;
        let velocities = if self.velocities {
            vel_f32 = state.v.iter().map(|&v| v as f32).collect();
            Some(vel_f32.as_slice())
        } else {
            None
        };

        let frc_f32: Vec<f32>;
        let forces = if self.forces {
            frc_f32 = state.f.iter().map(|&v| v as f32).collect();
            Some(frc_f32.as_slice())
        } else {
            None
        };

        // Scalars are f64 in Zarr wire format
        let mut scalars = HashMap::new();
        for scalar_name in &self.scalars {
            let val: f64 = match scalar_name.as_str() {
                "pe" => state.pe as f64,
                "ke" => state.ke as f64,
                other => state.scalars.get(other).copied().unwrap_or(0.0 as F) as f64,
            };
            scalars.insert(scalar_name.clone(), val);
        }

        let box_h = if self.box_h {
            state.simbox.as_ref().map(|sb| {
                let h = sb.h_view();
                let data: Vec<f32> = h.iter().map(|&v| v as f32).collect();
                data
            })
        } else {
            None
        };

        let data = StoreData {
            positions,
            velocities,
            forces,
            scalars,
            box_h: box_h.as_deref(),
        };

        writer
            .append(step, time, &data)
            .map_err(|e| MDError::DumpError {
                dump: self.name().into(),
                msg: e.to_string(),
            })?;

        Ok(())
    }

    fn cleanup(&mut self) -> Result<(), MDError> {
        if let Some(writer) = self.writer.take() {
            writer.close().map_err(|e| MDError::DumpError {
                dump: self.name().into(),
                msg: e.to_string(),
            })?;
        }
        Ok(())
    }
}
