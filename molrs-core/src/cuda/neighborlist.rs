use super::bindings;
use super::buffer::DeviceBuffer;
use super::device::CUDAStream;
use super::simbox::GPUSimBox;

/// GPU-resident neighbor list builder.
///
/// Builds a pair list of atom indices within a cutoff distance using
/// a GPU-resident pipeline (Eastman & Pande, J. Comput. Chem. 2010):
/// cell assignment → radix sort → pair counting → prefix sum → pair collection.
///
/// The pair list is stored in device memory as two DeviceBuffers (pair_i, pair_j).
/// All computation stays on GPU; only two small D2H copies (cell dims + total pairs).
pub struct NeighborListGPU {
    cutoff: f32,
    pair_i: Option<DeviceBuffer<i32>>,
    pair_j: Option<DeviceBuffer<i32>>,
    n_pairs: usize,
}

impl NeighborListGPU {
    pub fn new(cutoff: f64) -> Self {
        Self {
            cutoff: cutoff as f32,
            pair_i: None,
            pair_j: None,
            n_pairs: 0,
        }
    }

    /// Build the neighbor list from positions and a GPU-resident simulation box.
    ///
    /// `positions`: device buffer of atom positions (n_atoms * 3 floats)
    /// `simbox`: GPU-resident simulation box (box params already on device)
    pub fn build(
        &mut self,
        positions: &DeviceBuffer<f32>,
        simbox: &GPUSimBox,
        n_atoms: usize,
        stream: &CUDAStream,
    ) {
        // Free previous pair lists
        self.pair_i = None;
        self.pair_j = None;

        let mut raw_pair_i: *mut i32 = std::ptr::null_mut();
        let mut raw_pair_j: *mut i32 = std::ptr::null_mut();
        let mut n_pairs: i32 = 0;

        unsafe {
            bindings::molrs_neighborlist_build(
                positions.as_ptr(),
                n_atoms as i32,
                simbox.as_const_ptr(),
                self.cutoff,
                &mut raw_pair_i,
                &mut raw_pair_j,
                &mut n_pairs,
                stream.as_ptr(),
            );
        }

        self.n_pairs = n_pairs as usize;

        if !raw_pair_i.is_null() && self.n_pairs > 0 {
            self.pair_i = Some(unsafe { DeviceBuffer::from_raw(raw_pair_i, self.n_pairs) });
            self.pair_j = Some(unsafe { DeviceBuffer::from_raw(raw_pair_j, self.n_pairs) });
        }
    }

    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }

    pub fn pair_i(&self) -> Option<&DeviceBuffer<i32>> {
        self.pair_i.as_ref()
    }

    pub fn pair_j(&self) -> Option<&DeviceBuffer<i32>> {
        self.pair_j.as_ref()
    }

    pub fn cutoff(&self) -> f32 {
        self.cutoff
    }
}
