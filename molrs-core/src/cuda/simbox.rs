use super::bindings;
use super::device::CUDAStream;
use crate::core::region::simbox::SimBox;

/// GPU-resident simulation box.
///
/// Holds device memory for the box matrix H, its inverse, origin, and PBC flags.
/// GPU kernels (e.g. barostat) can obtain raw device pointers to write directly.
/// The neighbor list reads from this object's device buffers.
pub struct GPUSimBox {
    ptr: *mut bindings::GPUSimBox,
}

unsafe impl Send for GPUSimBox {}

impl GPUSimBox {
    /// Allocate a new GPU SimBox with zeroed device memory.
    pub fn new() -> Self {
        let ptr = unsafe { bindings::molrs_simbox_create() };
        assert!(!ptr.is_null(), "GPUSimBox allocation failed");
        Self { ptr }
    }

    /// Upload box parameters from a CPU-side SimBox.
    /// Used for initialization or when the box is updated on the CPU side.
    pub fn upload_from_host(&self, simbox: &SimBox, stream: &CUDAStream) {
        let h = simbox.h_raw();
        let inv = simbox.inv_raw();
        let origin = simbox.origin_raw();
        let periodic = simbox.pbc_as_int();
        unsafe {
            bindings::molrs_simbox_upload(
                self.ptr,
                h.as_ptr(),
                inv.as_ptr(),
                origin.as_ptr(),
                periodic.as_ptr(),
                stream.as_ptr(),
            );
        }
    }

    /// Recompute H^{-1} from H on the GPU.
    /// Call this after a GPU kernel (e.g. barostat) has modified H in-place.
    pub fn update_inverse(&self, stream: &CUDAStream) {
        unsafe {
            bindings::molrs_simbox_update_inverse(self.ptr, stream.as_ptr());
        }
    }

    /// Raw device pointer to H matrix ([f32; 9], row-major).
    /// GPU kernels can read from or write to this pointer.
    pub fn h_ptr(&self) -> *mut f32 {
        unsafe { bindings::molrs_simbox_h_ptr(self.ptr) }
    }

    /// Raw device pointer to H^{-1} matrix ([f32; 9], row-major).
    pub fn inv_ptr(&self) -> *mut f32 {
        unsafe { bindings::molrs_simbox_inv_ptr(self.ptr) }
    }

    /// Raw device pointer to origin ([f32; 3]).
    pub fn origin_ptr(&self) -> *mut f32 {
        unsafe { bindings::molrs_simbox_origin_ptr(self.ptr) }
    }

    /// Raw device pointer to periodic flags ([i32; 3]).
    pub fn periodic_ptr(&self) -> *mut i32 {
        unsafe { bindings::molrs_simbox_periodic_ptr(self.ptr) }
    }

    /// Opaque handle for passing to C FFI functions.
    pub fn as_ptr(&self) -> *mut bindings::GPUSimBox {
        self.ptr
    }

    /// Const handle for passing to C FFI functions.
    pub fn as_const_ptr(&self) -> *const bindings::GPUSimBox {
        self.ptr as *const _
    }
}

impl Drop for GPUSimBox {
    fn drop(&mut self) {
        unsafe {
            bindings::molrs_simbox_destroy(self.ptr);
        }
    }
}
