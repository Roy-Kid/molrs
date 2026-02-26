use std::marker::PhantomData;
use std::mem;

use super::bindings;

/// Typed GPU device memory with RAII semantics.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Wrap an existing device pointer. The buffer takes ownership and will
    /// call `molrs_free` on drop.
    ///
    /// # Safety
    /// `ptr` must have been allocated with `molrs_malloc` (cudaMalloc) and
    /// must point to at least `len` elements of type T.
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Allocate `n` elements of uninitialized device memory.
    pub fn alloc(n: usize) -> Self {
        let size = n * mem::size_of::<T>();
        let ptr = unsafe { bindings::molrs_malloc(size) as *mut T };
        assert!(!ptr.is_null(), "GPU allocation failed for {} bytes", size);
        Self {
            ptr,
            len: n,
            _marker: PhantomData,
        }
    }

    /// Allocate and upload host data to device.
    pub fn from_host(data: &[T]) -> Self {
        let buf = Self::alloc(data.len());
        let size = data.len() * mem::size_of::<T>();
        unsafe {
            bindings::molrs_memcpy_h2d(
                buf.ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                size,
            );
        }
        buf
    }

    /// Download device memory to a host Vec.
    pub fn to_host(&self) -> Vec<T> {
        let mut host = vec![unsafe { mem::zeroed::<T>() }; self.len];
        let size = self.len * mem::size_of::<T>();
        unsafe {
            bindings::molrs_memcpy_d2h(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                size,
            );
        }
        host
    }

    /// Zero all device memory.
    pub fn zero(&mut self) {
        let size = self.len * mem::size_of::<T>();
        unsafe {
            bindings::molrs_memset_zero(self.ptr as *mut std::ffi::c_void, size);
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                bindings::molrs_free(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}
