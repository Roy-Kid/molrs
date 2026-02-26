use super::bindings;

/// Represents a CUDA device (GPU).
pub struct CUDADevice {
    id: i32,
    name: String,
}

impl CUDADevice {
    /// Initialize and select a CUDA device by ID.
    pub fn new(device_id: i32) -> Result<Self, String> {
        let ret = unsafe { bindings::molrs_device_init(device_id) };
        if ret != 0 {
            return Err(format!("failed to initialize CUDA device {}", device_id));
        }

        let mut name_buf = vec![0u8; 256];
        let ret = unsafe {
            bindings::molrs_device_get_name(
                device_id,
                name_buf.as_mut_ptr() as *mut i8,
                name_buf.len() as i32,
            )
        };
        let name = if ret == 0 {
            let nul_pos = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_buf.len());
            String::from_utf8_lossy(&name_buf[..nul_pos]).to_string()
        } else {
            format!("GPU {}", device_id)
        };

        Ok(Self {
            id: device_id,
            name,
        })
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Wrapper around a CUDA stream. NULL stream = default stream.
pub struct CUDAStream {
    ptr: *mut std::ffi::c_void,
}

impl CUDAStream {
    /// Use the default CUDA stream.
    pub fn default_stream() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
        }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

// Default stream is safe to send between threads
unsafe impl Send for CUDAStream {}
