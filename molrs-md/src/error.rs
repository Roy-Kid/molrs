use std::fmt;

#[derive(Debug)]
pub enum MDError {
    #[cfg(feature = "cuda")]
    CUDAError(String),
    ForcefieldError(String),
    KernelNotFound {
        category: String,
        name: String,
    },
    #[cfg(feature = "cuda")]
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    ConfigError(String),
    FixError {
        fix: String,
        msg: String,
    },
    DumpError {
        dump: String,
        msg: String,
    },
}

impl fmt::Display for MDError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            Self::CUDAError(msg) => write!(f, "CUDA error: {}", msg),
            Self::ForcefieldError(msg) => write!(f, "forcefield error: {}", msg),
            Self::KernelNotFound { category, name } => {
                write!(f, "no kernel found for {}/{}", category, name)
            }
            #[cfg(feature = "cuda")]
            Self::OutOfMemory {
                requested,
                available,
            } => write!(
                f,
                "out of GPU memory: requested {} bytes, available {} bytes",
                requested, available
            ),
            Self::ConfigError(msg) => write!(f, "config error: {}", msg),
            Self::FixError { fix, msg } => write!(f, "fix '{}' error: {}", fix, msg),
            Self::DumpError { dump, msg } => write!(f, "dump '{}' error: {}", dump, msg),
        }
    }
}

impl std::error::Error for MDError {}
