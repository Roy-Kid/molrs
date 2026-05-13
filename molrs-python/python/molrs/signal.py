"""Signal processing utilities (FFT-based ACF, window functions, frequency grids).

All computation is in Rust; these are thin Python re-exports.
"""

from .molrs import signal_acf_fft as acf_fft
from .molrs import signal_apply_window as apply_window
from .molrs import signal_frequency_grid as frequency_grid
