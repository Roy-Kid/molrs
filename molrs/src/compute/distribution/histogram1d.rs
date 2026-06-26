//! Shared 1-D histogram with probability-density normalization.
//!
//! Ported from TRAVIS `src/df.cpp` (`CDF::AddToBin(double)` and `CDF::Create`,
//! commit 220729): linear binning with `m_fFac = resolution / (max - min)`,
//! out-of-range samples skipped (TRAVIS `m_fSkipEntries`), and running
//! sum / sum-of-squares / input min-max bookkeeping. The probability-density
//! normalization (∫ p dx = 1) is the molrs reading of TRAVIS's binned
//! distribution divided by the total entry count and bin width.
//!
//! This is the single 1-D histogram implementation behind every geometric
//! distribution function (ADF, DDF, distance DF).

use molrs::types::F;
use ndarray::Array1;

/// A linear 1-D histogram over `[min, max]` with `n_bins` equal-width bins.
///
/// Mirrors TRAVIS `CDF`: samples outside `[min, max]` are counted as skipped
/// (not binned), and the in-range count plus running statistics are retained.
#[derive(Debug, Clone)]
pub struct Histogram1d {
    n_bins: usize,
    min: F,
    max: F,
    bin_width: F,
    /// `n_bins / (max - min)` — TRAVIS `m_fFac`.
    fac: F,
    counts: Array1<F>,
    /// In-range binned entries (TRAVIS `m_fBinEntries`).
    binned: F,
    /// Out-of-range entries (TRAVIS `m_fSkipEntries`).
    skipped: F,
    sum: F,
    sq_sum: F,
}

impl Histogram1d {
    /// Create a histogram with `n_bins` bins spanning `[min, max]`.
    ///
    /// `n_bins` must be ≥ 1 and `max` strictly greater than `min`; callers
    /// (the observables) validate their own physical ranges before this point.
    pub fn new(n_bins: usize, min: F, max: F) -> Self {
        debug_assert!(n_bins >= 1, "histogram needs at least one bin");
        debug_assert!(max > min, "histogram max must exceed min");
        let bin_width = (max - min) / n_bins as F;
        Self {
            n_bins,
            min,
            max,
            bin_width,
            fac: n_bins as F / (max - min),
            counts: Array1::zeros(n_bins),
            binned: 0.0,
            skipped: 0.0,
            sum: 0.0,
            sq_sum: 0.0,
        }
    }

    /// Bin one sample (weight 1). Ported from TRAVIS `CDF::AddToBin(double d)`:
    /// values outside `[min, max]` are skipped, otherwise the bin index is
    /// `floor((d - min) * fac)` clamped to the last bin.
    pub fn add(&mut self, d: F) {
        self.add_weighted(d, 1.0);
    }

    /// Bin one sample with an explicit weight (TRAVIS `CDF::AddToBin(d, v)`).
    pub fn add_weighted(&mut self, d: F, w: F) {
        self.sum += d * w;
        self.sq_sum += d * d * w;
        if d < self.min || d > self.max {
            self.skipped += w;
            return;
        }
        // TRAVIS: ip = (int)((d - min) * fac); the d == max edge lands on
        // bin n_bins and is folded back into the last bin.
        let mut ip = ((d - self.min) * self.fac) as usize;
        if ip >= self.n_bins {
            ip = self.n_bins - 1;
        }
        self.counts[ip] += w;
        self.binned += w;
    }

    /// Bin edges (`n_bins + 1` values).
    pub fn edges(&self) -> Array1<F> {
        Array1::from_iter((0..=self.n_bins).map(|i| self.min + i as F * self.bin_width))
    }

    /// Bin centers (`n_bins` values).
    pub fn centers(&self) -> Array1<F> {
        Array1::from_iter((0..self.n_bins).map(|i| self.min + (i as F + 0.5) * self.bin_width))
    }

    /// Raw per-bin counts.
    pub fn counts(&self) -> &Array1<F> {
        &self.counts
    }

    /// Total in-range entries (sum of `counts`).
    pub fn binned(&self) -> F {
        self.binned
    }

    pub fn bin_width(&self) -> F {
        self.bin_width
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Probability density normalized so the trapezoidal integral over the bin
    /// range is 1: `density[i] = counts[i] / (binned * bin_width)`. Returns all
    /// zeros when nothing was binned.
    pub fn density(&self) -> Array1<F> {
        if self.binned <= 0.0 {
            return Array1::zeros(self.n_bins);
        }
        let denom = self.binned * self.bin_width;
        self.counts.mapv(|c| c / denom)
    }
}

/// Renormalize an arbitrary per-bin weight array to unit integral over bins of
/// width `bin_width` (∫ p dx = 1). Used for the sin θ-corrected ADF, whose raw
/// weights are `density / sin θ` before renormalization.
pub fn renormalize_density(weights: &Array1<F>, bin_width: F) -> Array1<F> {
    let total: F = weights.sum() * bin_width;
    if total <= 0.0 || !total.is_finite() {
        return Array1::zeros(weights.len());
    }
    weights.mapv(|w| w / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_matches_travis_indexing() {
        let mut h = Histogram1d::new(4, 0.0, 4.0); // bin width 1
        h.add(0.5); // bin 0
        h.add(1.5); // bin 1
        h.add(3.9); // bin 3
        h.add(4.0); // edge → folded into last bin (bin 3)
        assert_eq!(h.counts()[0], 1.0);
        assert_eq!(h.counts()[1], 1.0);
        assert_eq!(h.counts()[3], 2.0);
        assert_eq!(h.binned(), 4.0);
    }

    #[test]
    fn out_of_range_is_skipped_not_binned() {
        let mut h = Histogram1d::new(4, 0.0, 4.0);
        h.add(-1.0);
        h.add(5.0);
        assert_eq!(h.binned(), 0.0);
        assert!(h.counts().iter().all(|&c| c == 0.0));
    }

    #[test]
    fn density_integrates_to_one() {
        let mut h = Histogram1d::new(10, 0.0, 10.0);
        for i in 0..100 {
            h.add((i % 10) as F + 0.5);
        }
        let d = h.density();
        let integral: F = d.iter().map(|&p| p * h.bin_width()).sum();
        assert!((integral - 1.0).abs() < 1e-12, "integral = {integral}");
    }
}
