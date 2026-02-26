use std::ops::BitOr;

/// Bitmask indicating which execution stages a Fix participates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageMask(u8);

impl StageMask {
    pub const NONE: StageMask = StageMask(0);
    pub const INITIAL_INTEGRATE: StageMask = StageMask(1 << 0);
    pub const POST_INTEGRATE: StageMask = StageMask(1 << 1);
    pub const PRE_FORCE: StageMask = StageMask(1 << 2);
    pub const POST_FORCE: StageMask = StageMask(1 << 3);
    pub const FINAL_INTEGRATE: StageMask = StageMask(1 << 4);
    pub const END_OF_STEP: StageMask = StageMask(1 << 5);

    pub fn contains(self, other: StageMask) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn union(self, other: StageMask) -> StageMask {
        StageMask(self.0 | other.0)
    }
}

impl BitOr for StageMask {
    type Output = StageMask;
    fn bitor(self, rhs: StageMask) -> StageMask {
        self.union(rhs)
    }
}
