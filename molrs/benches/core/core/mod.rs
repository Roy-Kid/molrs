pub mod frame;
pub mod graph;
pub mod topology;
// `potential` bench was left stale by the 2026-Q1 split — potentials
// moved to molrs-ff. Re-home the bench into molrs-ff/benches/ in a
// follow-up; keep it out of compile for now.
// pub mod potential;
pub mod region;
