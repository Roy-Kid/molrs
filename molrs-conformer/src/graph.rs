//! Small graph helpers shared by the distance-geometry paths.

use std::collections::VecDeque;

/// All-pairs topological distance matrix (number of bonds between atoms) over a
/// plain adjacency list, computed by a BFS from each node. Unreachable pairs
/// are left as [`usize::MAX`].
///
/// Both the legacy first-principles bounds (`distance_geometry`) and the
/// RDKit-aligned bounds (`distgeom::bounds`) need this matrix; keeping one
/// implementation avoids the two copies drifting apart.
pub(crate) fn bfs_distance_matrix(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for start in 0..n {
        let row = &mut dist[start];
        row[start] = 0;
        let mut queue = VecDeque::from([start]);
        while let Some(i) = queue.pop_front() {
            let d = row[i];
            for &j in &adjacency[i] {
                if row[j] == usize::MAX {
                    row[j] = d + 1;
                    queue.push_back(j);
                }
            }
        }
    }
    dist
}
