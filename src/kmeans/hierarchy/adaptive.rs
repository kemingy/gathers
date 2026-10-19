//! Shared pilot calibration and checked probing for training and public assignment.

use rand::Rng;

use crate::distance::Distance;
use crate::kmeans::hierarchy::policy::{DEFAULT_PERCENTILE, DEFAULT_PILOTS, budget};
use crate::kmeans::hierarchy::{Hierarchy, ProbeOptions};
use crate::sampling::sample_indices;
use crate::utils::try_normalize_rows;

pub(crate) const GROUPED_ROWS: usize = 32_768;

#[derive(Default)]
pub(crate) struct ProbeWorkspace {
    pub(crate) pilots: Vec<f32>,
    pub(crate) pilot_labels: Vec<u32>,
    pub(crate) required: Vec<usize>,
}

impl ProbeWorkspace {
    /// Calibrate once over the entire incoming batch, before bounded query processing.
    /// Cosine pilots are normalized independently of the full query buffer.
    pub(crate) fn calibrate<R: Rng + ?Sized>(
        &mut self,
        hierarchy: &Hierarchy,
        vectors: &[f32],
        distance: Distance,
        rng: &mut R,
    ) -> Option<usize> {
        let rows = vectors.len() / hierarchy.dim;
        if rows == 0 {
            return None;
        }
        let indices = sample_indices(rows, DEFAULT_PILOTS.min(rows), rng);
        self.pilots.clear();
        for index in indices {
            self.pilots
                .extend_from_slice(&vectors[index * hierarchy.dim..(index + 1) * hierarchy.dim]);
        }
        let distance = if distance == Distance::Cosine {
            try_normalize_rows(&mut self.pilots, hierarchy.dim, false).ok()?;
            Distance::NegativeDotProduct
        } else {
            distance
        };
        let pilots = self.pilots.len() / hierarchy.dim;
        self.pilot_labels.resize(pilots, 0);
        self.required.resize(pilots, 0);
        hierarchy.calibrate_batched(
            &self.pilots,
            &mut self.pilot_labels,
            &mut self.required,
            distance,
        );
        if self
            .pilot_labels
            .iter()
            .any(|&label| label as usize >= hierarchy.leaf_order.len())
            || self.required.contains(&0)
        {
            return None;
        }
        let probes = budget(&self.required, DEFAULT_PERCENTILE);
        Some(probes)
    }
}

impl Hierarchy {
    pub(crate) fn probe_checked(
        &self,
        vectors: &[f32],
        labels: &mut [u32],
        options: ProbeOptions<'_>,
    ) -> bool {
        self.probe_batched(vectors, labels, options, GROUPED_ROWS);
        labels
            .iter()
            .all(|&label| (label as usize) < self.leaf_order.len())
    }
}
