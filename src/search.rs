//! Combining ranked lists with reciprocal rank fusion (RRF).

use std::collections::HashMap;
use std::hash::Hash;

pub const RRF_K: f32 = 60.0;

/// Fuse ranked lists: score(x) = sum over lists of 1 / (RRF_K + rank), rank starting at 1.
pub fn rrf<T: Eq + Hash + Clone>(lists: &[Vec<T>]) -> Vec<(T, f32)> {
    let mut scores: HashMap<T, f32> = HashMap::new();
    let mut order: Vec<T> = Vec::new();
    for list in lists {
        for (rank, item) in list.iter().enumerate() {
            let e = scores.entry(item.clone()).or_insert_with(|| {
                order.push(item.clone());
                0.0
            });
            *e += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(T, f32)> = order.into_iter().map(|t| {
        let s = scores[&t];
        (t, s)
    }).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1)); // stable: ties keep first-seen order
    out
}

/// SigLIP2 base/224's calibrated match probability: sigmoid(exp(logit_scale) * cos + logit_bias),
/// with the two constants read from the model weights.
pub fn photo_probability(cosine: f32) -> f32 {
    const SCALE: f32 = 112.6833; // exp(4.724453)
    const BIAS: f32 = -16.771725;
    1.0 / (1.0 + (-(SCALE * cosine + BIAS)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_rewards_agreement() {
        let fused = rrf(&[vec!["a", "b", "c"], vec!["b", "a", "d"]]);
        let names: Vec<&str> = fused.iter().map(|x| x.0).collect();
        assert_eq!(&names[..2], &["a", "b"]); // tie on score, first-seen order wins
        assert!(fused[0].1 > fused[2].1);
        assert_eq!(names.len(), 4);
    }

    #[test]
    fn photo_probability_is_calibrated() {
        assert!(photo_probability(0.05) < 0.01);
        assert!((photo_probability(0.1488) - 0.5).abs() < 0.01);
        assert!(photo_probability(0.25) > 0.99);
    }
}
