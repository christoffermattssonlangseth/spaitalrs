use crate::neighbors::radius_graph;
use anyhow::{bail, Result};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct CompositionRecord {
    pub cell_i: String,
    pub cell_type: String,
    pub fraction: f64,
    pub group: String,
}

#[derive(Serialize)]
pub struct EntropyRecord {
    pub cell_i: String,
    pub entropy: f64,
    pub group: String,
}

/// Compute the per-cell neighbourhood composition within `radius`.
/// For each cell, counts the cell types among its neighbours and returns
/// one row per (cell, neighbour_type) with the fraction of that type.
pub fn compute_composition(
    coords: &[[f64; 2]],
    barcodes: &[String],
    cell_types: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<CompositionRecord>> {
    if coords.len() != barcodes.len() {
        bail!(
            "coords length ({}) does not match barcodes length ({})",
            coords.len(),
            barcodes.len()
        );
    }
    if barcodes.len() != cell_types.len() {
        bail!(
            "barcodes length ({}) does not match cell_types length ({})",
            barcodes.len(),
            cell_types.len()
        );
    }

    // Bidirectional edges so iterating by cell_i gives all neighbours
    let edges = radius_graph(coords, barcodes, radius, group)?;

    let type_map: HashMap<&str, &str> = barcodes
        .iter()
        .zip(cell_types.iter())
        .map(|(b, t)| (b.as_str(), t.as_str()))
        .collect();

    // Build neighbour_map: cell_barcode → list of neighbour cell_types
    let mut neighbour_map: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &edges {
        let nb_type = type_map[edge.cell_j.as_str()];
        neighbour_map
            .entry(edge.cell_i.as_str())
            .or_default()
            .push(nb_type);
    }

    let mut records = Vec::new();

    for barcode in barcodes {
        let neighbours = match neighbour_map.get(barcode.as_str()) {
            Some(nb) => nb,
            None => continue, // isolated cell — skip
        };

        let total = neighbours.len() as f64;
        let mut type_counts: HashMap<&str, usize> = HashMap::new();
        for &t in neighbours {
            *type_counts.entry(t).or_insert(0) += 1;
        }

        for (ct, cnt) in type_counts {
            records.push(CompositionRecord {
                cell_i: barcode.clone(),
                cell_type: ct.to_string(),
                fraction: cnt as f64 / total,
                group: group.to_string(),
            });
        }
    }

    Ok(records)
}

/// Compute Shannon entropy (bits) of the neighbourhood composition per cell.
///
/// Given the output of `compute_composition`, groups rows by `(cell_i, group)` and
/// computes H = −Σₖ pₖ log₂(pₖ).  High entropy means a heterogeneous neighbourhood;
/// low entropy means a homogeneous one.  Cells absent from `composition` are omitted.
pub fn compute_entropy(composition: &[CompositionRecord]) -> Vec<EntropyRecord> {
    // Aggregate fractions by (cell_i, group)
    let mut cell_fracs: HashMap<(&str, &str), Vec<f64>> = HashMap::new();
    for rec in composition {
        cell_fracs
            .entry((rec.cell_i.as_str(), rec.group.as_str()))
            .or_default()
            .push(rec.fraction);
    }

    let mut records: Vec<EntropyRecord> = cell_fracs
        .into_iter()
        .map(|((cell_i, group), fracs)| {
            let entropy = fracs
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.log2())
                .sum();
            EntropyRecord {
                cell_i: cell_i.to_string(),
                entropy,
                group: group.to_string(),
            }
        })
        .collect();

    // Stable sort by cell_i for deterministic output
    records.sort_by(|a, b| a.cell_i.cmp(&b.cell_i));
    records
}

#[cfg(test)]
mod tests {
    use super::{compute_composition, compute_entropy, CompositionRecord};

    #[test]
    fn composition_rejects_non_positive_radius() {
        let coords = [[0.0, 0.0], [1.0, 1.0]];
        let barcodes = vec!["a".to_string(), "b".to_string()];
        let cell_types = vec!["t1".to_string(), "t2".to_string()];

        let err = match compute_composition(&coords, &barcodes, &cell_types, 0.0, "g") {
            Ok(_) => panic!("expected invalid radius error"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("radius must be a finite value > 0"));
    }

    #[test]
    fn entropy_uniform_composition_is_max() {
        // 3 equal fractions → H = log₂(3) ≈ 1.585 bits
        let comp = vec![
            CompositionRecord {
                cell_i: "c1".to_string(),
                cell_type: "A".to_string(),
                fraction: 1.0 / 3.0,
                group: "g".to_string(),
            },
            CompositionRecord {
                cell_i: "c1".to_string(),
                cell_type: "B".to_string(),
                fraction: 1.0 / 3.0,
                group: "g".to_string(),
            },
            CompositionRecord {
                cell_i: "c1".to_string(),
                cell_type: "C".to_string(),
                fraction: 1.0 / 3.0,
                group: "g".to_string(),
            },
        ];
        let records = compute_entropy(&comp);
        assert_eq!(records.len(), 1);
        let expected = (3.0f64).log2();
        assert!(
            (records[0].entropy - expected).abs() < 1e-10,
            "expected H={expected}, got {}",
            records[0].entropy
        );
    }

    #[test]
    fn entropy_pure_composition_is_zero() {
        // One cell type with fraction 1.0 → H = 0
        let comp = vec![CompositionRecord {
            cell_i: "c1".to_string(),
            cell_type: "A".to_string(),
            fraction: 1.0,
            group: "g".to_string(),
        }];
        let records = compute_entropy(&comp);
        assert_eq!(records.len(), 1);
        assert!(records[0].entropy.abs() < 1e-10, "expected H=0 for pure neighbourhood");
    }
}
