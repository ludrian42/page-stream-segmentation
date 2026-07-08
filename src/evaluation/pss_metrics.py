"""
PSS evaluation metrics — aligned with the DocSplit benchmark paper.

Three metrics reported:
  1. Boundary F1  — precision / recall / F1 for detected document start positions
  2. Grouping     — Rand Index + V-measure (homogeneity, completeness)
                    corresponds to "Grupowanie" in the thesis
  3. Ordering     — fraction of pages whose within-doc position is correct
                    corresponds to "Kolejność" in the thesis

All three operate on a flat list of pages with their ground-truth and
predicted group assignments.  The pairwise approach only predicts group
membership (no doc-type classification), so the full SPK metric is not
computed here; see notes at the bottom of this file.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score,
    rand_score,
    homogeneity_completeness_v_measure,
)


# ---------------------------------------------------------------------------
# Helper: reconstruct predicted groups from pairwise boundary predictions
# ---------------------------------------------------------------------------

def boundaries_to_groups(boundaries: List[bool]) -> List[int]:
    """Convert a list of N-1 boundary flags into N integer group labels.

    boundaries[i] == True  →  page i+1 starts a new group.

    Example:
        boundaries = [False, True, False]
        returns    = [0, 0, 1, 1]   (4 pages, split after page 1)
    """
    groups = [0]
    current = 0
    for is_boundary in boundaries:
        if is_boundary:
            current += 1
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Metric 1 — Boundary F1
# ---------------------------------------------------------------------------

def boundary_f1(
    gt_labels: List[str],
    pred_boundaries: List[bool],
) -> Dict[str, float]:
    """Compute precision, recall and F1 for boundary detection.

    Args:
        gt_labels:       Ground-truth group ID per page  (length N).
        pred_boundaries: Model's boundary flags          (length N-1).
                         pred_boundaries[i] == True → boundary before page i+1.

    Returns:
        dict with keys: precision, recall, f1, tp, fp, fn
    """
    assert len(gt_labels) == len(pred_boundaries) + 1, (
        f"Expected {len(pred_boundaries)+1} labels for {len(pred_boundaries)} boundaries"
    )

    gt_bnd = [gt_labels[i] != gt_labels[i - 1] for i in range(1, len(gt_labels))]

    tp = sum(g and p for g, p in zip(gt_bnd, pred_boundaries))
    fp = sum((not g) and p for g, p in zip(gt_bnd, pred_boundaries))
    fn = sum(g and (not p) for g, p in zip(gt_bnd, pred_boundaries))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Metric 2 — Grouping (Rand Index + V-measure)
# ---------------------------------------------------------------------------

def grouping_metrics(
    gt_labels: List[str],
    pred_boundaries: List[bool],
) -> Dict[str, float]:
    """Compute clustering quality of the predicted document groups.

    Rand Index:  fraction of page-pairs correctly clustered together or apart.
    V-measure:   harmonic mean of homogeneity and completeness.

    Args:
        gt_labels:       Ground-truth group ID per page  (length N).
        pred_boundaries: Model's boundary flags          (length N-1).

    Returns:
        dict with keys: rand_index, homogeneity, completeness, v_measure
    """
    pred_labels = boundaries_to_groups(pred_boundaries)

    ri = rand_score(gt_labels, pred_labels)
    h, c, v = homogeneity_completeness_v_measure(gt_labels, pred_labels)

    return {
        "rand_index":   float(ri),
        "homogeneity":  float(h),
        "completeness": float(c),
        "v_measure":    float(v),
    }


# ---------------------------------------------------------------------------
# Metric 3 — Ordering
# ---------------------------------------------------------------------------

def ordering_metric(
    gt_ordinals: List[int],
    pred_boundaries: List[bool],
) -> float:
    """Fraction of pages whose within-doc page position is correctly predicted.

    For a page at position i within its predicted group, the predicted ordinal
    is its 1-based rank inside that group.  We compare this to the ground-truth
    within-doc ordinal (local_doc_id_page_ordinal in the benchmark CSV).

    Args:
        gt_ordinals:     Ground-truth within-doc page number per page (1-based).
        pred_boundaries: Model's boundary flags (length N-1).

    Returns:
        Fraction of pages with correct ordering (float in [0, 1]).
    """
    pred_labels = boundaries_to_groups(pred_boundaries)

    # Compute predicted ordinal per page
    group_counters: Dict[int, int] = {}
    pred_ordinals: List[int] = []
    for grp in pred_labels:
        group_counters[grp] = group_counters.get(grp, 0) + 1
        pred_ordinals.append(group_counters[grp])

    correct = sum(g == p for g, p in zip(gt_ordinals, pred_ordinals))
    return correct / len(gt_ordinals) if gt_ordinals else 0.0


# ---------------------------------------------------------------------------
# Aggregate: compute all metrics for a single spliced document
# ---------------------------------------------------------------------------

def evaluate_document(
    gt_labels:    List[str],
    gt_ordinals:  List[int],
    pred_boundaries: List[bool],
) -> Dict[str, float]:
    """Run all three metrics for a single spliced document.

    Args:
        gt_labels:       group_id per page (N values).
        gt_ordinals:     local_doc_id_page_ordinal per page (N values).
        pred_boundaries: pairwise boundary flags (N-1 values).

    Returns:
        Flat dict of all metric values.
    """
    metrics = {}
    metrics.update({f"bnd_{k}": v for k, v in boundary_f1(gt_labels, pred_boundaries).items()})
    metrics.update(grouping_metrics(gt_labels, pred_boundaries))
    metrics["ordering"] = ordering_metric(gt_ordinals, pred_boundaries)
    return metrics


# ---------------------------------------------------------------------------
# Aggregate: macro-average over all spliced documents
# ---------------------------------------------------------------------------

def aggregate_metrics(per_doc_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averaged metrics across all evaluated documents.

    Args:
        per_doc_metrics: List of dicts returned by evaluate_document().

    Returns:
        Single dict with macro-averaged values.
    """
    if not per_doc_metrics:
        return {}

    keys = per_doc_metrics[0].keys()
    return {
        k: float(np.mean([m[k] for m in per_doc_metrics]))
        for k in keys
        if k not in ("tp", "fp", "fn")  # exclude raw counts from averaging
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict[str, float]) -> None:
    """Print a formatted metrics summary (for terminal / notebook output)."""
    print("\n" + "=" * 60)
    print("  PSS EVALUATION RESULTS")
    print("=" * 60)

    print("\nClustering:")
    print(f"  Rand Index   : {metrics.get('rand_index',   0):.4f}")
    print(f"  V-measure    : {metrics.get('v_measure',    0):.4f}")
    print(f"    homogeneity: {metrics.get('homogeneity',  0):.4f}")
    print(f"    completeness:{metrics.get('completeness', 0):.4f}")

    print("\nBoundary Detection (F1):")
    print(f"  Precision    : {metrics.get('bnd_precision', 0):.4f}")
    print(f"  Recall       : {metrics.get('bnd_recall',    0):.4f}")
    print(f"  F1           : {metrics.get('bnd_f1',        0):.4f}")

    print("\nPage Ordering:")
    print(f"  Ordering     : {metrics.get('ordering', 0):.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Notes on SPK (Strona + Podział + Kolejność)
# ---------------------------------------------------------------------------
# The full SPK metric from the DocSplit paper requires predicting THREE things
# for each page: (1) document category / doc_type, (2) which document instance
# the page belongs to, (3) page number within that document.
#
# The pairwise binary approach implemented here covers (2) and (3) but NOT (1),
# because we never ask the model to classify the document type — only whether
# two consecutive pages belong to the same document.
#
# To compute SPK you would need to additionally classify each detected group
# into one of the 13 document categories (a separate classification prompt or
# a dedicated document-type classifier).  This is left as future work and noted
# as a limitation of the pairwise baseline in the thesis.
