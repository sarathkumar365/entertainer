import numpy as np

from entertainer.evaluation import metrics


def test_ndcg_perfect_and_empty():
    rel = {1: 3.0, 2: 2.0, 3: 1.0}
    assert metrics.ndcg_at_k([1, 2, 3], rel, 3) == 1.0
    assert metrics.ndcg_at_k([9, 8, 7], rel, 3) == 0.0
    assert metrics.ndcg_at_k([3, 2, 1], rel, 3) < 1.0


def test_precision_recall_map_mrr():
    ranked = [1, 5, 2, 7]
    positives = {1, 2, 9}
    assert metrics.precision_at_k(ranked, positives, 4) == 0.5
    assert metrics.recall_at_k(ranked, positives, 4) == 2 / 3
    assert 0 < metrics.average_precision(ranked, positives, 4) <= 1
    assert metrics.reciprocal_rank(ranked, positives, 4) == 1.0
    assert metrics.reciprocal_rank([5, 1], positives, 2) == 0.5


def test_novelty_prefers_rare_items():
    popularity = {1: 1000, 2: 10}
    assert metrics.novelty([2], popularity) > metrics.novelty([1], popularity)


def test_intra_list_diversity_detects_duplicates():
    v = np.array([1.0, 0.0])
    w = np.array([0.0, 1.0])
    same = metrics.intra_list_diversity([1, 2], {1: v, 2: v})
    diff = metrics.intra_list_diversity([1, 2], {1: v, 2: w})
    assert diff > same


def test_summarise_reports_standard_error():
    rows = [{"x": 1.0}, {"x": 3.0}]
    out = metrics.summarise(rows)
    assert out["x"] == 2.0
    assert out["x_se"] > 0
