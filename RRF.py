import numpy as np

def rrf(rank_lists, alpha=60, default_rank=1000):
    """
    Reciprocal Rank Fusion (RRF).

    Takes multiple rank lists and produces a single fused ranking.
    Only considers RANK POSITION, not raw scores — making it safe to
    combine results from different scoring systems.

    Args:
        rank_lists: List of [(item_id, rank), ...] lists
        alpha: Smoothing constant (default=60, from Cormack et al. 2009)
        default_rank: Rank assigned to items not in a list (high = penalty)

    Returns:
        Sorted list of (item_id, rrf_score) tuples
    """
    all_items = set(item for rank_list in rank_lists for item, _ in rank_list)
    item_to_index = {item: idx for idx, item in enumerate(all_items)}

    # Matrix: rows = items, cols = rank lists, filled with default_rank
    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)

    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank

    # RRF formula: sum of 1/(alpha + rank) across all lists
    rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)

    sorted_indices = np.argsort(-rrf_scores)  # Descending
    sorted_items = [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]

    return sorted_items