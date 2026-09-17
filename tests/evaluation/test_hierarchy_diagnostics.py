import numpy as np

from apexid.evaluation import compute_hierarchical_similarity_diagnostics


def test_hierarchy_diagnostics_reports_ordered_distributions_and_hardest_tail():
    embeddings = np.asarray([
        [1.0, 0.0], [0.99, 0.01],
        [0.7, 0.7], [0.69, 0.71],
        [-1.0, 0.0], [-0.99, 0.01],
        [-0.7, 0.7], [-0.69, 0.71],
    ])
    identities = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    species = np.asarray(["a"] * 4 + ["b"] * 4)
    result = compute_hierarchical_similarity_diagnostics(
        embeddings, identities, species, max_pairs=100, chunk_size=3, random_state=1
    )
    assert result["hierarchy_diag/same_identity_mean"] > result[
        "hierarchy_diag/different_identity_same_species_mean"
    ]
    assert result["hierarchy_diag/different_identity_same_species_mean"] > result[
        "hierarchy_diag/different_species_mean"
    ]
    assert result["hierarchy_diag/delta_identity"] > 0
    assert result["hierarchy_diag/delta_species"] > 0
    assert result["hierarchy_diag/hardest_same_species_gap_count"] == 8
