import pandas as pd

from src.recommender.pipelinept1 import initial_rank


def test_initial_rank_orders_predictions_and_limits_results():
    predictions = pd.DataFrame(
        {
            "tmdbId": range(501),
            "predicted_rating": [float(rating) for rating in range(501)],
        }
    )

    result = initial_rank(predictions)

    assert len(result) == 500
    assert result["tmdbId"].iloc[[0, -1]].tolist() == [500, 1]
    assert result["user_rank"].iloc[[0, -1]].tolist() == [1, 500]


def test_initial_rank_uses_zero_z_scores_for_constant_predictions():
    predictions = pd.DataFrame({"predicted_rating": [4.0, 4.0]})

    result = initial_rank(predictions)

    assert result["z_score"].tolist() == [0.0, 0.0]
