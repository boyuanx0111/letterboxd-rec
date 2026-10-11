import numpy as np
from scipy.linalg import solve


def cf_predict_new_user(cf_model, user_ratings, candidate_ids, reg=10.0):

    trainset = cf_model.trainset

    movie_vectors = []
    targets = []

    # Get embeddings for movies the user has rated
    for _, row in user_ratings.iterrows():

        try:
            movie_id = trainset.to_inner_iid(int(row["tmdbId"]))
        except (ValueError, TypeError, OverflowError):
            continue

        movie_vectors.append(cf_model.qi[movie_id])

        targets.append(
            float(row["rating"])
            - trainset.global_mean
            - cf_model.bi[movie_id]
        )

    # Fallback if no rated movies exist in CF
    if not movie_vectors:
        return np.full(
            len(candidate_ids),
            trainset.global_mean
        )

    Q = np.asarray(movie_vectors, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    # Estimate user bias and latent embedding
    X = np.column_stack([
        np.ones(len(Q)),
        Q
    ])

    # Ridge regularisation
    reg_matrix = reg * np.eye(X.shape[1])
    reg_matrix[0, 0] = 1.0

    # Calculate X.T @ X without BLAS
    n_features = X.shape[1]

    A = np.empty(
        (n_features, n_features),
        dtype=np.float64
    )

    for i in range(n_features):
        A[i, :] = np.sum(
            X[:, i, None] * X,
            axis=0
        )

    A += reg_matrix

    # Calculate X.T @ y without BLAS
    b = np.sum(
        X * y[:, None],
        axis=0
    )

    # Solve for user bias and embedding
    user_params = solve(
        A,
        b,
        assume_a="pos"
    )

    user_bias = user_params[0]
    user_embedding = user_params[1:]

    # Predict ratings for candidate movies
    predictions = []

    for tmdb_id in candidate_ids:

        try:
            movie_id = trainset.to_inner_iid(int(tmdb_id))

            pred = (
                trainset.global_mean
                + user_bias
                + cf_model.bi[movie_id]
                + np.sum(
                    user_embedding * cf_model.qi[movie_id]
                )
            )

        except (ValueError, TypeError, OverflowError):
            pred = trainset.global_mean + user_bias

        predictions.append(
            np.clip(pred, 0.5, 5.0)
        )

    return np.asarray(predictions)