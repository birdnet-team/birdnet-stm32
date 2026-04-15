"""Score pooling strategies for aggregating chunk-level predictions to file-level."""

import numpy as np


def lme_pooling(scores: np.ndarray, beta: float = 10.0) -> np.ndarray:
    """Log-Mean-Exponential pooling over chunks.

    Computes: pooled = log(mean(exp(beta * s_i))) / beta

    Args:
        scores: [N_chunks, C] chunk scores in [0, 1].
        beta: Temperature; beta->0 approximates mean, beta->inf approximates max.

    Returns:
        [C] pooled scores.
    """
    if scores.size == 0:
        return scores
    m = np.max(beta * scores, axis=0, keepdims=True)
    lme = m + np.log(np.mean(np.exp(beta * scores - m), axis=0, keepdims=True) + 1e-12)
    return (lme / beta).ravel()


def pool_scores(chunk_scores: np.ndarray, method: str = "average", beta: float = 10.0) -> np.ndarray:
    """Pool chunk-level scores [N, C] to file-level [C].

    Args:
        chunk_scores: Array of shape [N_chunks, C].
        method: 'avg'|'mean'|'average' | 'max' | 'lme' (log-mean-exp).
        beta: Temperature for 'lme' pooling.

    Returns:
        [C] pooled scores.
    """
    method = method.lower()
    if chunk_scores.ndim != 2:
        raise ValueError("chunk_scores must be [N_chunks, C]")
    if chunk_scores.shape[0] == 0:
        return np.zeros((chunk_scores.shape[1],), dtype=np.float32)
    if method in ("avg", "mean", "average"):
        return np.mean(chunk_scores, axis=0)
    if method == "max":
        return np.max(chunk_scores, axis=0)
    if method in ("lme", "log_mean_exp", "log_mean_exponential"):
        return lme_pooling(chunk_scores, beta=beta)
    raise ValueError(f"Unsupported pooling method: {method}")
