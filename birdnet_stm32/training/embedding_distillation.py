"""Teacher embedding distillation: pull the student's embedding toward the teacher's.

Soft labels hand the student the teacher's scores for our own classes. The
teacher's embedding carries more: what a window sounds like across its whole
vocabulary, including species outside our schema, and a signal for the
non-bird classes and the noise recordings, for which it has no scores at all.

A training-only linear projector maps the student's pooled embedding (``gap``)
to the teacher's width, and a cosine loss pulls the two together. The
projector never reaches the checkpoint: the deployed graph is the student
alone, unchanged. Cosine rather than MSE because the teacher's embeddings are
non-negative pooled activations whose norm varies by a factor of ~2.6 between
windows; an MSE would spend most of its effort on the norm.

Chunks with no usable teacher window, and chunks mixup mixed, carry
``valid = 0`` from the loader and add nothing to this loss.
"""

import numpy as np
import tensorflow as tf

EMBEDDING_LAYER = "gap"
_EPS = 1e-8


def masked_cosine_distance(projected: tf.Tensor, teacher: tf.Tensor, valid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Mean ``1 - cos`` over the valid rows, and the per-row cosine.

    Args:
        projected: ``[B, D]`` projected student embeddings.
        teacher: ``[B, D]`` teacher embeddings (any scale).
        valid: ``[B]`` 1 where the row counts, 0 where it does not.

    Returns:
        ``(loss, cosine)``: the scalar mean distance over valid rows (0 when
        none are valid) and the ``[B]`` cosine similarities.
    """
    projected = tf.cast(projected, tf.float32)
    teacher = tf.cast(teacher, tf.float32)
    valid = tf.cast(valid, tf.float32)
    cosine = tf.reduce_sum(
        tf.math.l2_normalize(projected, axis=-1, epsilon=_EPS) * tf.math.l2_normalize(teacher, axis=-1, epsilon=_EPS),
        axis=-1,
    )
    loss = tf.math.divide_no_nan(tf.reduce_sum((1.0 - cosine) * valid), tf.reduce_sum(valid))
    return loss, cosine


class EmbeddingDistilledModel(tf.keras.Model):
    """Train a student on its labels plus a cosine loss to the teacher's embedding.

    Called, it is the student: same inputs, same outputs, so validation and
    checkpoint selection see exactly the deployed graph. Only ``train_step``
    differs; it expects labels ``(y, teacher_embedding, valid)``.

    Checkpoint the ``student`` (``train_model(checkpoint_model=...)``), not this
    wrapper: the projector exists for training only.

    Args:
        student: The model being trained. Its variables are shared.
        teacher_dim: Width of the teacher embedding.
        weight: Weight of the cosine distance against the supervised loss.
    """

    def __init__(self, student: tf.keras.Model, teacher_dim: int, weight: float):
        super().__init__(inputs=student.inputs, outputs=student.outputs, name=student.name)
        if weight <= 0:
            raise ValueError(f"embedding distillation weight must be positive, got {weight}")
        features = student.get_layer(EMBEDDING_LAYER).output
        self.embedder = tf.keras.Model(student.inputs, [student.outputs[0], features], name="student_with_embedding")
        self.projector = tf.keras.layers.Dense(int(teacher_dim), name="teacher_projector", dtype="float32")
        self.projector.build((None, int(features.shape[-1])))
        self.embedding_weight = float(weight)
        self.cosine_metric = tf.keras.metrics.Mean(name="teacher_cosine")

    def train_step(self, data):
        """The default Keras step, plus the embedding term."""
        x, (y, teacher, valid) = data
        with tf.GradientTape() as tape:
            y_pred, features = self.embedder(x, training=True)
            supervised = self.compute_loss(x=x, y=y, y_pred=y_pred, training=True)
            distance, cosine = masked_cosine_distance(self.projector(features), teacher, valid)
            loss = supervised + self.embedding_weight * distance
            self._loss_tracker.update_state(loss, sample_weight=tf.shape(y)[0])
            scaled = self.optimizer.scale_loss(loss)
        variables = self.trainable_weights
        self.optimizer.apply_gradients(zip(tape.gradient(scaled, variables), variables, strict=True))
        self.cosine_metric.update_state(cosine, sample_weight=valid)
        return self.compute_metrics(x, y, y_pred)

    def test_step(self, data):
        """The default step; validation carries no teacher embeddings, so no cosine."""
        logs = super().test_step(data)
        logs.pop(self.cosine_metric.name, None)
        return logs


def fit_projector(
    model: EmbeddingDistilledModel,
    batches,
    holdout: float = 0.2,
    ridges: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
) -> dict[str, float]:
    """Initialize the projector by ridge regression, for a student that is already trained.

    A random projector on a trained student starts the cosine loss near its
    maximum, and its first gradients reach the backbone before the projector has
    learned anything. Solved in closed form on the student's current
    embeddings, the projector starts where the two spaces already agree.

    A few thousand rows against a 1024 x 1280 map overfit easily, so the ridge
    strength is chosen on held-out rows, and the projector is then refitted on
    all of them.

    Args:
        model: The wrapper whose projector is set.
        batches: Iterable of training batches ``(x, (y, teacher, valid))``.
        holdout: Fraction of the valid rows held out to choose the ridge strength.
        ridges: Candidate ridge strengths, relative to the mean feature variance.

    Returns:
        Row counts, the chosen ridge strength and its held-out mean cosine.
    """
    feats, targets = [], []
    for x, (_, teacher, valid) in batches:
        _, features = model.embedder(x, training=False)
        keep = np.asarray(valid) > 0.5
        feats.append(np.asarray(features, dtype=np.float64)[keep])
        t = np.asarray(teacher, dtype=np.float64)[keep]
        targets.append(t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), _EPS))
    f = np.concatenate(feats)
    t = np.concatenate(targets)
    if len(f) < 4:
        raise ValueError("too few valid rows to fit the projector on")
    order = np.random.default_rng(0).permutation(len(f))
    n_fit = min(len(f) - 1, max(1, int(len(f) * (1.0 - holdout))))
    fit, held = order[:n_fit], order[n_fit:]

    def solve(rows, ridge):
        mean = f[rows].mean(axis=0)
        centred = f[rows] - mean
        gram = centred.T @ centred
        gram[np.diag_indices_from(gram)] += ridge * np.trace(gram) / gram.shape[0]
        kernel = np.linalg.solve(gram, centred.T @ t[rows])
        return kernel, t[rows].mean(axis=0) - mean @ kernel

    def mean_cosine(rows, kernel, bias):
        p = f[rows] @ kernel + bias
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), _EPS)
        return float(np.mean(np.sum(p * t[rows], axis=1)))

    scores = {ridge: mean_cosine(held, *solve(fit, ridge)) for ridge in ridges}
    best = max(scores, key=scores.get)
    kernel, bias = solve(np.arange(len(f)), best)
    model.projector.set_weights([kernel.astype(np.float32), bias.astype(np.float32)])
    return {
        "rows": int(len(f)),
        "ridge": float(best),
        "cosine_holdout": scores[best],
        "cosine_holdout_by_ridge": {float(k): round(v, 4) for k, v in scores.items()},
    }
