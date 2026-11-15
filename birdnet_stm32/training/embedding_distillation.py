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

Optionally a second training-only head, linear on the same embedding, predicts
where and when the recording was made (``birdnet_stm32.data.aux_targets``): the
location as a unit vector and the day of year as (sin, cos), each scored by a
cosine distance over the rows where it is known. It asks the embedding to carry
the species composition of a place and season. Like the projector, it never
reaches the checkpoint.
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
        output_teacher: Optional frozen larger model to distil from online, on the
            same batch (``distillation.OnlineDistilledModel``'s Bernoulli KL term).
        output_weight: Weight of that KL term.
        aux_weight: Weight of the location/date head's loss (the sum of its two
            cosine distances); 0 for none. With it, ``train_step`` expects labels
            ``(y, teacher_embedding, valid, aux, aux_valid)``.
    """

    def __init__(
        self,
        student: tf.keras.Model,
        teacher_dim: int,
        weight: float,
        output_teacher: tf.keras.Model | None = None,
        output_weight: float = 0.0,
        aux_weight: float = 0.0,
    ):
        super().__init__(inputs=student.inputs, outputs=student.outputs, name=student.name)
        if weight <= 0:
            raise ValueError(f"embedding distillation weight must be positive, got {weight}")
        self.output_teacher = None
        if output_teacher is not None:
            from birdnet_stm32.training.distillation import check_distillation_teacher

            check_distillation_teacher(student, output_teacher)
            if output_weight <= 0:
                raise ValueError(f"online distillation weight must be positive, got {output_weight}")
            output_teacher.trainable = False
            self.output_teacher = output_teacher
            self.distill_weight = float(output_weight)
            self.distill_metric = tf.keras.metrics.Mean(name="distill_kl")
        features = student.get_layer(EMBEDDING_LAYER).output
        self.embedder = tf.keras.Model(student.inputs, [student.outputs[0], features], name="student_with_embedding")
        self.projector = tf.keras.layers.Dense(int(teacher_dim), name="teacher_projector", dtype="float32")
        self.projector.build((None, int(features.shape[-1])))
        self.embedding_weight = float(weight)
        self.cosine_metric = tf.keras.metrics.Mean(name="teacher_cosine")
        if aux_weight < 0:
            raise ValueError(f"aux weight must be >= 0, got {aux_weight}")
        self.aux_weight = float(aux_weight)
        self.aux_head = None
        if self.aux_weight > 0:
            from birdnet_stm32.data.aux_targets import AUX_DIM

            self.aux_head = tf.keras.layers.Dense(AUX_DIM, name="aux_geo_date_head", dtype="float32")
            self.aux_head.build((None, int(features.shape[-1])))
            self.geo_metric = tf.keras.metrics.Mean(name="aux_geo_cosine")
            self.date_metric = tf.keras.metrics.Mean(name="aux_date_cosine")

    def train_step(self, data):
        """The default Keras step, plus the embedding term (and the location/date term)."""
        x, labels = data
        y, teacher, valid = labels[:3]
        with tf.GradientTape() as tape:
            y_pred, features = self.embedder(x, training=True)
            supervised = self.compute_loss(x=x, y=y, y_pred=y_pred, training=True)
            distance, cosine = masked_cosine_distance(self.projector(features), teacher, valid)
            loss = supervised + self.embedding_weight * distance
            if self.aux_head is not None:
                aux_t, aux_v = labels[3], labels[4]
                pred = self.aux_head(tf.cast(features, tf.float32))
                geo_d, geo_cos = masked_cosine_distance(pred[:, :3], aux_t[:, :3], aux_v[:, 0])
                date_d, date_cos = masked_cosine_distance(pred[:, 3:], aux_t[:, 3:], aux_v[:, 1])
                loss = loss + self.aux_weight * (geo_d + date_d)
            if self.output_teacher is not None:
                from birdnet_stm32.training.distillation import bernoulli_kl

                divergence = bernoulli_kl(tf.stop_gradient(self.output_teacher(x, training=False)), y_pred)
                self.distill_metric.update_state(divergence)
                loss = loss + self.distill_weight * divergence
            self._loss_tracker.update_state(loss, sample_weight=tf.shape(y)[0])
            scaled = self.optimizer.scale_loss(loss)
        variables = self.trainable_weights
        self.optimizer.apply_gradients(zip(tape.gradient(scaled, variables), variables, strict=True))
        self.cosine_metric.update_state(cosine, sample_weight=valid)
        if self.aux_head is not None:
            self.geo_metric.update_state(geo_cos, sample_weight=aux_v[:, 0])
            self.date_metric.update_state(date_cos, sample_weight=aux_v[:, 1])
        return self.compute_metrics(x, y, y_pred)

    def test_step(self, data):
        """The default step; validation carries no teacher embeddings, so no cosine (and no distillation)."""
        logs = super().test_step(data)
        logs.pop(self.cosine_metric.name, None)
        if self.output_teacher is not None:
            logs.pop(self.distill_metric.name, None)
        if self.aux_head is not None:
            logs.pop(self.geo_metric.name, None)
            logs.pop(self.date_metric.name, None)
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


def fit_aux_head(model: EmbeddingDistilledModel, batches, ridges: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)) -> dict:
    """Initialize the location/date head by ridge regression on the student's current embeddings.

    As with the projector: from a trained student, a random head would open at the largest loss and push
    its gradients into a converged backbone. The location columns are fitted on rows with a known location,
    the date columns on rows with a known date, each with the ridge strength that gives the best held-out
    mean cosine (20% of rows held out, then refitted on all).

    Args:
        model: The wrapper whose ``aux_head`` is set.
        batches: Iterable of ``(x, aux, aux_valid)`` arrays.

    Returns:
        Row counts, chosen ridge strengths and held-out mean cosines.
    """
    feats, targets, valids = [], [], []
    for x, aux_t, aux_v in batches:
        _, features = model.embedder(x, training=False)
        feats.append(np.asarray(features, np.float64))
        targets.append(np.asarray(aux_t, np.float64))
        valids.append(np.asarray(aux_v) > 0.5)
    f, t, v = np.concatenate(feats), np.concatenate(targets), np.concatenate(valids)
    kernel = np.zeros((f.shape[1], t.shape[1]))
    bias = np.zeros(t.shape[1])
    report = {}
    for name, cols, flag in (("geo", slice(0, 3), 0), ("date", slice(3, 5), 1)):
        rows = np.flatnonzero(v[:, flag])
        if len(rows) < 8:
            report[name] = {"rows": int(len(rows)), "skipped": True}
            continue
        order = np.random.default_rng(0).permutation(rows)
        n_fit = int(len(order) * 0.8)
        fit_rows, held = order[:n_fit], order[n_fit:]

        def solve(r, ridge, cols=cols):
            mean = f[r].mean(axis=0)
            c = f[r] - mean
            gram = c.T @ c
            gram[np.diag_indices_from(gram)] += ridge * np.trace(gram) / gram.shape[0]
            k = np.linalg.solve(gram, c.T @ t[r, cols])
            return k, t[r, cols].mean(axis=0) - mean @ k

        def cos(r, k, b, cols=cols):
            p = f[r] @ k + b
            p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), _EPS)
            q = t[r, cols] / np.maximum(np.linalg.norm(t[r, cols], axis=1, keepdims=True), _EPS)
            return float(np.mean(np.sum(p * q, axis=1)))

        scores = {r: cos(held, *solve(fit_rows, r)) for r in ridges}
        best = max(scores, key=scores.get)
        kernel[:, cols], bias[cols] = solve(rows, best)
        report[name] = {"rows": int(len(rows)), "ridge": float(best), "cosine_holdout": round(scores[best], 4)}
    model.aux_head.set_weights([kernel.astype(np.float32), bias.astype(np.float32)])
    return report
