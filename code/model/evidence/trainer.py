"""Teacher-forced evidence-training primitives, separate from REINFORCE.

The full TensorFlow training runner and checkpoint loop are not yet connected.
These functions prepare supervised targets for the existing MINERVA action
policy without changing its shared trainable weights.
"""
from typing import Sequence

import numpy as np

from code.model.evidence.supervision import lookup_gold_action, validate_gold_path


def teacher_forced_targets(
    start_entity: int,
    gold_path: Sequence[Sequence[int]],
    action_entities: Sequence[Sequence[int]],
    action_relations: Sequence[Sequence[int]],
    *,
    stop_relation: int = None,
    pad_action: int = 0,
) -> tuple:
    """Return action indices and a loss mask for one gold trajectory.

    action_entities[t]/action_relations[t] must be the candidate actions at the
    TEACHER-FORCED entity at step t, not at a sampled model position.
    When stop_relation is provided, append a STOP target after the gold path.
    """
    path = validate_gold_path(start_entity, gold_path)
    required_steps = len(path) + int(stop_relation is not None)
    if len(action_entities) != len(action_relations):
        raise ValueError("Action sequences must have the same number of steps")
    if required_steps > len(action_entities):
        raise ValueError("Evidence path exceeds the navigation budget")
    current = int(start_entity)
    targets = []
    for t, (head, rel, tail) in enumerate(path):
        targets.append(lookup_gold_action(current, (head, rel, tail), action_entities[t], action_relations[t]))
        current = tail
    if stop_relation is not None:
        t = len(path)
        targets.append(lookup_gold_action(current, (current, stop_relation, current), action_entities[t], action_relations[t]))
    loss_mask = [1.0] * len(targets)
    while len(targets) < len(action_entities):
        targets.append(pad_action)
        loss_mask.append(0.0)
    return np.asarray(targets, dtype=np.int32), np.asarray(loss_mask, dtype=np.float32)


def masked_supervised_loss(log_probabilities, target_actions, loss_mask):
    """TensorFlow loss for teacher-forced actions, including variable path length.

    log_probabilities: [batch, steps, max_actions]
    target_actions: [batch, steps]
    loss_mask: [batch, steps]
    Avoid importing TensorFlow until this routine is invoked.
    """
    import tensorflow as tf
    log_probabilities = tf.convert_to_tensor(log_probabilities)
    target_actions = tf.cast(target_actions, tf.int32)
    loss_mask = tf.cast(loss_mask, log_probabilities.dtype)
    selected = tf.gather(log_probabilities, target_actions, axis=2, batch_dims=2)
    per_example = -tf.reduce_sum(selected * loss_mask, axis=1)
    lengths = tf.maximum(tf.reduce_sum(loss_mask, axis=1), 1.0)
    return tf.reduce_mean(per_example / lengths)
