from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class EpisodeMemory(object):

    def __init__(self, template, capacity, max_length, scope):
        self._capacity = capacity
        self._max_length = max_length
        with tf.variable_scope(scope) as scope:
            self._scope = scope
            self._length = tf.Variable(tf.zeros(capacity, tf.int32), False)
            self._buffers = [
                tf.Variable(tf.zeros([capacity, max_length] + elem.shape.as_list(), elem.dtype), False)
                for elem in template
            ]

    def length(self, rows=None):
        rows = tf.range(self._capacity) if rows is None else rows
        return tf.gather(self._length, rows)

    def append(self, transitions, rows=None):
        rows = tf.range(self._capacity) if rows is None else rows
        assert rows.shape.ndims == 1
        assert_capacity = tf.assert_less(rows, self._capacity, message='capacity exceeded')
        with tf.control_dependencies([assert_capacity]):
            assert_max_length = tf.assert_less(tf.gather(self._length, rows),
                                               self._max_length,
                                               message='max length exceeded')
        append_ops = []
        with tf.control_dependencies([assert_max_length]):
            for buffer_, elements in zip(self._buffers, transitions):
                timestep = tf.gather(self._length, rows)
                indices = tf.stack([rows, timestep], 1) 
                append_ops.append(tf.scatter_nd_update(buffer_, indices, elements))
        with tf.control_dependencies(append_ops):
            episode_mask = tf.reduce_sum(tf.one_hot(rows, self._capacity, dtype=tf.int32), 0)
            return self._length.assign_add(episode_mask)

    def replace(self, episodes, length, rows=None):
        rows = tf.range(self._capacity) if rows is None else rows
        assert rows.shape.ndims == 1
        assert_capacity = tf.assert_less(rows, self._capacity, message='capacity exceeded')
        with tf.control_dependencies([assert_capacity]):
            assert_max_length = tf.assert_less_equal(length,
                                                     self._max_length,
                                                     message='max length exceeded')
        replace_ops = []
        with tf.control_dependencies([assert_max_length]):
            for buffer_, elements in zip(self._buffers, episodes):
                replace_op = tf.scatter_update(buffer_, rows, elements)
                replace_ops.append(replace_op)
        with tf.control_dependencies(replace_ops):
            return tf.scatter_update(self._length, rows, length)

    def data(self, rows=None):
        rows = tf.range(self._capacity) if rows is None else rows
        assert rows.shape.ndims == 1
        episode = [tf.gather(buffer_, rows) for buffer_ in self._buffers]
        length = tf.gather(self._length, rows)
        return episode, length

    def clear(self, rows=None):
        rows = tf.range(self._capacity) if rows is None else rows
        assert rows.shape.ndims == 1
        return tf.scatter_update(self._length, rows, tf.zeros_like(rows))