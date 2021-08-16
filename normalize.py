from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class StreamingNormalize(object):

    def __init__(self, template, center=True, scale=True, clip=10, name='normalize'):
        self._center = center
        self._scale = scale
        self._clip = clip
        self._name = name
        with tf.name_scope(name):
            self._count = tf.Variable(0, False)
            self._mean = tf.Variable(tf.zeros_like(template), False)
            self._var_sum = tf.Variable(tf.zeros_like(template), False)
            
    def transform(self, value):
        with tf.name_scope(self._name + '/transform'):
            no_batch_dim = value.shape.ndims == self._mean.shape.ndims
            if no_batch_dim:
                value = value[None, ...]
            if self._center:
                value -= self._mean[None, ...]
            if self._scale:
                value /= tf.cond(
                    self._count > 1, lambda: self._std() + 1e-8, lambda: tf.ones_like(self._var_sum))[None]
            if self._clip:
                value = tf.clip_by_value(value, -self._clip, self._clip)
            if no_batch_dim:
                value = value[0]
            return tf.check_numerics(value, 'value')
            
    def update(self, value):
        with tf.name_scope(self._name + '/update'):
            if value.shape.ndims == self._mean.shape.ndims:
                value = value[None, ...]
            count = tf.shape(value)[0]
            with tf.control_dependencies([self._count.assign_add(count)]):
                step = tf.cast(self._count, tf.float32)
                mean_delta = tf.reduce_sum(value - self._mean[None, ...], 0)
                new_mean = self._mean + mean_delta / step
                new_mean = tf.cond(self._count > 1, lambda: new_mean, lambda: value[0])
                var_delta = (value - self._mean[None, ...]) * (value - new_mean[None, ...])
                new_var_sum = self._var_sum + tf.reduce_sum(var_delta, 0)
            with tf.control_dependencies([new_mean, new_var_sum]):
                update = self._mean.assign(new_mean), self._var_sum.assign(new_var_sum)
            with tf.control_dependencies(update):
                if value.shape.ndims == 1:
                    value = tf.reduce_mean(value)
                return self._summary('value', tf.reduce_mean(value))
                
    def reset(self):
        with tf.name_scope(self._name + '/reset'):
            return tf.group(self._count.assign(0), self._mean.assign(tf.zeros_like(self._mean)),
                            self._var_sum.assign(tf.zeros_like(self._var_sum)))

    def summary(self):
        with tf.name_scope(self._name + '/summary'):
            mean_summary = tf.cond(self._count > 0, lambda: self._summary('mean', self._mean), str)
            std_summary = tf.cond(self._count > 1, lambda: self._summary('stddev', self._std()), str)
            return tf.summary.merge([mean_summary, std_summary])

    def _std(self):
        variance = tf.cond(self._count > 1, lambda: self._var_sum / tf.cast(
            self._count - 1, tf.float32), lambda: tf.ones_like(self._var_sum) * float('nan'))
        return tf.sqrt(variance + 1e-4)

    def _summary(self, name, tensor):
        if tensor.shape.ndims == 0:
            return tf.summary.scalar(name, tensor)
        else:
            return tf.summary.histogram(name, tensor)
