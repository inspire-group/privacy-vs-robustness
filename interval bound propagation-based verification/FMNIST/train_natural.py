# coding=utf-8
# Copyright 2018 The Interval Bound Propagation Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains IBP on Fashion-Mnist."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='2'
WIDTH_NUM = 16

from absl import app
from absl import flags
from absl import logging
import interval_bound_propagation as ibp
import tensorflow as tf
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'fmnist', 'Dataset (either "mnist" or "cifar10")')
flags.DEFINE_string('output_dir', './model_natural', 'Output directory.')

#flags.DEFINE_integer('width_num', 4, 'Width Number for CNN')
# Options.
flags.DEFINE_integer('steps', 80001, 'Number of steps in total.')
flags.DEFINE_integer('test_every_n', 5000,
                     'Number of steps between testing iterations.')
flags.DEFINE_integer('warmup_steps', 8000, 'Number of warm-up steps.')
flags.DEFINE_integer('rampup_steps', 30000, 'Number of ramp-up steps.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_float('epsilon', 0.1, 'Target epsilon.')
flags.DEFINE_string('learning_rate', '1e-4,1e-5@50000,1e-6@65000',
                    'Learning rate schedule of the form: '
                    'initial_learning_rate[,learning:steps]*. E.g., "1e-3" or '
                    '"1e-3,1e-4@15000,1e-5@25000".')
flags.DEFINE_float('nominal_xent_init', 1.,
                   'Initial weight for the nominal cross-entropy.')
flags.DEFINE_float('nominal_xent_final', 1.,
                   'Final weight for the nominal cross-entropy.')
flags.DEFINE_float('verified_xent_init', 0.,
                   'Initial weight for the verified cross-entropy.')
flags.DEFINE_float('verified_xent_final', 0.,
                   'Final weight for the verified cross-entropy.')
flags.DEFINE_float('attack_xent_init', 0.,
                   'Initial weight for the attack cross-entropy.')
flags.DEFINE_float('attack_xent_final', 0.,
                   'Initial weight for the attack cross-entropy.')


def show_metrics(step_value, metric_values, loss_value=None):
    print('{}: {}nominal accuracy = {:.2f}%, '
            'verified = {:.2f}%, attack = {:.2f}%'.format(
                step_value,
                'loss = {}, '.format(loss_value) if loss_value is not None else '',
                metric_values.nominal_accuracy * 100.,
                metric_values.verified_accuracy * 100.,
                metric_values.attack_accuracy * 100.))


def layers():
    """Returns the layer specification for a given model name."""
    
    width_num = WIDTH_NUM
    return (
        ('conv2d', (3, 3), 16*width_num, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 16*width_num, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 16*width_num, 'SAME', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 32*width_num, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 32*width_num, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 32*width_num, 'SAME', 2),
        ('activation', 'relu'),
        ('linear', 200),
        ('activation', 'relu'))


def main(unused_args):
    FINAL_OUTPUT_DIR = FLAGS.output_dir
    logging.info('Training IBP on %s...', FLAGS.dataset.upper())
    step = tf.train.get_or_create_global_step()

    # Learning rate.
    tokens = FLAGS.learning_rate.split(',')
    learning_rates = [float(tokens[0])]
    learning_rate_boundaries = []
    for t in tokens[1:]:
        lr, boundary = t.split('@', 1)
        learning_rates.append(float(lr))
        learning_rate_boundaries.append(int(boundary))
    logging.info('Learning rate schedule: %s at %s', str(learning_rates),
                 str(learning_rate_boundaries))
    learning_rate = tf.train.piecewise_constant(step, learning_rate_boundaries,
                                                learning_rates)
    print(learning_rate)

    # Dataset.
    input_bounds = (0., 1.)
    num_classes = 10
    if FLAGS.dataset == 'fmnist':
        data_train, data_test = tf.keras.datasets.fashion_mnist.load_data()
        print(np.amax(data_train[0]), np.amin(data_train[0]))
    else:
        assert FLAGS.dataset == 'cifar10', (
            'Unknown dataset "{}"'.format(FLAGS.dataset))
        data_train, data_test = tf.keras.datasets.cifar10.load_data()
        data_train = (data_train[0], data_train[1].flatten())
        data_test = (data_test[0], data_test[1].flatten())
    data = ibp.build_dataset(data_train, batch_size=FLAGS.batch_size,
                           sequential=False)
    #print('DATASET----', np.amax(data.image), np.amin(data.image))

    # Base predictor network.
    predictor = ibp.DNN(num_classes, layers(), 0.0001)
    predictor = ibp.VerifiableModelWrapper(predictor)

    # Training.
    train_losses, train_loss, _ = ibp.create_classification_losses(
        step,
        data.image,
        data.label,
        predictor,
        FLAGS.epsilon,
        loss_weights={
            'nominal': {'init': FLAGS.nominal_xent_init,
                        'final': FLAGS.nominal_xent_final},
            'attack': {'init': FLAGS.attack_xent_init,
                       'final': FLAGS.attack_xent_final},
            'verified': {'init': FLAGS.verified_xent_init,
                         'final': FLAGS.verified_xent_final},
        },
        warmup_steps=FLAGS.warmup_steps,
        rampup_steps=FLAGS.rampup_steps,
        input_bounds=input_bounds)
    saver = tf.train.Saver(predictor.wrapped_network.get_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(train_loss, step)

    # Test using while loop.
    def get_test_metrics(batch_size, attack_builder=ibp.UntargetedPGDAttack):
        """Returns the test metrics."""
        num_test_batches = len(data_test[0]) // batch_size
        assert len(data_test[0]) % batch_size == 0, (
            'Test data is not a multiple of batch size.')

        def cond(i, *unused_args):
            return i < num_test_batches

        def body(i, metrics):
            """Compute the sum of all metrics."""
            test_data = ibp.build_dataset(data_test, batch_size=batch_size,
                                          sequential=True)
            predictor(test_data.image, is_training=False)
            input_interval_bounds = ibp.IntervalBounds(
                tf.maximum(test_data.image - FLAGS.epsilon, input_bounds[0]),
                tf.minimum(test_data.image + FLAGS.epsilon, input_bounds[1]))
            predictor.propagate_bounds(input_interval_bounds)
            test_specification = ibp.ClassificationSpecification(
                test_data.label, num_classes)
            test_attack = attack_builder(predictor, test_specification, FLAGS.epsilon,
                                         input_bounds=input_bounds,
                                         optimizer_builder=ibp.UnrolledAdam)
            test_losses = ibp.Losses(predictor, test_specification, test_attack)
            test_losses(test_data.label)
            new_metrics = []
            for m, n in zip(metrics, test_losses.scalar_metrics):
                new_metrics.append(m + n)
            return i + 1, new_metrics

        total_count = tf.constant(0, dtype=tf.int32)
        total_metrics = [tf.constant(0, dtype=tf.float32)
                         for _ in range(len(ibp.ScalarMetrics._fields))]
        total_count, total_metrics = tf.while_loop(
            cond,
            body,
            loop_vars=[total_count, total_metrics],
            back_prop=False,
            parallel_iterations=1)
        total_count = tf.cast(total_count, tf.float32)
        test_metrics = []
        for m in total_metrics:
            test_metrics.append(m / total_count)
        return ibp.ScalarMetrics(*test_metrics)

    test_metrics = get_test_metrics(
        FLAGS.batch_size, ibp.UntargetedPGDAttack)
    summaries = []
    for f in test_metrics._fields:
        summaries.append(
            tf.summary.scalar(f, getattr(test_metrics, f)))
    test_summaries = tf.summary.merge(summaries)
    test_writer = tf.summary.FileWriter(os.path.join(FINAL_OUTPUT_DIR, 'test'))

    # Run everything.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.train.SingularMonitoredSession(config=tf_config) as sess:
        for _ in range(FLAGS.steps):
            iteration, loss_value, _ = sess.run(
                [step, train_losses.scalar_losses.nominal_cross_entropy, train_op])
            if iteration % 200 == 0:
                print('step: ', iteration, 'nominal loss_value: ', loss_value)
            if iteration % FLAGS.test_every_n == 0:
                metric_values, summary = sess.run([test_metrics, test_summaries])
                test_writer.add_summary(summary, iteration)
                show_metrics(iteration, metric_values, loss_value=loss_value)
                saver.save(sess._tf_sess(),  # pylint: disable=protected-access
                           os.path.join(FINAL_OUTPUT_DIR, 'checkpoint'),
                           global_step=iteration)


if __name__ == '__main__':
    app.run(main)
