"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../')
from utils import *
import data_loader

from model import Model

os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1' 
WIDTH_NUM = 2

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']

# Setting up the data and the model
train_data, train_label, test_data, test_label = YALE_split('../../datasets/yale/YALEBXF.mat')
print(train_data.shape, test_data.shape, np.amax(train_data), np.amin(test_data))
YALE_TRAIN = data_loader.DataSubset(train_data,train_label)
YALE_TEST = data_loader.DataSubset(test_data,test_label)

global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(width_num=WIDTH_NUM)

# Setting up the optimizer
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.xent + weight_decay * model.weight_decay_loss
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,
                                                   global_step=global_step)


# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']+'_natural'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=None)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

eps_step = np.linspace(0,config['epsilon'],config['num_schedule_steps'])

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):

    x_batch, y_batch = YALE_TRAIN.get_next_batch(batch_size)

    # Compute Adversarial Perturbations
    start = timer()

    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}


    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))

      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=nat_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))
      ###################################print test accuracy##############
      test_acc = sess.run(model.accuracy, feed_dict={model.x_input:test_data,model.y_input:test_label})
      print('current test accuracy: ', test_acc)

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start
