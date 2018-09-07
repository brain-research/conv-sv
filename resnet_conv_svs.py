"""
    Copyright 2018 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


"""Plot the singular values of the official resnet pretrained imagenet
network (as of 5/1/2018).

Load the pretrained resnet network and print out its singular values.


"""
import tensorflow as tf
import numpy as np
import os
import re
import sys
import conv2d_singular_values as convsv
import matplotlib.pyplot as plt
import matplotlib2tikz
from matplotlib2tikz import save as tikz_save
import absl
from absl import flags
from absl import app



FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir",
                    "/Users/hsedghi/Documents/git/convo_singular_values/resnet_v2_imagenet_checkpoint/",
                    "Official imagenet resnet checkpoint directory on 5/1/2018")

flags.DEFINE_string("graph",
                    "model.ckpt-250200.meta",
                    "Official imagenet resnet graph on 5/1/2018")

flags.DEFINE_string("plot_output_prefix",
                    "/Users/hsedghi/Documents/git/convo_singular_values//tmp/conv_svd_plot",
                    "File where to output the plot.")

flags.DEFINE_boolean("filter_one_by_one",
                     True,
                     "This is true if convolutions with 1x1 filters are not "
                     "included in the output")

flags.DEFINE_integer("xlim",
                     700000,
                     "The right endpoint of the range of values on the x-axis"
                     "of the plot")

MAX_LAYER_NUMBER = 56.0

def get_layer_number(conv_layer_name):
  match = re.search('conv2d_(\d+)',conv_layer_name)
  if match == None:
    return 0
  else:
    return int(match.group(1))


def conv_singular_values():
  """Get singular values."""
  with tf.Graph().as_default() as g:
    saver = tf.train.import_meta_graph(FLAGS.checkpoint_dir + FLAGS.graph,
                                       clear_devices=True)

    with tf.Session().as_default() as sess:
      saver.restore(sess,tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

      svd_list = list()
      for op in g.get_operations():
        if op.type == 'Conv2D':
          print '---'
          print op.name
          if len(op.inputs) == 2:
            feature_map, kernel = op.inputs
            print feature_map.shape.as_list()
            batch_size, num_channels, height, width = feature_map.shape.as_list()
            print kernel.shape.as_list()
            kernel_size_height, kernel_size_width, input_channels, output_channels = kernel.shape.as_list()
            if ((FLAGS.filter_one_by_one == False) or
                (kernel_size_height > 1) or
                (kernel_size_width > 1)):
              kernel_np = sess.run(kernel)
              tf_np = convsv.SVD_Conv_Tensor_NP(kernel_np, [height, width])
              this_layers_svds = np.flip(np.sort(tf_np.flatten()),0)
              svd_list.append([op.name,this_layers_svds])
          sys.stdout.flush()
  n = float(len(svd_list))
  for i in range(len(svd_list)):
    name, svds = svd_list[i]
    normalized_layer_number = (0.1 + get_layer_number(name))/(0.2 + MAX_LAYER_NUMBER)
    this_color = (1 - normalized_layer_number, normalized_layer_number, 0.1)
    short_name = name.replace('resnet_model/','')
    short_name = short_name.replace('/Conv2D','')
    plt.plot(range(len(svds)), svds, label = short_name, color = this_color)
  axes = plt.gca()
  plt.legend(fontsize='xx-small', ncol=3)
  axes.set_xlim([0,FLAGS.xlim])
  plt.xlabel('Singular value rank',fontsize=20)
  plt.ylabel('Singular value',fontsize=20)

  png_output_name = FLAGS.plot_output_prefix + ".png"
  plot_directory = os.path.dirname(png_output_name)
  if not os.path.isdir(plot_directory):
    os.mkdir(plot_directory)
  f = open(png_output_name, 'w')
  plt.savefig(f, dpi=256)
  tikz_save(FLAGS.plot_output_prefix + '.tex')

def main(argv):
  del argv
  conv_singular_values()

if __name__ == "__main__":
  app.run(main)