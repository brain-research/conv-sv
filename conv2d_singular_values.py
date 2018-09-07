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


import tensorflow as tf
import numpy as np

def SVD_Conv_Tensor(conv, inp_shape):
  """ Find the singular values of the linear transformation
  corresponding to the convolution represented by conv on
  an n x n x depth input. """
  conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
  conv_shape = conv.get_shape().as_list()
  padding = tf.constant([[0, 0], [0, 0],
                         [0, inp_shape[0] - conv_shape[0]],
                         [0, inp_shape[1] - conv_shape[1]]])
  transform_coeff = tf.fft2d(tf.pad(conv_tr, padding))
  singular_values = tf.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]),
                           compute_uv=False)
  return singular_values

def Clip_OperatorNorm(conv, inp_shape, clip_to):
  conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
  conv_shape = conv.get_shape().as_list()
  padding = tf.constant([[0, 0], [0, 0],
                         [0, inp_shape[0] - conv_shape[0]],
                         [0, inp_shape[1] - conv_shape[1]]])
  transform_coeff = tf.fft2d(tf.pad(conv_tr, padding))
  D, U, V = tf.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]))
  norm = tf.reduce_max(D)
  D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex64)
  clipped_coeff = tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped),
                                         V, adjoint_b=True))
  clipped_conv_padded = tf.real(tf.ifft2d(
      tf.transpose(clipped_coeff, perm=[2, 3, 0, 1])))
  return tf.slice(tf.transpose(clipped_conv_padded, perm=[2, 3, 0, 1]),
                  [0] * len(conv_shape), conv_shape), norm

def SVD_Conv_Tensor_NP(filter, inp_size):
  # compute the singular values using FFT
  # first compute the transforms for each pair of input and output channels
  transform_coeff = np.fft.fft2(filter, inp_size, axes=[0, 1])

  # now, for each transform coefficient, compute the singular values of the
  # matrix obtained by selecting that coefficient for
  # input-channel/output-channel pairs
  return np.linalg.svd(transform_coeff, compute_uv=False)

def Clip_OperatorNorm_NP(filter, inp_shape, clip_to):
  # compute the singular values using FFT
  # first compute the transforms for each pair of input and output channels
  transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0, 1])

  # now, for each transform coefficient, compute the singular values of the
  # matrix obtained by selecting that coefficient for
  # input-channel/output-channel pairs
  U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
  D_clipped = np.minimum(D, clip_to)
  if filter.shape[2] > filter.shape[3]:
    clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
  else:
    clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
  clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
  args = [range(d) for d in filter.shape]
  return clipped_filter[np.ix_(*args)]


