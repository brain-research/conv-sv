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

def Invert_Operator_NP(filter, inp_shape):
  # computed the inverse of a filter. However the result is much bigger,
  # (n x n) instead of (k x k), therefore, we later project it into (k x k)
  

  # compute the singular values using FFT
  # first compute the transforms for each pair of input and output channels
  transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0, 1])

  # now, for each transform coefficient, compute the singular values of the
  # matrix obtained by selecting that coefficient for
  # input-channel/output-channel pairs
  U, D, V_t = np.linalg.svd(transform_coeff, compute_uv=True,
                            full_matrices=False)
  D_inv = 1.0 / (np.sign(D) * (np.absolute(D) + 1e-6))
  U_t = np.transpose(U, (0, 1, 3, 2))
  V = np.transpose(V_t, (0, 1, 3, 2))
  if filter.shape[3] > filter.shape[2]:
    inv_transform_coeff = np.matmul(V, D_inv[..., None] * U_t)
  else:
    inv_transform_coeff = np.matmul(V * D_inv[..., None, :], U_t)
  inv_filter = np.fft.ifft2(inv_transform_coeff, axes=[0, 1]).real
  return inv_filter
