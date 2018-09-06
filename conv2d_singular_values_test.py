import math
import numpy as np
import tensorflow as tf
import time

from tensorflow.python.platform import test
import conv2d_singular_values as convsv

class CirculantSingularTest(test.TestCase):

  def testOneDimension(self):
    """Test singular values of the one dimensional convolution."""

    filter = np.array([2, 7, -1])
    n = 20

    # Compute the singular values directly
    transform_mat = np.zeros([n, n])
    for i in range(n):
      for j in range(filter.size):
        transform_mat[i, (j+i) % n] = filter[j]
    U, D, V = np.linalg.svd(transform_mat)

    # Compute the singular values using FFT
    D1 = np.sort(np.absolute(np.fft.fft(filter, n)))[::-1]
    self.assertAllCloseAccordingToType(D, D1)

  def twoDimOneChannel(self, filter, n):
    """Given a two-d, one-channel filter and an input size, return the matrix
    of the linear transformation corresponding to the filter"""
    transform_mat = np.zeros([n ** 2, n ** 2])
    for i1 in range(n):
      for i2 in range(n):
        for j1 in range(filter.shape[0]):
          for j2 in range(filter.shape[1]):
            col = ((i1 + j1) % n) * n + (i2 + j2) % n
            transform_mat[i1 * n + i2, col] = filter[j1, j2]
    return transform_mat

  def testTwoDimension(self):
    """Test singular values of the two dimensional convolution."""
    filter = np.array([[2, 7, -1], [1, 3, -8], [5, -3, 1]])
    n = 20
    # Compute the singular values directly
    transform_mat = self.twoDimOneChannel(filter, n)
    D = np.linalg.svd(transform_mat, compute_uv=False)

    # Compute the singular values using FFT
    D1 = np.sort(np.absolute(
        np.fft.fft2(filter, [n, n]).flatten()))[::-1]
    self.assertAllCloseAccordingToType(D, D1)

  def testTwoDimensionRectangular(self):
    """Test singular values of the two dimensional convolution."""
    filter = np.array([[2, 7, -1], [1, 3, -8]])
    n = 20
    # Compute the singular values directly
    transform_mat = self.twoDimOneChannel(filter, n)
    D = np.linalg.svd(transform_mat, compute_uv=False)

    # Compute the singular values using FFT
    D1 = np.sort(np.absolute(
        np.fft.fft2(filter, [n, n]).flatten()))[::-1]
    self.assertAllCloseAccordingToType(D, D1)

  def testMultiChannel(self):
    """Test the case where inputs and outputs have several channels each"""

    num_inp_channels = 2
    num_out_channels = 3
    filter_x = 3
    filter_y = 4
    filter_shape = (filter_x, filter_y, num_inp_channels, num_out_channels)
    filter = np.random.randint(low=-8, high=8,size=filter_shape)
    n = 32
    # Compute the singular values directly
    print("Start Full Matrix")
    start = time.time()
    transform_mat = np.zeros([num_inp_channels * (n ** 2),
                              num_out_channels * (n ** 2)])
    for c1 in range(num_inp_channels):
      for c2 in range(num_out_channels):
        first_row = n * n * c1
        first_col = n * n * c2
        this_block = self.twoDimOneChannel(filter[:,:,c1,c2], n)
        transform_mat[first_row:(first_row+n*n),
                      first_col:(first_col+n*n)] = this_block

    D = np.linalg.svd(transform_mat, compute_uv=False)
    print("Time for SVD Full Matrix:", time.time() - start)

    start = time.time()
    singular_vals_by_freq_pair = convsv.SVD_Conv_Tensor_NP(filter, [n, n])
    print("Short algorithm time:", time.time() - start)

    # sort singular values in decreasing order
    D1 = np.flip(np.sort(singular_vals_by_freq_pair.flatten()),0)
    self.assertAllCloseAccordingToType(D, D1)

  def testMultiChannelTF(self):
    """Test the case where inputs and outputs have several channels each.
    Using much bigger input, to check timing."""

    num_inp_channels = 64
    num_out_channels = 256
    filter_x = 3
    filter_y = 4
    filter_shape = (filter_x, filter_y, num_inp_channels, num_out_channels)
    filter = np.random.randint(low=-8, high=8,size=filter_shape)
    n = 32

    start = time.time()
    singular_vals_by_freq_pair = convsv.SVD_Conv_Tensor_NP(filter, [n, n])

    print("NP SVD time:", time.time() - start)
    # sort singular values in decreasing order
    D1 = np.flip(np.sort(singular_vals_by_freq_pair.flatten()),0)

    with self.test_session() as sess:
      filter_tf = tf.constant(filter, dtype=tf.float32)
      D2_tf = convsv.SVD_Conv_Tensor(filter_tf, [n, n])

      tf.global_variables_initializer().run()
      start = time.time()
      singular_vals = sess.run(D2_tf)
      print("TF SVD Time:", time.time() - start)
      D2 = np.flip(np.sort(singular_vals.flatten()),0)
      self.assertAllClose(D1 / D1[0], D2 / D2[0], atol=3e-5)


  def testMultiChannelClipRepeated(self):
    print("Testing Repeated Clipping")
    num_inp_channels = 3
    num_out_channels = 4
    filter_x = 3
    filter_y = 4
    filter_shape = (filter_x, filter_y, num_inp_channels, num_out_channels)
    filter = np.random.randint(low=-8, high=8,size=filter_shape)
    n = 32
    singular_vals = convsv.SVD_Conv_Tensor_NP(filter, [n, n])
    clipped_filter = convsv.Clip_OperatorNorm_NP(filter, [n, n],
                                               singular_vals.max())
    self.assertAllClose(filter, clipped_filter)

    clip_value = 10
    last_max = singular_vals.max()
    for round in range(10):
      clipped_filter = convsv.Clip_OperatorNorm_NP(clipped_filter, [n, n],
                                                 clip_value)
      clipped_singular_vals = convsv.SVD_Conv_Tensor_NP(clipped_filter, [n, n])
      self.assertTrue(last_max > clipped_singular_vals.max())
      last_max = clipped_singular_vals.max()

  def testMultiChannelClipTF(self):
    print("Testing Clipping TF vs Numpy")
    num_inp_channels = 32
    num_out_channels = 64
    filter_x = 3
    filter_y = 4
    filter_shape = (filter_x, filter_y, num_inp_channels, num_out_channels)
    filter = np.random.randint(low=-8, high=8,size=filter_shape)
    n = 32
    start = time.time()
    clipped_filter = convsv.Clip_OperatorNorm_NP(filter, [n, n], 10)
    print("Numpy Clipping Time:", time.time() - start)

    with self.test_session() as sess:
      filter_tf = tf.constant(filter, dtype=tf.float32)
      clipped_filter_tf, norm = convsv.Clip_OperatorNorm(filter_tf, [n, n], 10)

      tf.global_variables_initializer().run()
      start = time.time()
      clipped_filter2 = sess.run(clipped_filter_tf)
      print("TF Clipping Time:", time.time() - start)
      self.assertAllClose(clipped_filter2, clipped_filter, atol=3e-5)


  def testInverse(self):
    num_inp_channels = 2
    num_out_channels = 3
    filter_x = 3
    filter_y = 4
    filter_shape = (filter_x, filter_y, num_inp_channels, num_out_channels)
    filter = np.random.randint(low=-8, high=8, size=filter_shape)
    n = 16
    inverse_filter = convsv.Invert_Operator_NP(filter, [n, n])
    inverse_inverse_filter = convsv.Invert_Operator_NP(inverse_filter, [n, n])
    #print filter
    #print inverse_filter
    #print inverse_inverse_filter
    args = [range(d) for d in filter_shape]
    proj_filter = inverse_inverse_filter[np.ix_(*args)]
    self.assertAllClose(filter, proj_filter, atol=3e-4)

  """
  def testStrides(self):
    filter = np.array([2, 7, -1, 16])
    n = 30
    stride = 2
    # Compute the singular values directly
    transform_mat = np.zeros([n, n])
    for i in range(n):
      for j in range(filter.size):
        transform_mat[i, (j+i*stride) % n] = filter[j]
    U, D, V = np.linalg.svd(transform_mat)
    print D
    print np.linalg.eig(transform_mat)

    transform_mat = np.zeros([n/stride, n])
    for i in range(n/stride):
      for j in range(filter.size):
        transform_mat[i, (j+i*stride) % n] = filter[j]
    U, D, V = np.linalg.svd(transform_mat)
    print D * math.sqrt(stride)
  """

if __name__ == '__main__':
  test.main()
