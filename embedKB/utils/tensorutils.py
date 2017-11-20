import tensorflow as tf
import logging

def int_shapes(tensor):
    return [dim.value if dim.value else -1 for dim in tensor.shape]

def bilinear_product(left, middle, right):
    """
    Implements the bilinear product in tensorflow.
    bilinear_product(x, A, y) = x^T A y

    See answer:
    https://stackoverflow.com/a/34113467
    Special thanks to Guillaume for explaining to me what's going on.

    :param left: a Tensor with dims [-1, n, 1]
    :param middle: a 
    :param right: a Tensor with dims [-1, n, 1] 
    """
    dims = int_shapes(middle)
    with tf.name_scope('bilinear_product'):
        assert len(dims) >= 3, 'Middle matrix/tensor is underdefined with dims:{}'.format(dims)

        if len(dims) == 3:
            # the middle object is a matrix:
            batch_size, dim1, dim2 = dims
            _ = tf.matmul(left, middle, transpose_a=True)
            result = tf.matmul(_, right)
            result = tf.squeeze(result, axis=[-1])

        elif len(dims) == 4:
            # the middle object is a tensor
            batch_size, dim1, dim2, dim3 = dims
            _ = tf.matmul(left, tf.reshape(middle, [-1, dim1, dim2*dim3]), transpose_a=True)
            result = tf.matmul(right ,tf.reshape(_, [-1, dim1, dim3]), transpose_a=True)
            result = tf.squeeze(result, axis=[-2])

        else:
            logging.error('Matrix/Tensor dim wasnot correct.')
    return result

def create_placeholder(name):
    return tf.placeholder(tf.int32, shape=(None, 1), name=name)
