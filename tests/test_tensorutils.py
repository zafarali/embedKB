import tensorflow as tf
from embedKB.utils import tensorutils
import numpy as np

entity_dims = 18
relationship_dims = 15

# define the two vectors
left = tf.placeholder(tf.float32, shape=(None, entity_dims, 1))
right = tf.placeholder(tf.float32, shape=(None, entity_dims, 1))

# to test out the product, we create a feed dict.
# with one batch element.
single_batch_feed_dict = {left: np.linspace(0, 1, entity_dims).reshape(-1, entity_dims, 1),
                          right: np.linspace(0, 1, entity_dims).reshape(-1, entity_dims, 1),}

multi_batch_feed_dict = {left: np.linspace(0, 1, entity_dims*2).reshape(2, entity_dims, 1),
                         right: np.linspace(0, 1, entity_dims*2).reshape(2, entity_dims, 1),}

sess = tf.InteractiveSession()


def test_bilinear_product_with_matrix():
    # define the relationships
    matrix_relationship = tf.placeholder(tf.float32, shape=(None, entity_dims, entity_dims))

    # matrix result
    matrix_result = tensorutils.bilinear_product(left, matrix_relationship, right)

    # assert that the result must be a scalar.
    assert tensorutils.int_shapes(matrix_result) == [-1, 1]

    feed_dict_local = dict(single_batch_feed_dict)
    feed_dict_local[matrix_relationship] = np.linspace(0, 1, entity_dims*entity_dims).reshape(-1, entity_dims, entity_dims)

    tf_result = sess.run(matrix_result, feed_dict_local)
    
    # without batches
    numpy_result = np.matmul(feed_dict_local[left].squeeze(0).T, np.matmul(feed_dict_local[matrix_relationship].squeeze(0), feed_dict_local[right].squeeze(0)))
    assert np.allclose(numpy_result, tf_result)

    feed_dict_local = dict(multi_batch_feed_dict)
    feed_dict_local[matrix_relationship] = np.linspace(0, 1, entity_dims*entity_dims*2).reshape(2, entity_dims, entity_dims)
    tf_result = sess.run(matrix_result, feed_dict_local)
    # with batches
    numpy_result = np.matmul(np.transpose(feed_dict_local[left], (0, 2, 1)), np.matmul(feed_dict_local[matrix_relationship], feed_dict_local[right]))
    assert np.allclose(numpy_result.squeeze(), tf_result.squeeze())

# manual implementation of the bilinear product from definitions:
def manual_bilinear_products_for_tensors(a, M, b):
    assert len(M.shape) == 4
    result = np.zeros((a.shape[0], M.shape[-1]))
    for i in range(a.shape[0]): # go over all batches
        for m in range(M.shape[-1]): # go over the last dim
            result[i][m] = np.matmul(a[i].T, np.matmul(M[i, :, :, m], b[i]))

    return result

def test_bilinear_product_with_tensor():
    # define the relationships
    tensor_relationship = tf.placeholder(tf.float32, shape=(None, entity_dims, entity_dims, relationship_dims))

    # define the tensor result
    tensor_result = tensorutils.bilinear_product(left, tensor_relationship, right)

    # assert that the result has to be a vector
    assert tensorutils.int_shapes(tensor_result) == [-1, relationship_dims]

    feed_dict_local = dict(single_batch_feed_dict)
    feed_dict_local[tensor_relationship] = np.linspace(0, 1, entity_dims*entity_dims*relationship_dims).reshape(-1, entity_dims, entity_dims, relationship_dims)

    tf_result = sess.run(tensor_result, feed_dict_local)
    # without batches
    numpy_result = manual_bilinear_products_for_tensors(feed_dict_local[left], feed_dict_local[tensor_relationship], feed_dict_local[right])

    assert np.allclose(numpy_result, tf_result)

    feed_dict_local = dict(multi_batch_feed_dict)
    feed_dict_local[tensor_relationship] = np.linspace(0, 1, 2*entity_dims*entity_dims*relationship_dims).reshape(2, entity_dims, entity_dims, relationship_dims)

    tf_result = sess.run(tensor_result, feed_dict_local)
    # with batches
    numpy_result = manual_bilinear_products_for_tensors(feed_dict_local[left], feed_dict_local[tensor_relationship], feed_dict_local[right])

    assert np.allclose(numpy_result, tf_result)