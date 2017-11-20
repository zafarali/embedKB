import tensorflow as tf
from embedKB.utils import tensorutils


entity_dims = 18
relationship_dims = 15

# define the two vectors
left = tf.placeholder(tf.float32, shape=(None, entity_dims, 1))
right = tf.placeholder(tf.float32, shape=(None, entity_dims, 1))


def test_bilinear_product_with_matrix():
    # define the relationships
    matrix_relationship = tf.placeholder(tf.float32, shape=(None, entity_dims, entity_dims))

    # matrix result
    matrix_result = tensorutils.bilinear_product(left, matrix_relationship, right)

    # assert that the result must be a scalar.
    assert tensorutils.int_shapes(matrix_result) == [-1, 1]

def test_bilinear_product_with_tensor():
    # define the relationships
    tensor_relationship = tf.placeholder(tf.float32, shape=(None, entity_dims, entity_dims, relationship_dims))

    # define the tensor result
    tensor_result = tensorutils.bilinear_product(left, tensor_relationship, right)

    # assert that the result has to be a vector
    assert tensorutils.int_shapes(tensor_result) == [-1, relationship_dims]
