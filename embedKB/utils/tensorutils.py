import tensorflow as tf
import logging
def bilinear_product(left, middle, right):
    logging.warn('Note bilinear product is not implemented yet.')
    return tf.constant(0.0)

def create_placeholder(name):
    return tf.placeholder(tf.int32, shape=(None, 1), name=name)
