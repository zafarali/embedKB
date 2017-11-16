import tensorflow as tf
import logging

"""
a simple oneliner that prints
a tensor, shape, and a message
for example my_tensor = tfPrint('this is my_tensor:', my_tensor)
you must reassign it to your original tensor so that it is 
incorporated in your graph.
"""
tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)
