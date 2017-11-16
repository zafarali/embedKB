import tensorflow as tf

def margin_loss(positive_score, negative_score, margin=1):
    """
    Calculates the max-margin ranking loss
    """
    return tf.maximum(-negative_score + positive_score + margin, 0, name='margin_loss')