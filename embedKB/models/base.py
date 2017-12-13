import tensorflow as tf
from embedKB.utils.tensorutils import (
                bilinear_product, 
                create_placeholder,
                int_shapes)
from embedKB.training.trainer import TrainWrapper
import numpy as np
import os

### WORK IN PROGRESS!

class Model(object):
    def embed_entity(self, entity_ids):
        """
        Uses the entity embedding matrix to convert entity id's into their vector forms
        :param entity_ids: a list of the entity id's that need to be converted into vectors
        :return: A tensor of shape (len(entity_ids), entity_embed_dim) 
        """
        return tf.nn.embedding_lookup(self.W_entity_embedding, entity_ids)

    def entity_shape_correct(self, entity_embeddings, name=None):
        return tf.reshape(entity_embeddings,
                [-1, self.entity_embed_dim, 1], name=name)
    
    def relationship_shape_correct(self, relationship_embeddings, name=None):

        dims = int_shapes(relationship_embeddings)
        if len(dims) == 3: 
            return tf.reshape(relationship_embeddings, [-1, 1, self.relationship_embed_dim], name=name)
        elif len(dims) == 4:
            # vectors/matrices
            if dims[2] == 1:
                return tf.reshape(relationship_embeddings, [-1, 1, self.relationship_embed_dim], name=name)
            else:
                return tf.reshape(relationship_embeddings, [-1, self.relationship_embed_dim, self.entity_embed_dim], name=name)
        elif len(dims) == 5:
            # tensors
            return tf.reshape(relationship_embeddings, [-1, self.entity_embed_dim, self.entity_embed_dim, self.relationship_embed_dim])
        else:
            raise ValueError('The relationship embeddings are vectors, matrices or tensors')

    def score_function(self, head_entity_id, relationship, tail_entity_id):
        """
        The score function to be used for ranking/training
        """
        raise NotImplementedError('You must implement a score function')


class GeneralFramework(Model, TrainWrapper):
    name = 'GeneralFramework'
    def __init__(self, n_entities, entity_embed_dim, entity_embed_initializer=None):
        """
        General Framework as defined in 
        Yang, Bishan, et al. 
        "Learning multi-relational semantics using neural-embedding models." arXiv preprint arXiv:1411.4072 (2014).
        https://arxiv.org/pdf/1412.6575.pdf
        """
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.increment_global_step = 1 + self.global_step
        with tf.variable_scope('entity'):
            self.W_entity_embedding = tf.get_variable('embedding_matrix',
                                                      shape=(n_entities, entity_embed_dim),
                                                      initializer=entity_embed_initializer) 

        self.entity_embed_dim = entity_embed_dim
        self.n_entities = n_entities

        self.head_entity_id = create_placeholder('entity_id_head')
        self.tail_entity_id = create_placeholder('entity_id_tail')
        self.relationship_id = create_placeholder('realtionship_id')
        self.relationship_id_false = create_placeholder('relationship_id_false')
        self.head_entity_id_false = create_placeholder('entity_id_head_false')
        self.tail_entity_id_false = create_placeholder('entity_id_tail_false')

        with tf.name_scope('positive_triple'):
            self.score = self.score_function(self.head_entity_id, self.relationship_id, self.tail_entity_id)
        with tf.name_scope('negative_triple'):
            self.score_false = self.score_function(self.head_entity_id_false, self.relationship_id_false, self.tail_entity_id_false)

    def g_linear(self, embedded_head, relationship_matrix_head, relationship_matrix_tail, embedded_tail):
        """
        The linear component of the scoring function
        :param embedded_head: the vector representation of the tail head
        :param relationship_matrix_head: the relationship matrix that corresponds to the 
                                         first component of A_r from the paper.
        :param relationship_matrix_tail: the relationship matrix that corresponds to the
                                         second component of A_r from the paper.
        :param embedded_tail: the vector representation of the tail entity
        :return: tensor representing A_r.T * concat[embedded_head, embedded_tail]
        """
        embedded_head = self.entity_shape_correct(embedded_head, 'e_head')
        embedded_tail = self.entity_shape_correct(embedded_tail, 'e_tail')
        relationship_matrix_head = self.relationship_shape_correct(relationship_matrix_head, 'r_head')
        relationship_matrix_tail = self.relationship_shape_correct(relationship_matrix_tail, 'r_tail')
        print(relationship_matrix_head)
        to_return = tf.matmul(relationship_matrix_head, embedded_head)
        to_return += tf.matmul(relationship_matrix_tail, embedded_tail) 
        return tf.squeeze(to_return, axis=-1, name='g_linear')

    def g_bilinear(self, embedded_head, relationship_matrix, embedded_tail):
        """
        The bilinear component of the scoring function
        :param embedded_head: the vector representation of the tail head
        :param relationship_matrix: the relationship matrix that corresponds to A_r from the paper.
        :param embedded_tail: the vector representation of the tail entity
        :return: the bilinear_product embedded_head.T * relationship_matrix * embedded_tail
        """
        embedded_head = self.entity_shape_correct(embedded_head, 'e_head')
        embedded_tail = self.entity_shape_correct(embedded_tail, 'e_tail')
        relationship_matrix = self.relationship_shape_correct(relationship_matrix, 'r_tensor')
        return bilinear_product(embedded_head, relationship_matrix, embedded_tail)

    def _scoring_function(self, embedded_head, relationship, embedded_tail):
        """
        you must derive your score function in this framework.
        if you do, then all you have to implement is this method.
        If not, you can override `score_function` and everything should still work
        """
        raise NotImplementedError('_scoring_function must be implemented!')

    def score_function(self, head_entity_id, relationship, tail_entity_id):
        """

        """
        with tf.name_scope('entity_embedding'):
            head_entity_embedded = self.embed_entity(head_entity_id)
            tail_entity_embedded = self.embed_entity(tail_entity_id)
        score_result = self._scoring_function(head_entity_embedded, relationship, tail_entity_embedded)
        assert int_shapes(score_result) == [-1, 1], 'Score result was: {}, expected [-1, 1]'.format(int_shapes(score_result))
        return score_result
