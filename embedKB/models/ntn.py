from embedKB.models.base import GeneralFramework
import tensorflow as tf
from embedKB.utils import debug


class NeuralTensorNetwork(GeneralFramework):
    name = 'NeuralTensorNetwork'
    def __init__(self,
                 n_entities,
                 entity_embed_dim,
                 n_relationships,
                 relationship_dim,
                 relationship_embed_initalizer=None,
                 entity_embed_initializer=None):
        """
        Implementation of 
        Socher, Richard, et al.
        "Reasoning with neural tensor networks for knowledge base completion."
        Advances in neural information processing systems. 2013.
        http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf
        """

        with tf.variable_scope('NTN'):
            with tf.variable_scope('relationship'):
                self.W_relationship_embedding_head = tf.get_variable('embedding_matrix_head',
                                                                      shape=(n_relationships, relationship_dim, entity_embed_dim),
                                                                      initializer=relationship_embed_initalizer)
                self.W_relationship_embedding_tail = tf.get_variable('embedding_matrix_tail',
                                                                      shape=(n_relationships, relationship_dim, entity_embed_dim),
                                                                      initializer=relationship_embed_initalizer)
                self.T_relationship_tensor = tf.get_variable('relationship_tensor',
                                                                      shape=(n_relationships, entity_embed_dim, entity_embed_dim, relationship_dim),
                                                                      initializer=relationship_embed_initalizer)
                self.U_relationship_vector = tf.get_variable('relationship_vector',
                                                                      shape=(n_relationships, relationship_dim),
                                                                      initializer=relationship_embed_initalizer)
            self.relationship_embed_dim = relationship_dim
            self.n_relationships = n_relationships
            # Initialize entity embedding
            super().__init__(n_entities,
                             entity_embed_dim,
                             entity_embed_initializer=entity_embed_initializer)
    
    def _scoring_function(self, embedded_head, relationship, embedded_tail):
        with tf.variable_scope('relationship'):
            relationship_vectors_head = tf.nn.embedding_lookup(self.W_relationship_embedding_head,
                                                          relationship)
            relationship_vectors_tail = tf.nn.embedding_lookup(self.W_relationship_embedding_tail,
                                                          relationship)
            relationship_tensors = tf.nn.embedding_lookup(self.T_relationship_tensor,
                                                           relationship)
            u_r = self.relationship_shape_correct(tf.nn.embedding_lookup(self.U_relationship_vector, relationship), 'u_r')

        with tf.name_scope('scoringfunction'):
            linear_part = self.g_linear(embedded_head,
                                         relationship_vectors_head,
                                         relationship_vectors_tail,
                                         embedded_tail)
            bilinear_part = self.g_bilinear(embedded_head, relationship_tensors, embedded_tail)
            # print('u_r (prereshape)',u_r)
            print('u_r (post reshape)',tf.reshape(u_r, [-1, self.relationship_embed_dim]))
            print('tanh:',tf.tanh(linear_part + bilinear_part))
            a = tf.reshape(
                    tf.reduce_sum(
                        tf.multiply(
                            tf.reshape(u_r, [-1, self.relationship_embed_dim]),
                            tf.tanh(linear_part + bilinear_part)),
                        axis=[1]), 
                    [-1, 1], name='result')

            return a
