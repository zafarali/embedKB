from embedKB.models.base import GeneralFramework
import tensorflow as tf

class DistanceEmbedding(GeneralFramework):
    name = 'DistanceEmbedding'
    def __init__(self,
                 n_entities,
                 entity_embed_dim,
                 n_relationships,
                 relationship_dim,
                 relationship_embed_initalizer=None,
                 entity_embed_initializer=None):
        with tf.variable_scope('DistanceEmbedding'):
            with tf.variable_scope('relationship'):
                self.W_relationship_embedding_head = tf.get_variable('embedding_matrix_head',
                                                                      shape=(n_relationships, relationship_dim, entity_embed_dim),
                                                                      initializer=relationship_embed_initalizer)
                self.W_relationship_embedding_tail = tf.get_variable('embedding_matrix_tail',
                                                                      shape=(n_relationships, relationship_dim, entity_embed_dim),
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

        with tf.name_scope('scoringfunction'):
            linear_part = self.g_linear(embedded_head,
                                         relationship_vectors_head,
                                        -1.0 * relationship_vectors_tail,
                                         embedded_tail)

            norm = tf.norm(linear_part, ord=1, axis=1)
            norm_2d = tf.reshape(norm, [-1, 1])   #make column vector

        return -1 * norm_2d