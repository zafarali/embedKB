from embedKB.models.base import GeneralFramework
import tensorflow as tf

class Bilinear(GeneralFramework):
    name = 'Bilinear'
    def __init__(self,
                 n_entities,
                 entity_embed_dim,
                 n_relationships,
                 relationship_dim,
                 relationship_embed_initalizer=None,
                 entity_embed_initializer=None):
        with tf.variable_scope('Bilinear'):
            with tf.variable_scope('relationship'):
                self.M_relationship_matrix = tf.get_variable('relationship_matrix',
                                                             shape=(n_relationships, entity_embed_dim, entity_embed_dim),
                                                             initializer=relationship_embed_initalizer)
            self.relationship_embed_dim = relationship_dim
            self.n_relationships = n_relationships
            # Initialize entity embedding
            super().__init__(n_entities,
                             entity_embed_dim,
                             entity_embed_initializer=entity_embed_initializer)

    def _scoring_function(self, embedded_head, relationship, embedded_tail):
        with tf.variable_scope('relationship'):
            relationship_matrix = tf.nn.embedding_lookup(self.M_relationship_matrix,
                                                          relationship)

        with tf.name_scope('scoringfunction'):
            bilinear_part = self.g_bilinear(embedded_head, relationship_matrix, embedded_tail)

        return bilinear_part