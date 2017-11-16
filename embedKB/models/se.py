from embedKB.models.base import GeneralFramework
from embedKB.models.relationship_types import RelationshipMatrix
import tensorflow as tf
from embedKB.utils import debug


class StructuredEmbedding(GeneralFramework, RelationshipMatrix):
    name = 'StructuredEmbedding'
    def __init__(self,
                 n_entities,
                 entity_embed_dim,
                 n_relationships,
                 relationship_dim,
                 relationship_embed_initalizer=None,
                 entity_embed_initializer=None):
        """
        Structured Embedding model from
        Bordes, Antoine, et al. "Learning Structured Embeddings of Knowledge Bases." AAAI. Vol. 6. No. 1. 2011.
        https://ronan.collobert.com/pub/matos/2011_knowbases_aaai.pdf
        """
        with tf.variable_scope('StructuredEmbedding'):
            with tf.variable_scope('relationship'):
                self.W_relationship_embedding_head = tf.get_variable('embedding_matrix_head',
                                                                      shape=(n_relationships, relationship_dim, entity_embed_dim),
                                                                      initializer=relationship_embed_initalizer)
                self.W_relationship_embedding_tail = tf.get_variable('embedding_matrix_tail',
                                                                      shape=(n_relationships, relationship_dim, entity_embed_dim),
                                                                      initializer=relationship_embed_initalizer)

                # No bilinear relationship.
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

            # relationship_vectors = debug.tfPrint(relationship_vectors)

        with tf.name_scope('scoringfunction'):
            linear_part = 2 * self.g_linear(embedded_head,
                                            relationship_vectors_head,
                                            relationship_vectors_tail,
                                            embedded_tail)
            linear_part = -tf.norm(linear_part, ord=1, axis=1)
            # to keep it consistent with (Bx1) notation.
            return tf.reshape(linear_part, [-1, 1])
