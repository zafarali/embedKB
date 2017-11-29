from embedKB.models.base import GeneralFramework
import tensorflow as tf
from embedKB.utils import debug
from embedKB.utils import tensorutils

class TransE(GeneralFramework):
    name = 'TransE'
    def __init__(self,
                 n_entities,
                 embed_dim,
                 n_relationships,
                 relationship_embed_initalizer=None,
                 entity_embed_initializer=None):
        """
        TransE model from
        Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." 
        Advances in neural information processing systems. 2013.
        https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
        """
        # Initialize relationship embedding
        with tf.variable_scope('TransE'):
            with tf.variable_scope('relationship'):
                self.W_relationship_embedding = tf.get_variable('embedding_matrix',
                                                                shape=(n_relationships, embed_dim),
                                                                initializer=relationship_embed_initalizer)
                self.W_bilinear_relationship = tf.reshape(
                                                tf.tile(tf.eye(embed_dim),
                                                        [n_relationships, 1]),
                                                [n_relationships, embed_dim, embed_dim], name='identity')
            
            self.relationship_embed_dim = embed_dim
            self.n_relationships = n_relationships
            # Initialize entity embedding
            super().__init__(n_entities,
                             embed_dim,
                             entity_embed_initializer=entity_embed_initializer)

    def _scoring_function(self, embedded_head, relationship, embedded_tail):
        with tf.variable_scope('relationship'):
            relationship_vectors = tf.nn.embedding_lookup(self.W_relationship_embedding,
                                                          relationship)
            relationship_matrices = tf.nn.embedding_lookup(self.W_bilinear_relationship,
                                                           relationship)

        with tf.name_scope('scoringfunction'):
            print(tensorutils.int_shapes(relationship_vectors))
            linear_part = 2 * self.g_linear(embedded_head,
                                            relationship_vectors,
                                            -1.0 * relationship_vectors,
                                            embedded_tail)
            bilinear_part = -2 * self.g_bilinear(embedded_head, relationship_matrices, embedded_tail)
            norm = tf.square(tf.norm(relationship_vectors, ord='euclidean', axis=2))
            return linear_part + bilinear_part + norm
