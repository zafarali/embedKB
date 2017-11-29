from embedKB.models.base import GeneralFramework
import tensorflow as tf
from embedKB.utils import debug


class StructuredEmbedding(GeneralFramework):
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



class StructuredEmbedding_External(TrainWrapper):
    name = 'StructuredEmbedding_External'
    # an external implementation of TRANSE 
    # from https://github.com/thunlp/TensorFlow-TransX/blob/master/transE.py
    def __init__(self, n_entities, embed_dim, n_relationships):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.increment_global_step = 1 + self.global_step
        with tf.variable_scope('entity'):
            self.W_entity_embedding = tf.get_variable('embedding_matrix',
                                                      shape=(n_entities, embed_dim),
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False)) 
        with tf.variable_scope('relationships'):
            self.W_relationship_embedding = tf.get_variable('embedding_matrix',
                                                        shape=(n_relationships, embed_dim),
                                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.entity_embed_dim = embed_dim
        self.n_entities = n_entities
        self.n_relationships = n_relationships

        self.head_entity_id = tensorutils.create_placeholder('entity_id_head')
        self.tail_entity_id = tensorutils.create_placeholder('entity_id_tail')
        self.relationship_id = tensorutils.create_placeholder('realtionship_id')
        self.relationship_id_false = tensorutils.create_placeholder('relationship_id_false')
        self.head_entity_id_false = tensorutils.create_placeholder('entity_id_head_false')
        self.tail_entity_id_false = tensorutils.create_placeholder('entity_id_tail_false')

        with tf.name_scope('positive_triple'):
            self.score = self.score_function(self.head_entity_id, self.relationship_id, self.tail_entity_id)
        with tf.name_scope('negative_triple'):
            self.score_false = self.score_function(self.head_entity_id_false, self.relationship_id_false, self.tail_entity_id_false)

    def score_function(self, heads, relationships, tails):
        with tf.name_scope('score_function'):
            heads = tf.nn.embedding_lookup(self.W_entity_embedding, heads)
            relationships = tf.nn.embedding_lookup(self.W_relationship_embedding, relationships)
            tails = tf.nn.embedding_lookup(self.W_entity_embedding, tails)
            return tf.reduce_sum(tf.abs(heads + relationships - tails),
                                 axis=1)
