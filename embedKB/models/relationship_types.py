import tensorflow as tf
from embedKB.models.base import Model

class RelationshipVector(Model):
    def relationship_shape_correct(self, relationship_embeddings, name=None):
        return tf.reshape(relationship_embeddings,
                [-1, 1, self.entity_embed_dim], name=name)

class RelationshipMatrix(Model):
    def relationship_shape_correct(self, relationship_embeddings, name=None):    
        return tf.reshape(relationship_embeddings, 
                [-1, self.relationship_embed_dim, self.entity_embed_dim], name=name)

class RelationshipTensor(Model):
    def relationship_shape_correct(self, relationship_embeddings, name=None):
        raise NotImplementedError('Tensor methods are not yet supported')