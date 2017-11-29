import tensorflow as tf
from embedKB.utils import tensorutils
from embedKB.models.base import Model

class RelationshipVector(Model):
    def relationship_shape_correct(self, relationship_embeddings, name=None):
        raise DeprecationWarning('Relationship shape correct has been deprecated. Functionality is now included in base.Model')

class RelationshipMatrix(Model):
    def relationship_shape_correct(self, relationship_embeddings, name=None):    
        raise DeprecationWarning('Relationship shape correct has been deprecated. Functionality is now included in base.Model')

class RelationshipTensor(RelationshipMatrix):
    def relationship_shape_correct(self, relationship_embeddings, name=None,):
        raise DeprecationWarning('Relationship shape correct has been deprecated. Functionality is now included in base.Model')
