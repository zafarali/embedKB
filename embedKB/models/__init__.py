from embedKB.models.bilinear import Bilinear
from embedKB.models.transe import TransE
from embedKB.models.se import StructuredEmbedding
from embedKB.models.ntn import NeuralTensorNetwork
from embedKB.models.singlelayer import SingleLayer
import tensorflow as tf


def get_model(model_name,
              n_entities,
              n_relations,
              entity_embed_dim=100,
              relationship_embed_dim=None,
              regularization_weight=0,
              init_learning_rate=0.01):
    if relationship_embed_dim is None:
        relationship_embed_dim = entity_embed_dim

    if model_name == "StructuredEmbedding":
        framework = StructuredEmbedding(n_entities,
                                      entity_embed_dim,
                                      n_relations,
                                      relationship_embed_dim)
        framework.create_objective(regularize=[framework.W_relationship_embedding_head,
                                               framework.W_relationship_embedding_tail,
                                               framework.W_entity_embedding],
                                   regularization_weight=regularization_weight)
    elif model_name == "NeuralTensorNetwork":
        framework = NeuralTensorNetwork(n_entities,
                                        entity_embed_dim,
                                        n_relations,
                                        relationship_embed_dim)

        framework.create_objective(regularize=[framework.W_relationship_embedding_head,
                                               framework.W_relationship_embedding_tail,
                                               framework.T_relationship_tensor,
                                               framework.U_relationship_vector,
                                               framework.W_entity_embedding],
                                   regularization_weight=regularization_weight)

    elif model_name == 'Bilinear':
        framework = Bilinear(n_entities,
                             entity_embed_dim,
                             n_relations,
                             relationship_embed_dim)
        framework.create_objective(regularize=[framework.M_relationship_matrix],
                                   regularization_weight=regularization_weight)
    elif model_name == 'SingleLayer':
        framework = SingleLayer(n_entities,
                                entity_embed_dim,
                                n_relations,
                                relationship_embed_dim)
        framework.create_objective(regularize=[framework.W_relationship_embedding_head,
                                               framework.W_relationship_embedding_tail,
                                               framework.W_entity_embedding],
                                   regularization_weight=regularization_weight)
    elif model_name == 'TransE':
        framework = TransE(n_entities,
                           entity_embed_dim,
                           n_relations)
        framework.create_objective(regularize=[framework.W_relationship_embedding,
                                               framework.W_entity_embedding],
                                   regularization_weight=regularization_weight)

    else:
        raise ValueError('The model specified does not exist')

    framework.create_optimizer(optimizer=tf.train.AdagradOptimizer,
                               optimizer_args={'learning_rate': init_learning_rate})
    return framework