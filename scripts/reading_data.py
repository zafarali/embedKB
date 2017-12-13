import os
import sys
sys.path.insert(0, "/home/ml/zahmed8/dev/embedKB")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
# from embedKB.models.ntn import NeuralTensorNetwork
from embedKB.models.transe import TransE
from embedKB.datatools import KnowledgeBase, Dataset
from embedKB.benchmark import TripleClassificationTask

entity_embed_dim = 100
relationship_embed_dim = 50
batch_size = 32
regularization_weight = 0.001
total_epochs = 30
init_learning_rate = 0.01


kb = KnowledgeBase()
kb.load_converted_triples('./transE_internal/triples.npy')
kb.load_mappings_from_json('./transE_internal/entity2id.json',
						   './transE_internal/relation2id.json')

# framework = NeuralTensorNetwork(kb.n_entities,
# 				   entity_embed_dim,
# 				   kb.n_relations,
# 				   relationship_embed_dim)
framework = TransE(kb.n_entities,
				   entity_embed_dim,
				   kb.n_relations)
# create the objective
framework.create_objective()

framework.create_optimizer(optimizer=tf.train.AdagradOptimizer,
						   optimizer_args={'learning_rate':0.01})

# create summaries for visualization in tensorboard
framework.create_summaries()
framework.load_model('./transE_internal/transE_internal')

dset = Dataset(kb, batch_size=batch_size)

# derive a knowledge base of validation
kb_val = KnowledgeBase.derive_from(kb)
kb_val.load_raw_triples('../data/Release/valid.txt')
kb_val.convert_triples()
val_dset = Dataset(kb_val, batch_size=batch_size)
# framework.evaluate(dset, name='training')
# framework.evaluate(val_dset, name='val')

# create a TRUE and FALSE triple:
true_triple = (kb.entities['/m/07sbbz2'],
			   kb.relations['/music/genre/artists'],
			   kb.entities['/m/0p7h7'])
false_triple = (
			kb.entities['/m/07sbbz2'],
			kb.relations['/music/genre/artists'],
			kb.entities['/m/01386_']
)
print('True triple score:', framework.score_triple(*true_triple))
print('False triple score:',framework.score_triple(*false_triple))

tct = TripleClassificationTask(dset, workers=5)
tct.compute_threshold_values(framework, val_dset)
tct.benchmark(dset, framework)
tct.benchmark(val_dset, framework)
# tct.benchmark(val_dset, framework)