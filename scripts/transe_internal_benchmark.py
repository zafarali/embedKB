"""

"""
import os
import sys
sys.path.insert(0, "/home/ml/zahmed8/dev/embedKB")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
from embedKB.models.transe import TransE
from embedKB.datatools import KnowledgeBase, Dataset
from embedKB.benchmark import TripleClassificationTask

entity_embed_dim = 100
batch_size = 32
regularization_weight = 0.001
total_epochs = 100
init_learning_rate = 0.01

# load the data and convert triples into numpy arrays
kb = KnowledgeBase.load_from_raw_data('../data/Release/train.txt')
kb.convert_triples()
kb.save_converted_triples('./transE_internal/triples.npy')
kb.save_mappings_to_json('./transE_internal')

# create a TRUE and FALSE triple:
true_triple = (kb.entities['/m/07sbbz2'],
			   kb.relations['/music/genre/artists'],
			   kb.entities['/m/0p7h7'])
false_triple = (
			kb.entities['/m/07sbbz2'],
			kb.relations['/music/genre/artists'],
			kb.entities['/m/01386_']
)
# create a dataset that we can learn from
# this implements negative sampling!
dset = Dataset(kb, batch_size=batch_size)

# derive a knowledge base of validation
kb_val = KnowledgeBase.derive_from(kb)
kb_val.load_raw_triples('../data/Release/valid.txt')
kb_val.convert_triples()
val_dset = Dataset(kb_val, batch_size=batch_size)

# instantiate the model
framework = TransE(kb.n_entities,
				   entity_embed_dim,
				   kb.n_relations)

# you can score an individual triple like this:
# print(framework.score_triple(1, 4, 3))

# create the objective
framework.create_objective(regularize=[framework.W_relationship_embedding,
									   framework.W_entity_embedding],
						   regularization_weight=regularization_weight)

framework.create_optimizer(optimizer=tf.train.AdagradOptimizer,
						   optimizer_args={'learning_rate':init_learning_rate})

# create summaries for visualization in tensorboard
framework.create_summaries()
print('True triple score:', framework.score_triple(*true_triple))
print('False triple score:',framework.score_triple(*false_triple))

framework.evaluate(dset, name='training/start')
framework.evaluate(val_dset, name='val/start')
# this will print out stats every 100 batches
framework.train(dset,
				epochs=total_epochs,
				val_data=val_dset,
				batch_log_frequency=1000,
				logging_directory='./transE_internal/checkpoints')

framework.save_model('./transE_internal/transE_internal')
print('True triple score:', framework.score_triple(*true_triple))
print('False triple score:',framework.score_triple(*false_triple))

# we use workers to specify how many threads
# to use to prepare data
# this will implement the smart corruption
# and calculate the thresholds
tct = TripleClassificationTask(dset, workers=5)
tct.compute_threshold_values(framework, val_dset)


# derive a knowledge base for testing
kb_test = KnowledgeBase.derive_from(kb)
kb_test.load_raw_triples('../data/Release/test.txt')
kb_test.convert_triples()
test_dset = Dataset(kb_test, batch_size=batch_size)
# ideally we'd use the testing set here to bechmark
# once we've figured out good parameters.
print('VAL DATASET SCORES:')
tct.benchmark(val_dset, framework, batch_log_frequency=250)
print('TEST DATASET SCORES:')
tct.benchmark(test_dset, framework, batch_log_frequency=250)
