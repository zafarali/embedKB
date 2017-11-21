"""
This script will show you how train the Structured Embedding model 
and benchmark it on the Triple Classification Task. 
We will use the FBK15 Dataset available at
https://www.microsoft.com/en-us/download/details.aspx?id=52312

Structured Embeddings:
Bordes, Antoine, et al.
"Learning Structured Embeddings of Knowledge Bases." 
AAAI. Vol. 6. No. 1. 2011.

Triple Classification Task:
Socher, Richard, et al. 
"Reasoning with neural tensor networks for knowledge base completion." 
Advances in neural information processing systems. 2013. 

"""
import tensorflow as tf

# embedKB.models implments a few standard models
from embedKB.models.se import StructuredEmbedding

# embedKB.datatools has tools to manipulate knolwedge base files.
# see the README for compatible formats
from embedKB.datatools import KnowledgeBase, Dataset

# embedKB.benchmark has tasks for benchmarking
from embedKB.benchmark import TripleClassificationTask

# define the dimensions for the embeddings in the model
entity_embed_dim = 10
relationship_embed_dim = 12

# define the batch size for the model
batch_size = 32


# load the knowledge base.

# load the data and convert triples into numpy arrays
kb = KnowledgeBase.load_from_raw_data('../data/Release/train.txt')
kb.convert_triples()

# create a dataset that we can learn from
# this implements negative sampling!
dset = Dataset(kb, batch_size=batch_size)

# derive a knowledge base of validation
kb_val = KnowledgeBase.derive_from(kb)
kb_val.load_raw_triples('../data/Release/valid.txt')
kb_val.convert_triples()
val_dset = Dataset(kb_val, batch_size=256)

# instantiate the model
framework = StructuredEmbedding(kb.n_entities,
				   				entity_embed_dim,
				   				kb.n_relations,
				   				relationship_embed_dim)

# you can score an individual triple like this:
# print(framework.score_triple(1, 4, 3))

# create the objective
# here we specify which embedding matrices we want to regularize
# and the regularization weight (by default there is no regularization)
framework.create_objective(regularize=[framework.W_relationship_embedding_head,
									   framework.W_relationship_embedding_tail,
									   framework.W_entity_embedding],
						   regularization_weight=0.01)

# by default it uses SGD
framework.create_optimizer(optimizer=tf.train.AdamOptimizer)
# create summaries for visualization in tensorboard
framework.create_summaries()

# this will print out stats every 100 batches
framework.train(dset,
				epochs=10,
				val_data=val_dset)


# we could just evaluate:
#framework.evaluate(...)
# or move on with the task:

# we use workers to specify how many threads
# to use to prepare data
# this will implement the smart corruption
# and calculate the thresholds
tct = TripleClassificationTask(dset, workers=5)
tct.compute_threshold_values(framework, val_dset)

# ideally we'd use the testing set here to bechmark
# once we've figured out good parameters.
tct.benchmark(val_dset, framework)