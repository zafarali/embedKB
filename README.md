# EmbedKB

The goal of this repository is to allow rapid immplementation of knowledge base embedding models and evaluation on tasks. 

Some key features:

- Implementations of common knowledge base embedding models
- Implementation of the General Framework as described in [1]
- Full integration with Tensorboard including Embeddings visualizer
- Benchmarking tasks
- Knowledge base data manipulation functions.
- Unit testing

To install see [Installation](#installation). To get a brief overview of the features see [the introduction section](#easy-to-use) To use the command line interface to train and benchmark models see [training](#Training) and [Benchmarking](#benchmarking).

## Installation

from this directory run

```
pip3 install -e . --user
```

this way you install a development version of the module.
This code has been tested with Tensorflow 1.2.0


## Easy to use!

If you have data in the form of a knowledge base (for example [FBK15](https://www.microsoft.com/en-us/download/details.aspx?id=52312)) you can get started and train knolwedge base embeddings in a few lines of code!

```
# we want to use the StructuredEmbedding model:
from embedKB.models.se import StructuredEmbedding

# data handling techniques:
from embedKB.datatools import KnowledgeBase
from embedKB.datatools import Dataset

# load the training data
kb_train = KnowledgeBase.load_from_raw_data('./data/train.txt')
kb_train.convert_triples() # convert the triples into a numpy format
train_dset = Dataset(kb_train, batch_size=32) # a wrapper that implements negative sampling

framework.create_objective() # create the max-margin loss objective
framework.create_optimizer() # create the optimizer
framework.create_summaries() # create the summaries (optional)

# train!
framework.train(train_dset,
                 epochs=15)
```

To ask for the "score" for any given triple you can do `framework.score_triple(1, 4, 5)` or there is a batch mode that is available.


## Data

### Knowledge Base Preparation

Make sure that the triples are in a tab separated file of the form:
```
head_entity  relationship  tail_entity
head_entity  relationship  tail_entity
head_entity  relationship  tail_entity
```

You can then use `embedKB.datatools.KnowledgeBase` to manipulate and save the knowledge base into an appropriate format for downstream training:

```
from embedKB.datatools import KnowledgeBase

# load the raw txt files:
# this will also create a dict with the entity mappings.
kb = KnowledgeBase.load_from_raw_data('../data/train.txt')

# convert the triples from the file ../data/train.txt
# into a numpy array using the dicts we created above.
kb.convert_triples()
print(kb.n_triples) # this will print the number of triples available

# save the numpy converted triples
# save the mappings
kb.save_converted_triples('./processed/train.npy')
kb.save_mappings_to_json('./processed/')
```

### Negative Sampling and data consumption

Embeddings are usually trained with negative sampling. The object `embedKB.datatools.Dataset` implements this and will allow us to consume for learning. First we load our training and validation data:

```
# this reloads our training knowledge base
kb_train = KnowledgeBase()
# mappings get saved into standard names:
kb_train.load_mappings_from_json('./processed/entity2id.json', './processed/relationship2id.json')
kb_train.load_converted_triples('./train.npy')

# we now create a validation knowledge base:
# this just reuses the entities and relationss from `kb_train`
kb_val = KnowledgeBase.derive_from(kb_train)
# since we have not yet converted our validation data
# we load the raw triples.
kb_val.load_raw_triples('./data/valid.txt')
# as before, use this function to convert triples into numpy format.
kb_val.convert_triples()
```

The `Dataset` object takes in a `KnowledgeBase` and makes it ready for use in training. You must specify a `batch_size` during creation:

```
train_dset = Dataset(kb_train, batch_size=64)
val_dset = Dataset(kb_val, batch_size=64)
```

This is what you will feed into the Embedding models. The `Dataset` object has a generator which does negative sampling on the fly. To inspect a single batch:

```
print(next(train_dset.get_generator()))
```

You will see that it contains a tuple each with a tuple of three numpy arrays representing head_entity_ids, relationship_ids and tail_entity_ids.

## Tasks

There are currently two tasks implemented for benchmarking:

1. Triple Classification
2. Entity Prediction

It's as easy as a few lines:

```
# using the filtered version of the task:
task = EntityPredictionTask(kb, workers=5, filtered=True)
task.benchmark(val_dset, framework)
```

# Training

Model under from [1] are implemented for you. You can use `./scripts/training.py` to run them. 

```
usage: training.py [-h] -m MODEL_NAME [-e ENTITY_DIM] [-r RELATION_DIM]
                   [-data DATA] [-reg REG_WEIGHT] [-lr INIT_LEARNING_RATE]
                   [-gpu GPU] [-n_epochs N_EPOCHS]
                   [-batch_log_freq BATCH_LOG_FREQ] [-batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
                        model to run
  -e ENTITY_DIM, --entity_dim ENTITY_DIM
                        model to run
  -r RELATION_DIM, --relation_dim RELATION_DIM
                        model to run
  -data DATA, --data DATA
                        location of the data
  -reg REG_WEIGHT, --reg_weight REG_WEIGHT
                        regularization weight
  -lr INIT_LEARNING_RATE, --init_learning_rate INIT_LEARNING_RATE
                        initial learning rate
  -gpu GPU              ID of GPU to execute on
  -n_epochs N_EPOCHS, --n_epochs N_EPOCHS
                        number of epochs
  -batch_log_freq BATCH_LOG_FREQ, --batch_log_freq BATCH_LOG_FREQ
                        logging frequency
  -batch_size BATCH_SIZE, --batch_size BATCH_SIZE
```

For example:

```
cd scripts
python3 training.py -m TransE -e 50 -r 50 -data '../data/Release' -n_epochs 100
```

Will run TransE. The data and models are check pointed and saved into `./TransE`.

# Benchmarking

You can find the script for bencmarking in `./scripts` as well.

```
python3 scripts/benchmark.py -h
usage: benchmark.py [-h] -m MODEL_NAME [-e ENTITY_DIM] [-r RELATION_DIM] -f
                    FOLDER [-gpu GPU] -t TASK

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
                        model to run
  -e ENTITY_DIM, --entity_dim ENTITY_DIM
                        model to run
  -r RELATION_DIM, --relation_dim RELATION_DIM
                        model to run
  -f FOLDER, --folder FOLDER
                        location of the model and kb
  -gpu GPU              ID of GPU to execute on
  -t TASK, --task TASK  the task to benchmark upon
```

Two tasks are already implemented:

- `ept`: Entity Prediction as described in [2]
- `tct`: Triple Classification as described in [3]

## Testing

There are a few unit tests. To run:


```
python3 -m pytest
```

# Reference

[1] [Yang, Bishan, et al. "Learning multi-relational semantics using neural-embedding models." arXiv preprint arXiv:1411.4072 (2014).](https://arxiv.org/pdf/1412.6575.pdf)

[2] [Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Advances in neural information processing systems. 2013.](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

[3] [Socher, Richard, et al. "Reasoning with neural tensor networks for knowledge base completion." Advances in neural information processing systems. 2013.](https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf)
