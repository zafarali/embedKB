# embedKB

The goal of this repository is to implement embedding models for knowledge bases according to the general framework in [1]

[1] [Yang, Bishan, et al. "Learning multi-relational semantics using neural-embedding models." arXiv preprint arXiv:1411.4072 (2014).](https://arxiv.org/pdf/1412.6575.pdf)

Some key features:

- Implementations of common knowledge base embedding models
- Full integration with Tensorboard including Embeddings visualizer
- Benchmarking tasks
- Knowledge base data manipulation functions.
- Unit testing

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
head_entity\trelationship\ttail_entity
head_entity\trelationship\ttail_entity
head_entity\trelationship\ttail_entity
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


## Installation

from this directory run

```
pip3 install -e . --user
```

this way you install a development version of the module.
This code has been tested with Tensorflow 1.2.0


## Testing

There are a few unit tests. To run:


```
python3 -m pytest
```

# How to use