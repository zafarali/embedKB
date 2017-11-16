import numpy as np


def negative_sampling(data, n_possibilities):
    """
    Implements vanilla negative sampling. 
    Either the head entity or the tail entity is replaced with an entity
    from the total number of possible entities.
    """
    # select whether we should replace the head or the tails
    data = data.copy()
    entity_to_replace = np.random.choice([0, 2], replace=True, size=data.shape[0])
    entity_to_replace_with = np.random.randint(n_possibilities, size=data.shape[0])
    data[np.arange(0, data.shape[0]), entity_to_replace] = entity_to_replace_with
    return data

class Dataset(object):
    def __init__(self, knowledge_base, batch_size=32, sampler=negative_sampling):
        self.all_data = np.array(knowledge_base.triples, dtype=np.int32)
        self.kb = knowledge_base
        self.batch_size = batch_size
        self.sampler = sampler

    def get_generator(self):
        shuffled_idx = np.random.permutation(self.kb.n_triples)

        for i in range(0, self.kb.n_triples, self.batch_size):
            idx = shuffled_idx[i: i+self.batch_size]
            selection_idx = np.zeros(self.kb.n_triples)
            selection_idx[idx] = 1
            selection_idx = selection_idx.astype(bool)
            minibatch = self.all_data[selection_idx]
            positive_data = (minibatch[:, 0].reshape(-1,1), 
                             minibatch[:, 1].reshape(-1,1), 
                             minibatch[:, 2].reshape(-1,1))
            negative_minibatch = self.sampler(minibatch, self.kb.n_entities)
            negative_data = (negative_minibatch[:, 0].reshape(-1,1),
                             negative_minibatch[:, 1].reshape(-1,1),
                             negative_minibatch[:, 2].reshape(-1,1))
            yield positive_data, negative_data