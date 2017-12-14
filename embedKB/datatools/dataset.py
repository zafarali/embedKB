import numpy as np
import itertools
import multiprocessing

MAX_ITERATES = 5000

class Sampling(object):
    def __init__(self, seed=1):
        self.reset(seed)

    def reset(self, seed=1):
        self.rng = np.random.RandomState(seed)
        self.seed = seed

class NegativeSampling(Sampling):
    def __init__(self, seed=1):
        super().__init__(seed=seed)

    def sample(self, data, n_possibilities):
        """
        Implements vanilla negative sampling. 
        Either the head entity or the tail entity is replaced with an entity
        from the total number of possible entities.
        """
        # select whether we should replace the head or the tails
        data = data.copy()
        entity_to_replace = self.rng.choice([0, 2], replace=True, size=data.shape[0])
        entity_to_replace_with = self.rng.randint(n_possibilities, size=data.shape[0])
        data[np.arange(0, data.shape[0]), entity_to_replace] = entity_to_replace_with
        return data

class SmartNegativeSampling(Sampling):
    """
    Implements smart negative sampling where the head or tail entity is replaced
    with an entity that is allowed at that position based on the relationship.
    
    To use this in Dataset:
    ```
        sns = SmartNegativeSampling(kb)
        dset = Dataset(kb, sampler=sns.smart_triple_corruption)
    ```
    """
    def __init__(self, kb, workers=4, seed=1):
        super().__init__(seed=seed)
        self.workers = workers
        self.kb = kb
        assert kb._converted_triples
        data = np.array(kb.triples, dtype=np.int32)
        
        # this stores the unique relations in our dataset
        unique_relations = np.unique(data[:, 1]).tolist()

        # we now find all possible heads and tails that satisfy the relation
        # this will be used in the classificaiton task as the triples
        # that should be scored negatively
        possible_heads = {}
        possible_tails = {}

        for r in unique_relations:
            possible_heads[r] = data[np.where(data[:, 1] == r), 0][0].tolist()
            possible_tails[r] = data[np.where(data[:, 1] == r), 2][0].tolist()

        self.unique_entities = list(set(itertools.chain.from_iterable(possible_tails.values())))
        self.possible_heads = possible_heads
        self.possible_tails = possible_tails
        self.unique_relations = unique_relations

    def pick_new_entity(self, current_entity, choose_from, max_iterates=MAX_ITERATES):
        """
        Picks a new entity from a list of entities in choose_from.
        :param current_entity: the current value of the entity
        :param choose_from: the list of entities we can choose from
        :result: the int representing the new entity to replace with.
        """
        new_entity = current_entity
        iterates = 0
        while new_entity == current_entity:
            new_entity = self.rng.choice(choose_from)
            iterates += 1
            if len(choose_from) == 1 or iterates > max_iterates:
                # we picked the same entity too many times
                # of there was only one entity to choose from anyway.
                new_entity = self.rng.choice(self.unique_entities)
        return new_entity

    def _corrupt(self, triple):
        # decice which entity to replace:
        entity_to_replace = self.rng.choice([0, 2])
        to_return = None
        if entity_to_replace == 0:
            new_entity = self.pick_new_entity(triple[0], self.possible_heads[triple[1]])
            to_return = (new_entity, triple[1], triple[2])
        elif entity_to_replace == 2:
            new_entity = self.pick_new_entity(triple[2], self.possible_tails[triple[1]])
            to_return = (triple[0], triple[1], new_entity)
        else:
            raise ValueError('Unknown entity to replace.')
        return to_return

    def sample(self, data_, *args):
        """
        As described in the paper, this corruption mechanism only creates
        _plausibly_ corrupt triples. As quoted from the original paper:

        "For example, given a correct triplet (Pablo Picaso, nationality, Spain),
        a potential negative example is (Pablo Picaso, nationality,United States). 
        This forces the model to focus on harder cases and makes the evaluation harder since it does not include obvious non-relations such 
        as (Pablo Picaso, nationality, Van Gogh)"

        Since this is more computationally intensive than regular negative sampling
        we make use of the multiprocessing module to ensure we can do it in parallel
        make sure to pass in workers > 1 in the constructor if you have more than one
        core available for this.

        :param data_: the data to do the corruption on.
        :param *args: for compatability reasons.
        """
        data = data_.copy()
        data = data.tolist()

        # use multiprocessing so that we can perform the triple corruption
        # in parallel
        with multiprocessing.Pool(self.workers) as pool:
            join = np.array(pool.map(self._corrupt, data))
        return join


class Dataset(object):
    def __init__(self,
                 knowledge_base,
                 batch_size=32,
                 sampler=NegativeSampling(),
                 inflation_factor=1):
        """
        Creates a wrapper around the knowledge base for training.
        This will create positive and negative batches for training
        :param knowledge_base: the knowledge base to learn from
        :param batch_size: the number of triples in each batch
        :param sampler: the sampler to use (by default random NegativeSampling is used)
        :param inflation_factor: the number of corrupted triples per triple in the knowledge base. 
        """
        self.all_data = np.array(knowledge_base.triples, dtype=np.int32)
        self.kb = knowledge_base
        self.batch_size = batch_size
        self.sampler = sampler
        assert inflation_factor >= 1, 'Cannot have an inflation factor < 1!'
        self.inflation_factor = inflation_factor

    def get_generator(self):
        shuffled_idx = self.sampler.rng.permutation(self.kb.n_triples)

        for i in range(0, self.kb.n_triples, self.batch_size):
            idx = shuffled_idx[i: i+self.batch_size]
            selection_idx = np.zeros(self.kb.n_triples)
            selection_idx[idx] = 1
            selection_idx = selection_idx.astype(bool)
            minibatch = self.all_data[selection_idx]

            if not minibatch.shape[0] > 0:
                raise StopIteration('End of minibatches')
                
            positive_data = (minibatch[:, 0].reshape(-1,1), 
                             minibatch[:, 1].reshape(-1,1), 
                             minibatch[:, 2].reshape(-1,1))

            for j in range(self.inflation_factor):
                negative_minibatch = self.sampler.sample(minibatch, self.kb.n_entities)
                negative_data = (negative_minibatch[:, 0].reshape(-1,1),
                                 negative_minibatch[:, 1].reshape(-1,1),
                                 negative_minibatch[:, 2].reshape(-1,1))
                yield positive_data, negative_data