from .task import Task
import numpy as np
from embedKB.datatools import Dataset
import multiprocessing
import itertools

class TripleClassificationTask(Task):
    def __init__(self, dataset, workers=1):
        """
        This class implements the Triple Classification Task
        commonly used to benchmark knowledge base embedding models.  
        Socher, Richard, et al. 
        "Reasoning with neural tensor networks for knowledge base completion." 
        Advances in neural information processing systems. 2013. 
        https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
        
        :param dataset: this data will be used to calculate possible head and tail
                         replacement entities. This is a dataset generator object
        :param workers: number of pools to use
        """
        # get the positive data
        data = dataset.all_data
        
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

        self.possible_heads = possible_heads
        self.possible_tails = possible_tails
        self.unique_relations = unique_relations
        self.dataset = dataset
        self.workers = workers

    def compute_threshold_values(self, model, dataset=None):
        threshold_values = np.zeros(len(self.unique_relations))
        
        dataset = self.dataset.all_data if not dataset else dataset.all_data

        for r in self.unique_relations:
            # contains correct triples that satisfy the relation
            data_subset = dataset[np.where(dataset[:, 1] == r), :][0]
            
            
            per_triple_score = model.batch_score_triples(data_subset[:, 0],
                                                         data_subset[:, 1],
                                                         data_subset[:, 2])

            # set the threshold value to be the mean of the per
            # triple score for that relation
            threshold_values[r] = per_triple_score.mean()

        self.threshold_values = threshold_values

    def _corrupt(self, triple):
        # decice which entity to replace:
        entity_to_replace = np.random.choice([0, 2])
        to_return = None
        if entity_to_replace == 0:
            new_entity = np.random.choice(self.possible_heads[triple[1]])
            to_return = (new_entity, triple[1], triple[2])
        elif entity_to_replace == 2:
            new_entity = np.random.choice(self.possible_heads[triple[1]])
            to_return = (triple[0], triple[1], new_entity)
        else:
            raise ValueError('Unknown entity to replace.')
        return to_return

    def smart_triple_corruption(self, data_, *args):
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
        entity_to_replace = np.random.choice([0, 2], replace=True, size=data.shape[0])
        # entity_to_replace_with = np.random.randint(n_possibilities, size=data.shape[0])
        data = data.tolist()

        # use multiprocessing so that we can perform the triple corruption
        # in parallel
        with multiprocessing.Pool(self.workers) as pool:
            join = np.array(pool.map(self._corrupt, data))
        return join

    def benchmark(self, dataset, model, batch_log_frequency=10):
        dset = Dataset(dataset.kb,
                       dataset.batch_size,
                       self.smart_triple_corruption)

        total_correct = 0
        total_instances = 0
        total_pos_correct = 0
        total_neg_correct = 0
        per_relation_accuracy = np.zeros_like(self.threshold_values)

        for i, (positive_batch, negative_batch) in enumerate(dset.get_generator()):
            
            # first prepare the thresholds
            relationships = positive_batch[1]
            thresholds = self.threshold_values[relationships]

            # score the batches
            pos_scores = model.batch_score_triples(*positive_batch)
            neg_scores = model.batch_score_triples(*negative_batch)

            # get the classification for each triple
            pos_classification = pos_scores < thresholds
            neg_classification = neg_scores < thresholds

            # check with the true values:
            pos_correct = pos_classification == np.ones_like(thresholds)
            neg_correct = neg_classification == np.zeros_like(thresholds)
            
            # collect statistics:
            total_correct += np.sum(pos_correct) + np.sum(neg_correct)
            total_pos_correct += np.sum(pos_correct)
            total_neg_correct += np.sum(neg_correct)

            # TODO: implement per_relation_accuracy
            total_instances += pos_correct.shape[0]
            if i % batch_log_frequency == 0:
                print('Batch {:d} (combined average): acc {:.4f} | pos_acc {:.4f}  | neg_acc {:.4f}'.format(
                    i, total_correct/(2*total_instances),
                    total_pos_correct / total_instances,
                    total_neg_correct / total_instances))
        print('TripleClassificationTask Benchmarking Results')
        print('Total instances: ', total_instances)
        print('% instances correctly classified:', total_correct/(2*total_instances))
        print('% positive instances correctly: ', total_pos_correct / total_instances)
        print('% negative instances correctly: ', total_neg_correct / total_instances)
        return total_correct, total_pos_correct, total_neg_correct, total_instances
