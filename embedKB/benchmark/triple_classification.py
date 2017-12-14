from .task import Task
import numpy as np
import logging
from embedKB.datatools import Dataset, SmartNegativeSampling, NegativeSampling
import multiprocessing
import itertools

MAX_ITERATES = 5000

class TripleClassificationTask(Task):
    def __init__(self,
                 dataset,
                 workers=1,
                 sampler='smart',
                 max_accuracy=1,
                 epsilon=0.01):
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
        self.unique_relations = unique_relations
        self.dataset = dataset
        self.workers = workers
        assert epsilon >= 0
        self.epsilon = epsilon
        assert max_accuracy <= 1
        self.max_accuracy = max_accuracy

        if sampler == 'smart':
            self.sns = SmartNegativeSampling(dataset.kb, workers)
        else:
            self.sns = NegativeSampling()

        self.sample = self.sns.sample

    def compute_threshold_values(self, model, dataset=None):
        print('Computing Thresholds.')
        dataset = self.dataset.all_data if not dataset else dataset.all_data
        nulls = 0
        unknown_relationships = []

        pos_scores = []
        neg_scores = []
        for r in self.unique_relations:
            try:
                # contains correct triples that satisfy the relation
                data_subset = dataset[np.where(dataset[:, 1] == r), :][0]

                if not len(data_subset) > 0:
                    unknown_relationships.append(r)
                    nulls += 1
                    pos_scores.append(np.array([]).reshape(-1, 1))
                    neg_scores.append(np.array([]).reshape(-1, 1))
                    continue

                negative_data_subset = self.sample(data_subset)
                
                assert np.all(negative_data_subset[:, 1] == data_subset[:, 1])

                per_triple_score = model.batch_score_triples(data_subset[:, 0],
                                                             data_subset[:, 1],
                                                             data_subset[:, 2])

                neg_triple_score = model.batch_score_triples(negative_data_subset[:, 0],
                                                             negative_data_subset[:, 1],
                                                             negative_data_subset[:, 2])

                
                pos_scores.append(per_triple_score)
                neg_scores.append(neg_triple_score)

            except Exception as e:
                print('An exception occured in relaitonship:',r)
                print('Exception was:',str(e))

        # concatenate all the scores we've got so far.
        pos_scores_concat = np.concatenate(pos_scores)
        neg_scores_concat = np.concatenate(neg_scores)

        # get the min and max scores
        min_score = min(pos_scores_concat.min(), neg_scores_concat.min())
        max_score = max(pos_scores_concat.max(), neg_scores_concat.max())
        mean_score = np.mean(pos_scores_concat) + np.mean(neg_scores_concat)

        # initialize arrays
        threshold_values = np.ones(len(self.unique_relations)) * min_score
        per_relation_accuracy = np.ones(len(self.unique_relations)) * -1

        score = min_score
        increments = 0.01

        # SAVE (TEMP)
        from collections import defaultdict
        acc_vs_threshold = defaultdict(lambda: {'accuracy':[], 'threshold':[]})

        while score <= max_score:
            for r in self.unique_relations:
                pos_scores_for_relation = pos_scores[r]
                neg_scores_for_relation = neg_scores[r]

                if len(pos_scores_for_relation) == 0:
                    threshold_values[r] = mean_score
                    continue

                mean_classification_acc = np.mean(
                                            np.concatenate(
                                                [pos_scores_for_relation <= score,
                                                 neg_scores_for_relation > score]))
                
                if mean_classification_acc > per_relation_accuracy[r]:
                    per_relation_accuracy[r] = mean_classification_acc
                    threshold_values[r] = score
                    acc_vs_threshold[r]['accuracy'].append(float(mean_classification_acc))
                    acc_vs_threshold[r]['threshold'].append(float(score))
            #endfor
            score += increments
        
        # # SAVE (TEMP)
        # import json
        # with open('./debugging/accvsthresh.json', 'w') as f:
        #     json.dump(acc_vs_threshold, f)
        
        if nulls > 0:
            logging.warn('There were {} relations with no thresholds. They were set to 0.'.format(nulls))
            print(unknown_relationships)

        self.threshold_values = threshold_values


    def benchmark(self, dataset, model, batch_log_frequency=10):

        total_correct = 0
        total_instances = 0
        total_pos_correct = 0
        total_neg_correct = 0
        per_relation_accuracy = np.zeros_like(self.threshold_values)

        for i, (positive_batch, negative_batch) in enumerate(dataset.get_generator()):
            
            # first prepare the thresholds
            relationships = positive_batch[1]
            thresholds = self.threshold_values[relationships]

            # score the batches
            pos_scores = model.batch_score_triples(*positive_batch)
            neg_scores = model.batch_score_triples(*negative_batch)

            # get the classification for each triple
            pos_classification = pos_scores < thresholds #positive classified as plausible
            neg_classification = neg_scores < thresholds #negative classified as plausible
            
            # check with the true values:
            pos_correct = pos_classification == np.ones_like(thresholds) # should be plausible
            neg_correct = neg_classification == np.zeros_like(thresholds) # should actually be implausible
            
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
        print('% positive instances classified as positive: ', total_pos_correct / total_instances)
        print('% negative instances classified as negative: ', total_neg_correct / total_instances)
        return total_correct, total_pos_correct, total_neg_correct, total_instances
