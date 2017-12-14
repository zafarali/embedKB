from .task import Task
import numpy as np
import random
import multiprocessing

random.seed(1)
np.random.seed(1)

def reprepare_data(data):
    heads, relationships, tails = data
    heads = heads.reshape(-1).tolist() 
    relationships = relationships.reshape(-1).tolist() 
    tails = tails.reshape(-1).tolist() 
    return zip(heads, relationships, tails)

class EntityPredictionTask(Task):
    """
    Implements the entity prediction task as described in 
    
    Bordes, Antoine, et al. 
    "Translating embeddings for modeling multi-relational data." 
    Advances in neural information processing systems. 2013.
    https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
    
    and 
    
    Bordes, Antoine, et al. 
    "Learning Structured Embeddings of Knowledge Bases." 
    AAAI. Vol. 6. No. 1. 2011.
    https://ronan.collobert.com/pub/matos/2011_knowbases_aaai.pdf
    """
    def __init__(self, kb, workers=1, filtered=False, top_k=10):
        """
        :param kb: the knowledge base
        :param workers: the number of multiprocessing threads to launch
        :param filtered: runs the "filtered" version of the task, 
                         where we remove true corruptions that are in the KB
                         from the test set. 
        :param top_k: ranks and returns the top_k entities
        """
        self.entities = np.arange(kb.n_entities)
        self.kb = kb
        self.top_k = top_k
        self.workers = workers
        self.filtered = filtered
        if self.filtered:
            # super efficent storage
            self.kb_triples = set(map(tuple, self.kb.triples))
        self.MODE = 'filtered' if self.filtered else 'RAW'

    def remove_valid_triples(self, triples_array, correct_triple):
        triples = set(map(tuple, triples_array))

        # remove triples in the kb from the list:
        triples = triples.difference(self.kb_triples)
        triples.add(correct_triple)
        return np.array(list(triples))

    def benchmark(self, dataset, model, entity_subsample=256, batch_subsample=16, batch_log_frequency=10):
        """
        :param entity_subsample: subsample of the entities to pick and evaluate on
        :param batch_subsample: the subsample of the batch to pick and evaluate on
        """
        self.model = model
        subsample = min(dataset.batch_size, batch_subsample)
        # pool = multiprocessing.Pool(self.workers)
        results = []
        print('MODE: {}'.format(self.MODE))
        print('Starting benchmarking...')
        total_correct = 0
        total_rank = 0
        total_instances = 0
        for i, (data, _) in enumerate(dataset.get_generator()):
            data = list(reprepare_data(data))
            to_evaluate = random.sample(data, subsample)
            for triple in to_evaluate:
                corr, rank = self.score_triple(triple, entity_subsample)
                total_rank += rank
                total_correct += corr
                total_instances += 2
            if i % batch_log_frequency == 0:
                print('benchmarked {} batches.'.format(i))
                print('\t correct instances:{}/{} = {}'.format(total_correct, total_instances, total_correct/total_instances))
                print('\t mean rank: {}'.format(total_rank / total_instances))
            # results.append(pool.map_async(self.score_triple, to_evaluate))

        # pool.close()
        # pool.join()


        # print('Collecting results...')
        # for batch_result in results:
        #     result = batch_result.get()
        #     total_correct += sum(result)
        #     total_instances += 2*len(result)

        print('Entity Prediction Task ({}) results:'.format(self.MODE))
        print('number of instances:', total_instances)
        print('k=',self.top_k)
        print('number of instances where head/tail was in top_k:', total_correct)
        print('Accuracy:', total_correct/total_instances)
        print('Mean Rank:', total_rank/total_instances)
        return total_correct/total_instances, total_rank/total_instances

    def score_triple(self, triple, entity_subsample):
        """
        Scores a single triple against a sample of the entities.
        """
        head_replaced = np.repeat(np.array(triple).reshape(-1, 3), entity_subsample+1, axis=0)
        tail_replaced = np.repeat(np.array(triple).reshape(-1, 3), entity_subsample+1, axis=0)
        
        head_replaced[1:, 0] = np.random.choice(self.entities, entity_subsample)
        tail_replaced[1:, 2] = np.random.choice(self.entities, entity_subsample)
        
        if self.filtered:
            head_replaced = self.remove_valid_triples(head_replaced, triple)
            tail_replaced = self.remove_valid_triples(tail_replaced, triple)
        else:
            np.random.shuffle(head_replaced)
            np.random.shuffle(tail_replaced)

        head_replaced_scores = self.model.batch_score_triples(head_replaced[:, 0],
                                                         head_replaced[:, 1],
                                                         head_replaced[:, 2])

        tail_replaced_scores = self.model.batch_score_triples(tail_replaced[:, 0],
                                                         tail_replaced[:, 1],
                                                         tail_replaced[:, 2])

        # sort the indices by score
        heads_idx_sorted = np.argsort(head_replaced_scores.reshape(-1))
        tails_idx_sorted = np.argsort(tail_replaced_scores.reshape(-1))

        # obtain the actual heads
        heads_sorted = head_replaced[heads_idx_sorted, 0]

        # obtain the rank of the true triple
        head_rank = np.where(heads_sorted == triple[0])[0].tolist()
        head_rank = head_rank[0] if len(head_rank) > 0 else head_replaced.shape[0]

        # do the same for tails
        tails_sorted = tail_replaced[tails_idx_sorted, 2]
        tail_rank = np.where(tails_sorted == triple[2])[0].tolist()
        tail_rank = tail_rank[0] if len(tail_rank) > 0 else tail_replaced.shape[0]

        # obtain the top k heads/tails
        top_k_heads = heads_idx_sorted[:self.top_k]
        top_k_tails = tails_idx_sorted[:self.top_k]
        
        best_fit_heads = head_replaced[top_k_heads, 0].tolist()
        best_fit_tails = tail_replaced[top_k_tails, 2].tolist()

        return int(triple[0] in best_fit_heads) + int(triple[2] in best_fit_tails), (head_rank + tail_rank)

