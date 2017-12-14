from embedKB.benchmark import TripleClassificationTask, EntityPredictionTask
from embedKB.datatools import Dataset, KnowledgeBase
import pytest
import numpy as np

kb = KnowledgeBase.load_from_raw_data('./tests/test_kb.txt')
kb.convert_triples()
dset = Dataset(kb, batch_size=5)
tct = TripleClassificationTask(dset, workers=2)

class FakeModel(object):
    def batch_score_triples(self, a, b, c):
        b = b.reshape(-1)
        scores = np.where(b == 0,
                 np.random.uniform(low=0, high=0.5, size=b.shape[0]), 
                 np.random.uniform(low=0.5, high=1, size=b.shape[0]))
        return scores.reshape(-1, 1)

class GoodModel(object):
    def __init__(self, kb):
        self.kb = kb
    def batch_score_triples(self, a, b, c):
        all_triples = set([tuple(i) for i in self.kb.triples.tolist()])
        print(all_triples)
        scores = np.random.uniform(low=0.1, high=0.9, size=b.shape[0]).reshape(-1)
        for i, triple in enumerate(zip(a,b,c)):
            if tuple(triple) in all_triples:
                scores[i] = 0
        return scores.reshape(-1, 1)

def test_creation_of_list():
    assert tct.sns.possible_heads[kb.relations['played_by']] == [kb.entities['Games'], kb.entities['GTA']]
    assert set(tct.sns.possible_tails[kb.relations['type_of']]) == set([kb.entities['embedding'], kb.entities['children'], kb.entities['machine_learning']])

def test_corrupt():
    original_triple = (kb.entities['TransE'], kb.relations['type_of'], kb.entities['embedding'])
    
    print(original_triple)
    corrupted_triple = tct.sns._corrupt(original_triple)

    print(corrupted_triple)
    assert original_triple != corrupted_triple

    for i in range(30):
        assert tct.sns._corrupt(original_triple)[2] != kb.entities['adults']


def test_compute_threshold_values():
    model = FakeModel()

    tct.compute_threshold_values(model)
    assert tct.threshold_values[0] < 0.5
    assert tct.threshold_values[1] > 0.5
    print(tct.threshold_values)
    # run the benchmark 10 times (since it is random)
    # we should find that the positively classified instances
    # should be >= 3 at least 70% of the time
    # TODO need to figure this out:
    # well_done = 0
    # for i in range(10):
    #     _, total_pos_correct, _, _ = tct.benchmark(dset , model)
    #     well_done += total_pos_correct >= 3
    # assert well_done >= 7


def test_entity_prediction_score_triple():
    model = GoodModel(kb)
    ept = EntityPredictionTask(kb, top_k=1)
    ept.model = model
    score = ept.score_triple((kb.entities['GTA'], kb.relations['played_by'], kb.entities['adults']),  entity_subsample=kb.n_entities)
    assert score[0] == 2

    model = FakeModel()
    ept = EntityPredictionTask(kb, top_k=1)
    ept.model = model
    score = ept.score_triple((kb.entities['GTA'], kb.relations['played_by'], kb.entities['adults']),  entity_subsample=kb.n_entities)
    # @TODO: this test fails sometimes because of something stochastic going on. Need to investigate.
    assert score[0] == 0


    model = FakeModel()
    ept = EntityPredictionTask(kb, top_k=10)
    ept.model = model
    score = ept.score_triple((kb.entities['GTA'], kb.relations['played_by'], kb.entities['adults']), entity_subsample=kb.n_entities)
    assert score[0] == 2