from embedKB.benchmark import TripleClassificationTask
from embedKB.datatools import Dataset, KnowledgeBase
import pytest
import numpy as np

kb = KnowledgeBase.load_from_raw_data('./tests/test_kb.txt')
kb.convert_triples()
dset = Dataset(kb, batch_size=5)
tct = TripleClassificationTask(dset)

def test_creation_of_list():
    assert tct.possible_heads[kb.relations['played_by']] == [kb.entities['Games'], kb.entities['GTA']]
    assert set(tct.possible_tails[kb.relations['type_of']]) == set([kb.entities['embedding'], kb.entities['children'], kb.entities['machine_learning']])

def test_corrupt():
    original_triple = (kb.entities['TransE'], kb.relations['type_of'], kb.entities['embedding'])
    
    print(original_triple)
    corrupted_triple = tct._corrupt(original_triple)

    print(corrupted_triple)
    assert original_triple != corrupted_triple

    for i in range(30):
        assert tct._corrupt(original_triple)[2] != kb.entities['adults']

