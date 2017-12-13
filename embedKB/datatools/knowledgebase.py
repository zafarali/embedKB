import numpy as np
import os
import json
import logging
from collections import namedtuple

Triple = namedtuple('Triple', ['head_entity', 'relationship', 'tail_entity']) 
TripleConverted = namedtuple('TripleConverted', ['head_id', 'relation_id', 'tail_id'])

class KnowledgeBase(object):
    _converted_triples = False
    def __init__(self,
                 entities=None,
                 relations=None,
                 triples=None):
        """
        A knolwedge base object.
        :param entities: a dict with entities in the KB and mapping to an id
        :param relations: a dict with relations in the KB and mapping to an id
        :param triples: a list of the triples in the KB.
        """
        self.entities = entities
        self.relations = relations
        self.triples = triples

    @property
    def n_triples(self):
        if self._converted_triples:
            return self.triples.shape[0]
        else:
            return len(self.triples) if self.triples else 0
    
    @property
    def n_entities(self):
        return len(self.entities) if self.entities else 0
    
    @property
    def n_relations(self):
        return len(self.relations) if self.relations else 0

    def convert_triples(self, verbose_errors=False, ignore_relationships=[]):
        """
        Converts the triples to a numpy format suitable for consumption
        Note after this is done, you cannot load more triples.
        """
        assert not self._converted_triples, 'Triples already converted!'
        converted_triples = []
        errors = []
        for triple in self.triples:
            # skip some relationships:
            if triple.relationship in ignore_relationships:
                continue
            try:
                converted_triples.append(TripleConverted(self.entities[triple.head_entity],
                                                         self.relations[triple.relationship],
                                                         self.entities[triple.tail_entity]))
            except KeyError as e:
                errors.append(triple)

        if len(errors) > 0:
            logging.warn('{} triples had a relationship and/or entity missing from the vocab'.format(len(errors)))
            if verbose_errors:
                print(errors)

        self.triples = np.array(converted_triples)
        self._converted_triples = True

    def load_converted_triples(self, path_to_data):
        """
        Loads triples from a numpy file
        """
        assert self.triples is None, 'You already have triples in the KB!'
        self.triples = np.load(path_to_data)
        self._converted_triples = True

    def load_raw_triples(self, path_to_data):
        """
        loads triples from a txt file
        """
        assert not self._converted_triples, 'You have already converted triples. Cannot load new ones'
        if self.n_triples > 0:
            logging.warn('Already have triples. You are appending to it.')
        triples = self.triples if self.triples else []
        with open(path_to_data, 'r') as f:
            for line in f:
                head, relationship, tail = line.strip().split('\t')
                triples.append(Triple(head, relationship, tail))
        self.triples = triples

    def save_converted_triples(self, path):
        """
        Saves the converted triples
        """
        assert self._converted_triples, 'You need to have triples converted to save them.'
        np.save(os.path.join(path),
                np.array(self.triples,
                dtype=np.int32))

    @staticmethod
    def derive_from(kb):
        """
        Derives a new knowledgebase with entities 
        and relations from the argument
        :param kb: a Knowledgebase
        """
        return KnowledgeBase(kb.entities, kb.relations)

    @staticmethod
    def load_from_raw_data(path_to_data):
        """
        Loads data from a txt file representing a KB
        """
        # load the entities and ids
        entities = set()
        relations = set()
        triples = []
        with open(path_to_data, 'r') as f:
            for line in f:
                head, relationship, tail = line.strip().split('\t')
                entities.update([head])
                entities.update([tail])
                relations.update([relationship])
                triples.append(Triple(head, relationship, tail))

        entity2id = { entity:i for i, entity in enumerate(entities) }
        relation2id = { relation:i for i, relation in enumerate(relations) }

        return KnowledgeBase(entity2id, relation2id, triples)

    def save_mappings_to_json(self, directory):
        with open(os.path.join(directory, 'entity2id.json'), 'w') as f:
            json.dump(self.entities, f, indent=4) 
        with open(os.path.join(directory, 'relation2id.json'), 'w') as f:
            json.dump(self.relations, f, indent=4)

    def load_mappings_from_json(self, path_to_entites, path_to_relations):
        with open(path_to_entites, 'r') as f:
            self.entities = json.load(f)
        with open(path_to_relations, 'r') as f:
            self.relations = json.load(f)
