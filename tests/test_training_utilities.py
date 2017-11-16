from embedKB.models.transe import TransE
import pytest
import numpy as np

batch_size = 32
n_entities=10
n_relationships=5
entity_embed_dim = 10
relationship_embed_dim = 12

framework = TransE(n_entities, entity_embed_dim, n_relationships)

class TestModel(object):
    @pytest.fixture(scope="module")
    def model(self):
        return framework
    def test_triple_score(self, model):
        assert model.score_triple(4, 4, 1)

    def test_train_on_batch(self, model):
        positive_data = (
            np.random.randint(n_entities, size=batch_size).reshape(-1, 1),
            np.random.randint(n_relationships, size=batch_size).reshape(-1, 1),
            np.random.randint(n_entities, size=batch_size).reshape(-1, 1)
        )

        negative_data = (
            positive_data[2],
            positive_data[1],
            positive_data[0]
        )

        model.create_objective()
        model.create_optimizer()
        model.create_summaries()
        objectives = []
        for i in range(10):
            objectives.append(framework.train_batch(positive_data, positive_data)[0])

        for i in range(1, len(objectives)):
            assert objectives[i-1] > objectives[i], 'Model did not train successfully (loss not decreasing)'