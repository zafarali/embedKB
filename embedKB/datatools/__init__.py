from .knowledgebase import KnowledgeBase
from .dataset import Dataset
from .dataset import SmartNegativeSampling, NegativeSampling

def get_data(data_path, batch_size):
    train_path = data_path + 'train.txt'
    valid_path = data_path + 'valid.txt'
    test_path = data_path + 'test.txt'

    # load knowledge base of train data
    kb_train = KnowledgeBase.load_from_raw_data(train_path)
    kb_train.convert_triples()
    dset_train = Dataset(kb_train, batch_size=batch_size)

    # derive a knowledge base of validation data
    kb_val = KnowledgeBase.derive_from(kb_train)
    kb_val.load_raw_triples(valid_path)
    kb_val.convert_triples()
    dset_val = Dataset(kb_val, batch_size=batch_size)

    # derive a knowledge base of testing data
    kb_test = KnowledgeBase.derive_from(kb_train)
    kb_test.load_raw_triples(test_path)
    kb_test.convert_triples()
    dset_test = Dataset(kb_test, batch_size=batch_size)

    return kb_train, dset_train, kb_val, dset_val, kb_test, dset_test


def load_saved_data(folder):

    #currently only the kb for the training data was saved.
    kb_train = KnowledgeBase()
    kb_train.load_converted_triples(folder + '/triples.npy')
    kb_train.load_mappings_from_json(folder + '/entity2id.json',
                               folder + '/relation2id.json')

    # have to do this temporarily. should be fixed to be more general
    batch_size = 32
    data_path = '../data/Release/'
    valid_path = data_path + 'valid.txt'
    test_path = data_path + 'test.txt'

    dset_train = Dataset(kb_train, batch_size=batch_size)

    # derive a knowledge base of validation data
    kb_val = KnowledgeBase.derive_from(kb_train)
    kb_val.load_raw_triples(valid_path)
    kb_val.convert_triples()
    dset_val = Dataset(kb_val, batch_size=batch_size)

    # derive a knowledge base of testing data
    kb_test = KnowledgeBase.derive_from(kb_train)
    kb_test.load_raw_triples(test_path)
    kb_test.convert_triples()
    dset_test = Dataset(kb_test, batch_size=batch_size)

    return kb_train, dset_train, kb_val, dset_val, kb_test, dset_test
