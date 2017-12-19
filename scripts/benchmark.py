import os
import sys
# uncomment for running on local:
# sys.path.insert(0, os.path.abspath(".."))
import argparse
from embedKB.models import get_model
from embedKB.datatools import load_saved_data
from embedKB.benchmark import TripleClassificationTask, EntityPredictionTask

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', help="model to run", required=True)
parser.add_argument('-e', '--entity_dim', help="model to run", default=50)
parser.add_argument('-r', '--relation_dim', help="model to run", default=50)
parser.add_argument('-f', '--folder', help="location of the model and kb", required=True)
parser.add_argument('-gpu', help="ID of GPU to execute on", default='0')
parser.add_argument('-t', '--task', help="the task to benchmark upon (ept or tct)", required=True)

args = parser.parse_args()
print('Running benchmark code for: {0} on GPU {1}...'.format(args.model_name, args.gpu))

# specify GPU ID on target machine
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# get saved data
kb_train, dset_train, kb_val, dset_val, kb_test, dset_test = \
    load_saved_data(args.folder)

# create model
model = get_model(args.model_name,
                  kb_train.n_entities,
                  kb_train.n_relations,
                  args.entity_dim,
                  args.relation_dim)
model.create_summaries()
model.load_model(args.folder + '/' + args.model_name)

if (args.task == 'tct'):
    tct = TripleClassificationTask(dset_train, workers=12)
    tct.compute_threshold_values(model, dset_val)

    print('Benchmarking on training set...')
    tct.benchmark(dset_train, model, batch_log_frequency=100000)

    print('Benchmarking on validation set...')
    tct.benchmark(dset_val, model, batch_log_frequency=100000)

    print('Benchmarking on test set...')
    tct.benchmark(dset_test, model, batch_log_frequency=100000)
elif args.task == 'ept':
    ept = EntityPredictionTask(kb_train, workers=12)

    print('Benchmarking on training set...')
    ept.benchmark(dset_train, model, batch_log_frequency=100000)

    print('Benchmarking on validation set...')
    ept.benchmark(dset_val, model, batch_log_frequency=100000)

    print('Benchmarking on test set...')
    ept.benchmark(dset_test, model, batch_log_frequency=100000)
else:
    raise KeyError('Unknown benchmarking, we currently support "ept" and "tct"')
