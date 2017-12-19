import os
import sys
# uncomment for running on local:
# sys.path.insert(0, os.path.abspath(".."))
import argparse
from embedKB.datatools import get_data
from embedKB.models import get_model


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', help="model to run", required=True)
parser.add_argument('-e', '--entity_dim', help="model to run", default=50)
parser.add_argument('-r', '--relation_dim', help="model to run", default=50)
parser.add_argument('-data', '--data', help="location of the data", default='../data/Release/')
parser.add_argument('-reg', '--reg_weight', help="regularization weight", default=0.001)
parser.add_argument('-lr', '--init_learning_rate', help="initial learning rate", default=0.01)
parser.add_argument('-gpu', help="ID of GPU to execute on", default='0')
parser.add_argument('-n_epochs', '--n_epochs', help="number of epochs", default=100)
parser.add_argument('-batch_log_freq', '--batch_log_freq', help="logging frequency", default=10000)
parser.add_argument('-batch_size', '--batch_size', default=32)
args = parser.parse_args()

#specify GPU ID on target machine
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# get data and initialise model
kb_train, dset_train, kb_val, dset_val, kb_test, dset_test = get_data(args.data, args.batch_size)
model = get_model(args.model_name, kb_train.n_entities, kb_train.n_relations, args.entity_dim, args.relation_dim,
                  args.reg_weight, args.init_learning_rate)
model.create_summaries()

# create output directory
output_dir = './' + args.model_name
create_folder = lambda f: [os.makedirs(os.path.join('./', f)) if not os.path.exists(os.path.join('./', f)) else False]
create_folder(output_dir)

# save triples and mappings
kb_train.save_converted_triples(output_dir + '/triples.npy')
kb_train.save_mappings_to_json(output_dir + '/')

# train model
model.train(dset_train,
				epochs=args.n_epochs,
				val_data=dset_val,
				batch_log_frequency=args.batch_log_freq,
				logging_directory= output_dir + '/checkpoints')
model.evaluate(dset_train, name='training/start')
model.evaluate(dset_val, name='val/start')

# save final model
model.save_model(output_dir + '/' + args.model_name)