import torch
import gensim
import argparse
import logging
import pickle
from bio.load import load
import sys
import os
import shutil
from model.trainer import accuracy, set_to_gold, set_to_null, \
    clear_tree_variables, flatten_params
from visual import visualize
import webbrowser
from sklearn.metrics import classification
from tabulate import tabulate
from visual.visualize import visualize_dependencies
from posix import mkdir
from itertools import groupby


logger = logging.getLogger("Eventor.Decoder")
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Event extractor dataset visualizer')

parser.add_argument('-model', type=str,
                    help='Model file (index file is model file + "_index"', dest="model")
parser.add_argument('-eval', type=str,
                    help='Evaluation file', dest="eval")
parser.add_argument('-data', type=str,
                    help='Data set', dest="data")
parser.add_argument('-msuf', type=str, default="",
                    help='Model suffix (empty by default)', dest="msuf")
parser.add_argument('-vsuf', type=str, default="",
                    help='Visualization suffix (empty by default)', dest="vsuf")


# the following lines are only for test purposes; they should be deleted for the first code publication
evalfile = "/data/nlp/judeaax/event2/eventor.eval.train.one.txt"
datafile = "/data/nlp/judeaax/event2/eventor.train.one.simple.carved.no_it.txt"
modelfile = "model.obj"
args = "-eval %s -data %s -model %s -msuf .avg" % (evalfile, datafile, modelfile)

arguments = sys.argv[1:] if len(sys.argv) > 1 else args.split(" ")

parsed_arguments = parser.parse_args(arguments)

logger.info(parsed_arguments)


def compute_confusion_matrix(index, gold, predicted, reverse_index_f, title):
    # and compute/print the confusion matrix
    indices = [x for x in range(len(index))]
    CM = classification.confusion_matrix(gold, predicted, labels=indices)
    names = [reverse_index_f(x) for x in indices]
    rows = [[title] + names]
    rows += [[names[x]] + [str(y) for y in CM[x]] for x in range(len(names))]
    final_string = tabulate(rows, headers="firstrow")
    print(final_string)
    return final_string


is_cuda = torch.cuda.is_available()

logger.info("Loading model {0} and index file {0}_index".format(parsed_arguments.model + parsed_arguments.msuf))
if is_cuda:
    model = torch.load(parsed_arguments.model + parsed_arguments.msuf)
else:
    model = torch.load(parsed_arguments.model + parsed_arguments.msuf, map_location=lambda storage, loc: storage)
with open(parsed_arguments.model + "_index", "rb") as f:
    model_index = pickle.load(f)

# logger.info("Loading word vectors")
# w2v = gensim.models.KeyedVectors.load_word2vec_format(parsed_arguments.wordvectors, binary=parsed_arguments.wordvectors.endswith("bin"))
logger.info("Loading data")
model.index.freeze()
dataset, _, _ = load(parsed_arguments.data, parsed_arguments.eval, model_index)

model.eval()

shutil.rmtree("trees_%s" % parsed_arguments.msuf, ignore_errors=True)

model.cudaFlag = is_cuda
# model.child_sum_tree.sentence_lstm.flatten_parameters()

accuracy("perf", set_to_gold(dataset), model_index, -1, print_numbers=True)
set_to_null(dataset, model_index)

# we need this to display a progress percentage
i = 0.0
for point in dataset:
    loss = model(point, None)
    i += 1
    print('\r%.2f' % (i / len(dataset)), end='')

# evaluate
results = accuracy("", dataset, model_index, 0, print_numbers=True)
# and pack precision, recall, and f1 for triggers and arguments into a new tuple
prf1 = results[:-3]
# produce an HTML visualization
doc = visualize.htmlout(dataset, model_index, prf1, parsed_arguments.vsuf)
# and write it to disk
with open('output_%s.html' % parsed_arguments.vsuf, 'w') as f:
    f.write(str(doc))
# create the prediction output directory
os.makedirs('output_predictions', exist_ok=True)
# group the data by document ID
grouped = groupby(dataset, lambda x: x.did)
# then, for each document, we construct one file
for did, graphs in grouped:
    graphs = list(graphs)
    # assert that all graphs have the same document ID and the same genre
    genre = graphs[0].genre
    path = "output_predictions/%s" % (did)
    logger.info("Writing " + path)
    with open(path, 'w') as f:
        for graph in graphs:
            f.write(graph.getEventPredictions())
            f.write('\n')
