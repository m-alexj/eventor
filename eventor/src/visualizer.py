import torch
import gensim
import argparse
import logging
import pickle
from bio.load import load
import numpy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import sys
from visual.visualize import visualize_one_change, visualize_backprop_graph
import os
import shutil
from model.trainer import accuracy
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

logger = logging.getLogger("EventorVisualizer")
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Event extractor dataset visualizer')
parser.add_argument('-wv', type=str,
                        help='Word vector file', dest="wordvectors")
parser.add_argument('-model', type=str,
                    help='Model file (index file is model file + "_index"', dest="model")
parser.add_argument('-eval', type=str,
                    help='Evaluation file', dest="eval")
parser.add_argument('-data', type=str,
                    help='Data set', dest="data")
parser.add_argument('-only', type=str, default="",
                    help='If set, will only produce output for data points with this sentence ID', dest="only")
parser.add_argument('-limit', type=int, default=22,
                    help='Maximum distance of nodes in backpropagation graph to root node (=loss) (default 10)', dest='backproplimit')

# the following lines are only for test purposes; they should be deleted for the first code publication
evalfile = "/hits/fast/nlp/judeaax/event2/eventor.eval.dev.mini.txt"
datafile = "/hits/fast/nlp/judeaax/event2/eventor.dev.simple.filtered.mini.txt"
wordvectorfile = "/data/nlp/judeaax/event2/cyb.txt"
modelfile = "model.obj"
args = "-wv %s -eval %s -data %s -model %s -only 1467CNN_CF_20030303.1900.02" % (wordvectorfile, evalfile, datafile, modelfile)

arguments = sys.argv[1:] if len(sys.argv) > 1 else args.split(" ")

parsed_arguments = parser.parse_args(arguments)

logger.info(parsed_arguments)

logger.info("Loading model {0} and index file {0}_index".format(parsed_arguments.model))
model = torch.load(parsed_arguments.model, map_location=lambda storage, loc: storage)
with open(parsed_arguments.model + "_index", "rb") as f:
    model_index = pickle.load(f)

logger.info("Loading word vectors")
w2v = gensim.models.KeyedVectors.load_word2vec_format(parsed_arguments.wordvectors, binary=parsed_arguments.wordvectors.endswith("bin"))
logger.info("Loading data")
dataset, _, _ = load(parsed_arguments.data, parsed_arguments.eval, model_index, True)

model.eval()

shutil.rmtree("trees", ignore_errors=True)
os.makedirs("trees/visualizer", exist_ok=True)

path = "trees/visualizer"

model.cudaFlag = False
model.child_sum_tree.cudaFlag = False

for point in dataset:
    if len(parsed_arguments.only) > 0 and point[0].sid != parsed_arguments.only:
        continue
    losses, trees = model(point, None)

    matrix = point[0].matrix[-1]
    visualize_one_change(matrix, point[0], model_index, "%s/%s.%s.%s"
                         % (path, point[0].sid, point[0].word,
                            str(point[0].begin)), addPdfPostfix=True)
    graph = visualize_backprop_graph(losses[0][0], {x: y for (x, y) in
                                                    model.named_parameters()}, limit=parsed_arguments.backproplimit)
#     graph.save("%s.%s.%s.backprop" % (point[0].sid, point[0].word,
#                                         str(point[0].begin)), path)
    graph.format = 'svg'
    graph.render("%s.%s.%s.backprop" % (point[0].sid, point[0].word,
                                        str(point[0].begin)), path,
                cleanup=True)
accuracy("", dataset, model_index, 0, True)


