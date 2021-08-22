'''
Created on Jul 28, 2017

@author: judeaax
'''

import torch
import gensim
import argparse
import logging
import pickle
from bio.load import load
import numpy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

logger = logging.getLogger("EventorModelCompare")
logging.basicConfig(level=logging.INFO)

default_train_eval = "/data/nlp/judeaax/event2/event.mention.graphs/enhanced_pp/event.mini.eval.train.txt"
default_dev_eval = "/data/nlp/judeaax/event2/event.mention.graphs/enhanced_pp/event.eval.dev.txt"

  
default_train = "/data/nlp/judeaax/event2/event.mention.graphs/enhanced_pp/graph.mini.onlyverbtrim.train.txt"          
default_dev = "/data/nlp/judeaax/event2/event.mention.graphs/enhanced_pp/graph.onlyverbtrim.dev.txt"


parser = argparse.ArgumentParser(description='Event extractor model comparator')
parser.add_argument('-wv', default="/data/nlp/judeaax/event2/cyb.txt", type=str,
                        help='Word vector file', dest="wordvectors")
parser.add_argument('-evaltrain', default=default_train_eval, type=str,
                    help='Training data evaluation file', dest="evaltrain")
parser.add_argument('-evaldev', default=default_dev_eval, type=str,
                    help='Development data evaluation file', dest="evaldev")

parser.add_argument('-train', default=default_train, type=str,
                    help='Training data', dest="trainfile")
parser.add_argument('-dev', default=default_dev, type=str,
                        help='Development data', dest="devfile")
parser.add_argument('-file1', type=str,
                        help='First model file')

parser.add_argument('-file2', type=str,
                        help='Second model file')

args = parser.parse_args()

file1 = args.file1
file2 = args.file2

logger.info("Loading %s" % file1)
model1 = torch.load(file1)
with open(file1 + "_index", "rb") as f:
    model1_index = pickle.load(f)
logger.info("Loading %s" % file2)
model2 = torch.load(file2)
with open(file2 + "_index", "rb") as f:
    model2_index = pickle.load(f)
logger.info("Loading word vectors")
w2v = gensim.models.KeyedVectors.load_word2vec_format(args.wordvectors, binary=args.wordvectors.endswith("bin"))
logger.info("Loading data")
train_dataset1, _, _ = load(args.trainfile, args.evaltrain, model1_index, True)
train_dataset2, _, _ = load(args.trainfile, args.evaltrain, model2_index, True)
dev_dataset1, _, _ = load(args.devfile, args.evaldev, model1_index)
dev_dataset2, _, _ = load(args.devfile, args.evaldev, model2_index)

def visualize_diff(t1, t2, index1, index2, name):
    assert len(t1) == len(t2)
    matrix = numpy.zeros((len(t1), len(t2)))
    for i in range(len(t1)):
        for j in range(len(t2)):
            
            if i == j:
                assert t1[i].word == t2[j].word and t1[i].id == t2[j].id
            
            h1 = t1[i].h.data.cpu().numpy().reshape(1, -1)
            h2 = t2[j].h.data.cpu().numpy().reshape(1, -1)
            sim = cosine_similarity(h1, h2)
            matrix[i, j] = sim
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')

    heatmap = ax.pcolor(matrix, cmap="hot")
    
    # legend
    fig.colorbar(heatmap)
    
    # axes
    labels = [(x.word, x.id) for x in t1]
    
    labels = sorted(labels, key=lambda l:l[1])
    ax.set_xticks([x[1] + 0.5 for x in labels])
    ax.set_yticks([x[1] + 0.5 for x in labels])
    ax.set_xticklabels([x[0] for x in labels], rotation=90)
    ax.set_yticklabels([x[0] for x in labels])
    plt.savefig("%s.matrix.pdf" % name)
    
    g1 = [x for x in t1 if x.is_trigger][0].toGraph(index1)
    g1.write_svg("%s.tree1.svg" % name)
    
    g2 = [x for x in t2 if x.is_trigger][0].toGraph(index2)
    g2.write_svg("%s.tree2.svg" % name)



def test(points, model1, model2, index1, index2, ground=False):
    model1.eval()
    model2.eval()
    for point in points:
        _, trees1 = model1(point[0], None)
        _, trees2 = model2(point[1], None)
        assert trees1[0].sid == trees2[0].sid and trees1[0].word == trees2[0].word
        visualize_diff(trees1, trees2, index1, index2, "compare%s.%s.%s" % (".ground" if ground else "", trees1[0].sid, trees1[0].word))

sid = "1358AFP_ENG_20030327.0224"
points1 = [x for x in dev_dataset1 if x[0].sid == sid]
points2 = [x for x in dev_dataset2 if x[0].sid == sid]
points = [x for x in zip(points1, points2)]
test(points, model1, model2, model1_index, model2_index)
test([x for x in zip([train_dataset1[0]], [train_dataset2[0]])], model1, model2, model1_index, model2_index, ground=True)
