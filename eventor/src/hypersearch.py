'''
Created on Mar 7, 2018

@author: judeaax
'''

from hyperopt import fmin, tpe
from torch.optim.adam import Adam
from hyperopt import hp
from model.model import MasterModel
import argparse
from bio.load import load
import gensim
from model.index import Index
import logging
import torch
from model.trainer import accuracy
from torch import nn

logger = logging.getLogger("Eventor")
logging.basicConfig(level=logging.DEBUG)

infix = '.one'
default_train = "/data/nlp/judeaax/event2/eventor.train%s.simple.carved.no_it.txt" % (infix)
default_train_eval = "/data/nlp/judeaax/event2/eventor.eval.train%s.txt" % (infix)
default_dev = default_train  # "/data/nlp/judeaax/event2/eventor.test.simple.carved.no_it.txt"
default_dev_eval = default_train_eval  # "/data/nlp/judeaax/event2/eventor.eval.test.txt"

parser = argparse.ArgumentParser(description='Hyperparameter optimizer')
parser.add_argument('-wv', default="/data/nlp/judeaax/event2/cyb.txt", type=str,
                        help='Word vector file', dest="wv")
parser.add_argument('-train', default=default_train, type=str,
                    help='Training data', dest="trainfile")
parser.add_argument('-dev', default=default_dev, type=str,
                    help='Development data', dest="devfile")
parser.add_argument('-evaltrain', default=default_train_eval, type=str,
                    help='Training data evaluation file', dest="evaltrain")
parser.add_argument('-evaldev', default=default_dev_eval, type=str,
                    help='Development data evaluation file', dest="evaldev")


def fill_embeddings(model_embeddings, embedding_vectors, N, vocab_size, index):
        '''
        Fills the embeddings
        :param model_embeddings:
        :param embedding_vectors:
        :param N:
        '''
        wemb = torch.zeros(vocab_size, N).normal_(-0.05, 0.05)
        word_embedding_misses = 0
        for word, idx in index.word_index.items():
            if word in embedding_vectors:
                wemb[idx] = torch.FloatTensor(embedding_vectors[word].astype("float64"))
            elif idx > 0:
                word_embedding_misses += 1
            else:
                wemb[idx] = torch.FloatTensor(N).zero_()
        # plug these into embedding matrix inside model
        if torch.cuda.is_available():
            wemb = wemb.cuda()
        model_embeddings.state_dict()['weight'].copy_(wemb)


def objective(args):
    lr, h, aux, wd, d, b, nap = args
    print([x for x in zip(['lr', 'h', 'aux', 'wd', 'd', 'b', 'nap'], args)])

    logging.getLogger('Eventor.model').setLevel(logging.INFO)
    model = MasterModel(N_word_embeddings, int(aux), int(h), int(b), float(d), float(nap), index, torch.cuda.is_available())
    model.index = index
    model.word_embeddings = nn.Embedding(vocab_size, N_word_embeddings, padding_idx=0)
    if torch.cuda.is_available():
        model.word_embeddings = model.word_embeddings.cuda()
    fill_embeddings(model.word_embeddings, w2v, N_word_embeddings, vocab_size, index)
    model.word_embeddings.weight.requires_grad = True
    if torch.cuda.is_available():
        model = model.cuda()
    # create the optimizer
    optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.optimizer = optim
    best_f1 = float('-inf')
    best_values = None
    # train for 20 epochs
    for epoch in range(15):
        print('\r%d' % (epoch + 1), end='')
        model.train()
        for i in range(len(train_dataset)):
            graph = train_dataset[i]
            model(graph, epoch + 1)
            graph.reset()
        model(None, epoch + 1)
        model.eval()
        for j in range(len(dev_dataset)):
            graph = dev_dataset[j]
            model(graph, epoch + 1)
            graph.reset()
        model(None, epoch + 1)
        p_triggers, r_triggers, f1_triggers, p_arguments, r_arguments, f1_arguments, numbers_trig, numbers_arg, _ = accuracy("", dev_dataset, index, 0, None, print_numbers=False)
        if (f1_triggers + f1_arguments) / 2 > best_f1:
            best_f1 = (f1_triggers + f1_arguments) / 2
            best_values = (f1_triggers, f1_arguments)
    print()
    print(best_f1, best_values)
    del model
    return -best_f1


def define_search_space():

    # we need to tune the learning rate
    lr = hp.uniform('lr', 0.000001, 0.01)
    # we need to tune the hidden dimensions
    h = hp.quniform('h', 50, 500, 1)
    # we need to tune the aux dimensions
    aux = hp.quniform('aux', 5, 200, 1)
    # we need to tune the weight decay
    wd = hp.uniform("wd", 0.0, 0.0001)
    # we need to tune the dropout rate
    d = hp.uniform('d', 0.0, 0.9)
    # we need to tune the batch size()
    b = hp.quniform('b', 1, 50, 1)
    #  we need to tune the null arg penalty
    nap = hp.uniform('nap', 0, 4)
    return [lr, h, aux, wd, d, b, nap]


if __name__ == '__main__':
    pargs = parser.parse_args()

    w2v = gensim.models.KeyedVectors.load_word2vec_format(pargs.wv, binary=pargs.wv.endswith("bin"))

    index = Index(w2v)
    N_word_embeddings = w2v.vector_size
    train_dataset, N_classes_argument, N_classes_trigger = load(pargs.trainfile, pargs.evaltrain, index, True)

    if len(pargs.devfile) > 1:

        dev_dataset, N_classes_argument_dev, N_classes_event_dev = load(pargs.devfile, pargs.evaldev, index)

        N_classes_argument = max([N_classes_argument, N_classes_argument_dev])
        N_classes_trigger = max([N_classes_trigger, N_classes_event_dev])

    else:
        dev_dataset = None

    logger.debug("event types: %s" % str(index.event_index))
    index.freeze()
    vocab_size = len(index.word_index)
    space = define_search_space()
    best = fmin(fn=objective, space=space, algo=tpe.rand.suggest, max_evals=50, verbose=2)
    print(best)
