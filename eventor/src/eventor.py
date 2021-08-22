'''
Created on Apr 28, 2017

@author: judeaax
'''

import sys, warnings, traceback, torch
import random
from model import config
import numpy
import logging
import argparse
import shutil
import os
import gensim
from model.index import Index
from bio.load import load
from model.model import MasterModel
from torch import optim, nn
from model.trainer import train
from optim.powersign import PowerSign

random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)

logger = logging.getLogger("Eventor")
logging.basicConfig(level=logging.DEBUG)


def parse_args():
    infix = '.smaller'
    default_train = "/data/nlp/judeaax/event2/eventor.train%s.simple.carved.no_it.txt" % (infix)
    default_train_eval = "/data/nlp/judeaax/event2/eventor.eval.train%s.txt" % (infix)
    default_dev = default_train # "/data/nlp/judeaax/event2/eventor.dev.simple.carved.no_it.txt"
    default_dev_eval = default_train_eval # "/data/nlp/judeaax/event2/eventor.eval.dev.txt"
    default_test = default_train # "/data/nlp/judeaax/event2/eventor.test.simple.carved.no_it.txt"
    default_test_eval = default_train_eval # "/data/nlp/judeaax/event2/eventor.eval.test.txt"

    parser = argparse.ArgumentParser(description='PyTorch Child-Sum dependency tree event extractor')
    parser.add_argument('-epochs', default=2, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-evaltrain', default=default_train_eval, type=str,
                        help='Training data evaluation file', dest="evaltrain")
    parser.add_argument('-evaldev', default=default_dev_eval, type=str,
                        help='Development data evaluation file', dest="evaldev")
    parser.add_argument('-evaltest', default=default_test_eval, type=str,
                        help='Test data evaluation file', dest="evaltest")
    parser.add_argument('-train', default=default_train, type=str,
                        help='Training data', dest="trainfile")
    parser.add_argument('-dev', default=default_dev, type=str,
                        help='Development data', dest="devfile")
    parser.add_argument('-test', default=default_test, type=str,
                        help='Test data', dest="testfile")
    parser.add_argument('-arglosspenalty', default=0.0, type=float,
                        help='Argument training loss penalty for false negatives', dest="arglosspenalty")
    parser.add_argument('-wv', default="/data/nlp/judeaax/event2/cyb.txt", type=str,
                        help='Word vector file', dest="wordvectors")
    parser.add_argument('-tlr', default=0.0008, type=float,
                        metavar='tlr', help='initial learning rate for triggers')
    parser.add_argument('-alr', default=0.0008, type=float,
                        metavar='alr', help='initial learning rate for arguments')
    parser.add_argument('-olr', default=0.0008, type=float,
                        metavar='olr', help='initial learning rate for others')
    parser.add_argument('-batch_size', default=1, type=float,
                        metavar='batch_size', help='BAtch size')
    parser.add_argument('-skipnull', default=0.8, type=float,
                        help='Percentage of non-events to skip during training (default: 0.8)')
    parser.add_argument('-wd', default=0.0, type=float,
                        help='weight decay (default: 0.0)')
    parser.add_argument('-optim', default='powersign',
                        help='argument_optimizer (default: adam)')
    parser.add_argument('-seed', default=9913, type=int,
                        help='random seed (default: 123)')
    parser.add_argument('-evalevery', default=1, type=int,
                        help='evaluate on the dev set every X epochs (default 1)')
    parser.add_argument('-momentum', default=0, type=float, dest="moment",
                        help='Momentum (default 0)')
    parser.add_argument('-nesterov', default=False, dest='nest', action='store_true', help="Activates Nesterov momentum")
    parser.add_argument('-printwrong', default=False, dest='printwrong', action='store_true', help="Prints wrong classification decisions, if logger level is set to DEBUG")
    parser.add_argument('-traintriggers', default=True, dest='traintriggers', action='store_true', help="If set, applies trigger training. Ohterwise, gold triggers are used.")
    parser.add_argument('-trainarguments', default=True, dest='trainarguments', action='store_true', help="If set, applies argument training. Ohterwise, gold arguments are used.")
    parser.add_argument('-o', default="model.obj", type=str,
                        help='Name to which the model should be saved')
    parser.add_argument('-addDevToTest', default="", type=str,
                        help='Sentence ID; every data point with this ID will be added to the training data')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('-cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('-no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    shutil.rmtree("graphs", ignore_errors=True)
    os.makedirs("graphs", exist_ok=True)
    global args
    args = parse_args()
    print(args)

    w2v = gensim.models.KeyedVectors.load_word2vec_format(args.wordvectors, binary=args.wordvectors.endswith("bin"))

    index = Index(w2v)
    N_word_embeddings = w2v.vector_size
    train_dataset, N_classes_argument, N_classes_trigger = load(args.trainfile, args.evaltrain, index, True)

    if len(args.devfile) > 1:

        dev_dataset, N_classes_argument_dev, N_classes_event_dev = load(args.devfile, args.evaldev, index)
        # this is a dirty and cheap way to get indices which are valid also for the test set
        if args.testfile is not None:
            test_dataset, _, _ = load(args.testfile, args.evaltest, index)

        N_classes_argument = max([N_classes_argument, N_classes_argument_dev])
        N_classes_trigger = max([N_classes_trigger, N_classes_event_dev])

    else:
        dev_dataset = None

    logger.debug("event types: %s" % str(index.event_index))
    index.freeze()
    vocab_size = len(index.word_index)
    logger.debug("Vocab size: %d, vector size: %d, classes trigger: %d, classes argument: %d, training samples: %d" % (vocab_size, N_word_embeddings, N_classes_trigger, N_classes_argument, len(train_dataset)))
#     write_trees(train_dataset, index, "train")
    logger.debug("%d dependencies: %s" % (len(index.dependency_index), str(index.dependency_index)))
    args.h_dim = 300
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    def fill_embeddings(model_embeddings, embedding_vectors, N):
        '''
        Fills the embeddings
        :param model_embeddings:
        :param embedding_vectors:
        :param N:
        '''
        wemb = torch.zeros(vocab_size, N).normal_(-0.25, 0.25)
        word_embedding_misses = 0
        for word, idx in index.word_index.items():
            if word in embedding_vectors:
                wemb[idx] = torch.FloatTensor(embedding_vectors[word].astype("float64"))
            elif idx > 0:
                word_embedding_misses += 1
            else:
                wemb[idx] = torch.zeros(N)
        logger.debug("Word embedding miss ratio: %.2f" % (word_embedding_misses / len(index.word_index)))
        # plug these into embedding matrix inside model
        if args.cuda:
            wemb = wemb.cuda()
        model_embeddings.state_dict()['weight'].copy_(wemb)

    word_embeddings = nn.Embedding(vocab_size, N_word_embeddings, padding_idx=0)
    if args.cuda:
        word_embeddings = word_embeddings.cuda()
    fill_embeddings(word_embeddings, w2v, N_word_embeddings)
    word_embeddings.weight.requires_grad = True

    # initialize model, criterion/loss_function, argument_optimizer
    model = MasterModel(N_word_embeddings, 50, args.h_dim, args.batch_size, 0.0, args.arglosspenalty, args.skipnull, index, word_embeddings, args.cuda)
    if args.cuda:
        model = model.cuda()
    model.index = index

    trig_params, arg_params, other_params = model.custom_parameters()
    parameter_list = [{'params': trig_params, 'lr': args.tlr},
                      {'params': arg_params, 'lr': args.alr},
                      {'params': other_params, 'lr': args.olr}]
    if args.optim == 'powersign':
        optimizer = PowerSign(parameter_list)
    if args.optim == 'adam':
        optimizer = optim.Adam(parameter_list,
                                        weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(parameter_list, weight_decay=args.wd)
    elif args.optim == "adadelta":
        optimizer = optim.Adadelta(parameter_list, weight_decay=args.wd, rho=0.95)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(parameter_list, weight_decay=args.wd, momentum=args.moment, nesterov=args.nest)
    elif args.optim == "rmsprop":
        optimizer = optim.RMSprop(parameter_list, weight_decay=args.wd)
    model.optimizer = optimizer
    parameters_with_grads = filter(lambda p: p[1].requires_grad, model.named_parameters())
    logger.debug("Parameters requiring gradients:%s" % ("\n".join([x[0] for x in parameters_with_grads])))
    train(args, model, optimizer, args.o if hasattr(args, "o") else None, train_dataset, dev_dataset, test_dataset)
