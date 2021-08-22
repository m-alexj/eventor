'''
Created on Apr 10, 2018

@author: judeaax
'''
from eventor import parse_args
import gensim
from model.index import Index
from bio.load import load, buildRestrictions, Data
import logging
from random import shuffle
from tqdm import tqdm
from torch import nn, optim
import torch
from model.model import MasterModel
from optim.powersign import PowerSign
from model.trainer import flatten_params, load_params, set_to_null, accuracy
import numpy
from itertools import groupby
import itertools
from builtins import int

logger = logging.getLogger('CrossTrainer')


def getWordEmbeddings(vocab_size, w2v, N_word_embeddings):
    '''
    Produces word embeddings.
    :param vocab_size: Size of the vocabulary.
    :param w2v: A gensim w2v instance.
    :param N_word_embeddings: Embeddings dimensionality.
    '''
    def fill_embeddings(model_embeddings, embedding_vectors, N):
        '''
        Fills the embeddings.
        :param model_embeddings: nn.Embeddings instance.
        :param embedding_vectors: Gensim word vectors.
        :param N: Embeddings dimensionality.
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
        # logger.debug("Word embedding miss ratio: %.2f" % (word_embedding_misses / len(index.word_index)))
        # plug these into embedding matrix inside model
        if args.cuda:
            wemb = wemb.cuda()
        model_embeddings.state_dict()['weight'].copy_(wemb)
    # create what the model will use
    word_embeddings = nn.Embedding(vocab_size, N_word_embeddings, padding_idx=0)
    if args.cuda:
        word_embeddings = word_embeddings.cuda()
    # fill it with embeddings
    fill_embeddings(word_embeddings, w2v, N_word_embeddings)
    # set that embeddings should be updates
    word_embeddings.weight.requires_grad = True
    return word_embeddings


def getModel(args, word_embeddings):
    '''
    Creates the model.
    :param args: Parameters
    :param word_embeddings: A nn.Embeddings object filled with word embeddings.
    '''
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
    return model


if __name__ == '__main__':
    args = parse_args()
    # train, dev, test files
    trainfile = args.trainfile
    devfile = args.devfile
    testfile = args.testfile
    # train, dev, test eval files
    eval_trainfile = args.evaltrain
    eval_devfile = args.evaldev
    eval_testfile = args.evaltest

    # load all the data
    w2v = gensim.models.KeyedVectors.load_word2vec_format(args.wordvectors, binary=args.wordvectors.endswith("bin"))

    index = Index(w2v)
    N_word_embeddings = w2v.vector_size
    train_dataset, N_classes_argument, N_classes_trigger = load(trainfile, eval_trainfile, index)
    logger.info("Loaded %d training instances" % len(train_dataset))
    dev_dataset, N_classes_argument, N_classes_trigger = load(devfile, eval_devfile, index)
    # traindata is our new, bigger training dataset
    traindata = train_dataset + dev_dataset
    traindata.eval_triggers = {**train_dataset.eval_triggers, ** dev_dataset.eval_triggers}
    traindata.eval_arguments = {**train_dataset.eval_arguments, **dev_dataset.eval_arguments}
    # assert len(traindata.eval_triggers) == len(train_dataset.eval_triggers) + len(dev_dataset.eval_triggers), "%d + %d != %d" % (len(train_dataset.eval_triggers), len(dev_dataset.eval_triggers), len(traindata.eval_triggers))
    # assert len(traindata.eval_arguments) == len(train_dataset.eval_arguments) + len(dev_dataset.eval_arguments)
    # here we build the event argument restrictions
    buildRestrictions(traindata, index, N_classes_argument)
    logger.info("Loaded %d dev instances" % len(dev_dataset))
    test_dataset, N_classes_argument, N_classes_trigger = load(testfile, eval_testfile, index)
    logger.info("Loaded %d test instances" % len(test_dataset))
    logger.info("Combine training and dev instances: %d" % len(traindata))
    # we only allow CUDA if it was requested AND it is available
    args.cuda = args.cuda and torch.cuda.is_available()
    # we hardwire the number of representation dimensions
    args.h_dim = 300
    # the split ratio defines how much of the total data we want to use for development on each fold
    split_ratio = 0.2

    def getNextSplit(data):
        '''
        Produces a split of the provided data and returns the split in terms of
        two lists of data points.
        :param data:
        '''
        d1 = Data()
        d2 = Data()
        trig_eval_1 = {}
        trig_eval_2 = {}
        arg_eval_1 = {}
        arg_eval_2 = {}

        pointsByDocument = {x: list(y) for x, y in groupby(data, lambda x: x.did)}
        docIndices = list(pointsByDocument.keys())
        shuffle(docIndices)
        splitpoint = int(len(docIndices) * split_ratio)
        s1, s2 = docIndices[splitpoint:], docIndices[:splitpoint]
        for p in s1:
            points = pointsByDocument[p]
            for point in points:
                d1.add(point)
                if point.sid in data.eval_triggers:
                    trig_eval_1[point.sid] = data.eval_triggers[point.sid]
                if point.sid in data.eval_arguments:
                    arg_eval_1[point.sid] = data.eval_arguments[point.sid]
        d1.eval_triggers = trig_eval_1
        d1.eval_arguments = arg_eval_1
        for p in s2:
            points = pointsByDocument[p]
            for point in points:
                d2.add(point)
                if point.sid in data.eval_triggers:
                    trig_eval_2[point.sid] = data.eval_triggers[point.sid]
                if point.sid in data.eval_arguments:
                    arg_eval_2[point.sid] = data.eval_arguments[point.sid]
        d2.eval_triggers = trig_eval_2
        d2.eval_arguments = arg_eval_2
        return d1, d2
#         d1 = Data()
#         etrig1 = {x: y for x, y in data.eval_triggers if x in s1}
#         earg1 = {x: y for x, y in data.eval_arguments if x in s1}
#         
#         return l1, l2

    def train(points, model, epoch):
        '''
        Performs one training epoch and returns the loss.
        :param indices: Training points.
        :param model: A model.
        :param epoch: The epoch number (model behavior may depend on epoch number).
        '''
        model.train()
        loss = 0.0
        for i in range(len(points)):
            graph = points[i]
            r = model(graph, epoch)
            graph.reset()
            loss += r
            print("\r%f" % (i / len(points)), end='')
        # this final call is to make an final update if the final batch is smaller
        # than the batch size
        r += model(None, epoch)
        return loss / len(points)

    def test(points, model, epoch):
        model.eval()
        original_param = flatten_params(model)  # save current params
        load_params(model, model.avg_param)
        print(model.avg_param)
        loss = 0
        for i in range(len(points)):
            graph = points[i]
            loss += model(graph, epoch)
            graph.reset()
        load_params(model, original_param)
        return loss

    # size of the word embeddings vocabulary
    vocab_size = len(index.word_index)
    # set the logging of the 'model' module to info
    logging.getLogger("Eventor.model").setLevel(logging.INFO)
    # the number of folds
    folds = 2
    # the number of epochs
    epochs = 2
    # the final parameters
    final_averaged_parameters = 0
    for fold in range(folds):
        # first we create a new split for the new fold
        train_points, dev_points = getNextSplit(traindata)
        # we create embeddings (we cannot do this outside of the loop because embeddings may be updated during training)
        # for research, this is restricted to the words in train/dev/test. In the wild, ALL words we have embeddings for should be in here
        embeddings = getWordEmbeddings(vocab_size, w2v, N_word_embeddings)
        # then we create a new model
        model = getModel(args, embeddings)
        # here we save the currently best model parameters of this fold
        current_best_weights = None
        #
        max_dev_f1 = float('-inf')
        best_trig_f1, best_arg_f1 = None, None
        # here we do the actual training
        for epoch in range(epochs):
            # do one round of training
            train_loss = train(train_points, model, epoch)
            logger.info("Epoch %d: training loss %f" % (epoch, train_loss))
            # assert than we did not wrongly copy averaged parameters across folds
            if epoch == 0:
                assert isinstance(model.avg_param, int)
            # average model parameters
            model.avg_param = 0.9 * model.avg_param + 0.1 * flatten_params(model) if not isinstance(model.avg_param, int) else flatten_params(model)
            # do one round of testing
            dev_loss = test(dev_points, model, epoch)
            logger.info("Epoch %d: testing loss %f" % (epoch, dev_loss))
            # get evaluation numbers
            dev_p_trig, dev_r_trig, dev_f1_trig, dev_p_arg, dev_r_arg, dev_f1_arg, dev_predictions_trigger, dev_predictions_argument, _ = accuracy("dev", dev_points, model.index, dev_loss, True, True, True)
            # reset all predictions
            set_to_null(traindata, model.index)
            # the measure we want to keep track of is the average of trigger and argument f1
            dev_f1 = numpy.average([dev_f1_trig, dev_f1_arg])

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                best_trig_f1 = dev_f1_trig
                best_arg_f1 = dev_f1_arg
                current_best_weights = model.avg_param.clone()
        # here we are done with this fold
        logger.info("Done with fold %d. Best trigger f1: %.3f, best argument f1: %.3f" % (fold, best_trig_f1, best_arg_f1))
        # add the current best weights to those determined in the last fold; we will average later
        final_averaged_parameters += current_best_weights
    final_averaged_parameters /= folds
    logger.info('Finished cross-training. Evaluating on the test set')
    # make the embeddings for the last time
    embeddings = getWordEmbeddings(vocab_size, w2v, N_word_embeddings)
    # then we create a new model
    model = getModel(args, embeddings)
    # we load the final parameters
    model.avg_param = final_averaged_parameters
    # perform a test on the test set
    test_loss = test(test_dataset, model, -1)
    # and print evaluation results
    test_p_trig, test_r_trig, test_f1_trig, test_p_arg, test_r_arg, test_f1_arg, test_predictions_trigger, test_predictions_argument, _ = accuracy("test", test_dataset, model.index, test_loss, True, True)
    set_to_null(test_dataset, model.index)
