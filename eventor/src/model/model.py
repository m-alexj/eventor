import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform, xavier_normal, sparse, normal, eye, \
    uniform
import logging
from utils.utils import singleValueToLongTensor, toTensor, null_var, singleValueToFloatTensor, \
    toFloatTensor, toLongTensor, print_sizes
from itertools import groupby
from torch.nn.functional import cosine_similarity
from model.crelu import CReLU
import numpy
from utils import utils
from torch.nn.utils.clip_grad import clip_grad_norm
from theano.gof.cmodule import last_access_time
import re
from model import config
import mmh3
from matplotlib.cbook import tofloat
from boto.beanstalk.response import Trigger
import collections
from torch.autograd.variable import Variable
from torch import optim
from torch.nn.modules.loss import NLLLoss
import random
from model.trainer import flatten_params

logger = logging.getLogger("Eventor.model")


class GCNLayer(nn.Module):
    '''
    A Graph Convolution layer.
    '''
    def __init__(self, hin, hout, N_dependencies):
        super(GCNLayer, self).__init__()
        self.w = Parameter(torch.zeros(hout, hin))

        self.w_gate = Parameter(torch.zeros(1, hin))

        self.b = Parameter(torch.zeros(hout, N_dependencies))
        self.b_gate = Parameter(torch.zeros(1, N_dependencies))

    def forward(self, _inp, adj_matrix, adj_matrix_dependencies):
        d = Variable(adj_matrix_dependencies)
        # compute v = w * _inp + b
        # note that the bias depends on the actual dependencies of a word.
        # therefore, we sum up the dependency embeddings of all active
        # dependencies of each word. Dependency embeddings have
        # dimensionality hout
        v = _inp@self.w.t() + d@self.b.t()
        # compute the scalar gate for each word. the same applies here
        # to the bias (depends on dependencies of a word)
        g = F.sigmoid(F.linear(_inp, self.w_gate, d@self.b_gate.t()))
        # multiply v and g
        v = v * g
        # compute sum over neighbors of each word;
        # the adjacency matrix is zero for pairs which are not neighbors,
        # so that the resulting part there is also zero, otherwise it is
        # equal to the sum of elements of v corresponding to the neighbors.
        # Note that every node is a neighbor of itself.
        s = adj_matrix@v
        return s


class GCN(nn.Module):
    '''
    Produces a vector representation of a sentence.
    The representation is based on the dependency
    graph of the sentence, and on a sequence of
    input representations (representations of
    words in the sentence).
    '''

    def __init__(self, hin, hout, p, N_dependencies):
        super(GCN, self).__init__()
        self.g1 = GCNLayer(hin, hout, N_dependencies)
        self.g2 = GCNLayer(hout, hout, N_dependencies)
        self.dropout_p = p

    def forward(self, x, graph):
        # apply edge dropout; we do this by
        # dropping out values in the adjacency
        # matrix
        adj_mask = torch.bernoulli(Variable(torch.zeros_like(graph.adj_matrix)).fill_(self.dropout_p))
        adj = adj_mask * Variable(graph.adj_matrix)
        x = F.relu(self.g1(x, adj, graph.adj_matrix_dependencies))
        # x = F.relu(self.g2(x, adj, graph.adj_matrix_dependencies))
        return x


class ContextEncoder(nn.Module):
    '''
    Produces a vector representation for lexical contexts.
    Every index which falls out of the allowed boundaries
    (either < 0, or > sentence size) will be resolved by
    a zero vector.
    '''

    def __init__(self, word_embeddings, cudaFlag):
        '''
        Initialization.
        :param word_embeddings: Embeddings instance. The value '0' should be
            used for masking.
        :param cudaFlag: Indicates if CUDA is available.
        '''
        super(ContextEncoder, self).__init__()
        self.word_embeddings = word_embeddings
        self.cudaFlag = cudaFlag

    def forward(self, wid, wsize, graph):
        begin = max([0, wid - wsize])
        end = min([graph.sentenceSize(), wid + wsize + 1])
        wids = graph.words[begin:end]
        if len(wids) < 2 * wsize + 1:
            # we have to pad
            pad_left = wsize - wid
            if pad_left > 0:
                wids = [0] * pad_left + wids
            pad_right = wsize - (graph.sentenceSize() - wid - 1)
            if pad_right > 0:
                wids = wids + [0] * pad_right
        assert len(wids) == 2 * wsize + 1
        indices = Variable(toLongTensor(wids, self.cudaFlag))
        return self.word_embeddings(indices)


class TriggerEncoder(nn.Module):
    '''
    Produces a vector representation for a trigger candidate.
    '''

    def __init__(self, word_embeddings, cudaFlag):
        super(TriggerEncoder, self).__init__()
        self.cudaFlag = cudaFlag
        self.word_embeddings = word_embeddings

        self.context_encoder = ContextEncoder(self.word_embeddings, self.cudaFlag)

    def forward(self, i, graph, sentence, word_feature_abstractions):
        trigger_sentence_h = sentence[i].view(1, -1)
        trigger_feature_h = word_feature_abstractions[i].view(1, -1)
        context = self.context_encoder(i, 2, graph)
        total_features = [trigger_sentence_h, trigger_feature_h, context.view(1, 5 * context.size()[1])]
        trigger_prediction_input = torch.cat(total_features, 1)
        return trigger_prediction_input


class ArgumentEncoder(nn.Module):
    '''
    Produces a vector representation for an argument candidate.
    '''

    def __init__(self, word_embeddings, event_embeddings, entity_type_embeddings, feature_abstractor, cudaFlag):
        super(ArgumentEncoder, self).__init__()
        self.word_embeddings = word_embeddings
        self.event_embeddings = event_embeddings
        self.entity_type_embeddings = entity_type_embeddings
        self.feature_abstractor = feature_abstractor
        self.cudaFlag = cudaFlag

        self.context_encoder = ContextEncoder(self.word_embeddings, self.cudaFlag)

    def forward(self, i, j, graph, sentence):
        # event_h = event_embeddings(Variable(singleValueToLongTensor(self.predicted_triggers[i], self.cuda)))
        entity_type_h = self.entity_type_embeddings(Variable(singleValueToLongTensor(graph.mentions[j][2], self.cudaFlag)))
        begin, end = graph.mentions[j][:2]
        begin = end
        sentence_h = sentence[end].view(1, -1)

        context = self.context_encoder(end, 2, graph)
        trigger_context = self.context_encoder(i, 2, graph)

        features, feature_indices = graph.mention_features[j]
        onehot_argument_features = null_var((config.hash_dimensions,), self.cudaFlag)
        onehot_argument_features[feature_indices[i].cuda() if self.cudaFlag else feature_indices[i]] = 1
        feature_repr = torch.nn.functional.sigmoid(self.feature_abstractor(onehot_argument_features.view(1, -1)))

        total_features = [sentence_h, feature_repr, trigger_context.view(1, 5 * trigger_context.size()[1]), context.view(1, 5 * context.size()[1]), entity_type_h, sentence[i].detach().view(1, -1)]

        return torch.cat(total_features, 1)


class SentenceEncoder(nn.Module):
    '''
    Produces a matrix-representation for a sentence.
    Each word will correspond to one element of the matrix.
    '''

    def __init__(self, word_embeddings, entity_type_embeddings, N_word_embeddings, h_dim, aux_dim, index, cudaFlag):
        super(SentenceEncoder, self).__init__()
        self.word_embeddings = word_embeddings
        self.entity_type_embeddings = entity_type_embeddings
        self.trigger_feature_abstractor = nn.Linear(config.trigger_hash_dimensions, h_dim, bias=False)
        self.sentence_rnn = nn.GRU(N_word_embeddings + aux_dim + len(index.dependency_index), h_dim, bidirectional=True)
        self.cudaFlag = cudaFlag
        self.index = index
        self.trigger_gcn = GCN(2 * h_dim, h_dim, 0.5, len(index.dependency_index))

    def _encode_words(self, graph):
        if self.cudaFlag:
            graph.adj_matrix_dependencies = graph.adj_matrix_dependencies.cuda()
            graph.adj_matrix = graph.adj_matrix.cuda()
        # input to the RNN is a concatenation of (word, entity BILOU) embeddings and feature abstractions
        if not hasattr(self, 'word_indices'):
            graph.word_indices = toLongTensor(graph.words, self.cudaFlag)
        if not hasattr(self, 'bilou_entity_tag_indices'):
            graph.bilou_entity_tag_indices = toLongTensor(graph.bilou_entity_tags, self.cudaFlag)
        words = self.word_embeddings(Variable(graph.word_indices))
        bilou_tags = self.entity_type_embeddings(Variable(graph.bilou_entity_tag_indices))
        word_feature_abstractions = self.trigger_feature_abstractor(Variable(graph.onehot_node_feature_vectors.cuda() if self.cudaFlag else graph.onehot_node_feature_vectors))
        if graph.adj_matrix_dependencies.size()[-1] >= len(self.index.dependency_index):
            graph.adj_matrix_dependencies = graph.adj_matrix_dependencies[:, :len(self.index.dependency_index)]
        rnn_input = torch.cat([words, bilou_tags, Variable(graph.adj_matrix_dependencies)], 1).unsqueeze(1).transpose(0, 1)
        sentence_repr = self.sentence_rnn(rnn_input)[0].squeeze(0)
        return sentence_repr, word_feature_abstractions

    def forward(self, graph):
        sentence, word_feature_abstractions = self._encode_words(graph)
        sentence = self.trigger_gcn(sentence, graph)
        return sentence, word_feature_abstractions


class MasterModel(nn.Module):

    def __init__(self, N_word_embeddings, aux_dim, h_dim, batch_size, penultimate_argument_dropout, arg_loss_penalty, skipnull, index, word_embeddings, cudaFlag):
        super(MasterModel, self).__init__()
        self.N_events = len(index.event_index)
        self.N_arguments = len(index.argument_index)
        self.batch_size = batch_size
        self.arg_loss_penalty = arg_loss_penalty
        self.skipnull = skipnull

        self.word_embeddings = word_embeddings

        self.entity_type_embeddings = nn.Embedding(len(index.entity_index), aux_dim)
        self.event_type_embeddings = nn.Embedding(len(index.event_index), aux_dim, 0)
        self.argument_feature_abstractor = nn.Linear(config.hash_dimensions, h_dim, bias=False)
        self.trigger_penultimate = nn.Linear(2 * h_dim + 5 * N_word_embeddings, 600)
        self.trigger_ultimate = nn.Linear(600, self.N_events)
        self.argument_penultimate = nn.Linear(10 * N_word_embeddings + aux_dim + 3 * h_dim + self.N_events, 600)
        self.argument_ultimate = nn.Linear(600, self.N_arguments)
        self.sentence_encoder = SentenceEncoder(self.word_embeddings, self.entity_type_embeddings, N_word_embeddings, h_dim, aux_dim, index, cudaFlag)
        self.trigger_encoder = TriggerEncoder(self.word_embeddings, cudaFlag)
        self.argument_encoder = ArgumentEncoder(self.word_embeddings, self.event_type_embeddings, self.entity_type_embeddings, self.argument_feature_abstractor, cudaFlag)
        self.penultimate_argument_dropout = nn.Dropout(penultimate_argument_dropout)
        self.penultimate_trigger_dropout = nn.Dropout(0.3)

        self.criterion = NLLLoss()

        self.cudaFlag = cudaFlag
        self.reset_parameters()

        self.losses = []

        self.avg_param = 0

    def reset_parameters(self):

        parameters = [x for x in self.named_parameters() if len(x[1].size()) > 1]
        logger.debug("Initializing %d parameters" % len(parameters))
        for name, par in parameters:
            if name.startswith('word_embeddings'):
                logger.debug("NOT initializing " + name)
            else:
                logger.debug("Initializing " + name)
                xavier_uniform(par)

    def custom_parameters(self):
        '''
        Returns parameters concerning triggers, arguments, and others
        separately as three different lists.
        '''
        triggers, arguments, other = [], [], []
        for name, parameter in self.named_parameters():
            if name. startswith('trigger'):
                triggers.append(parameter)
            elif name.startswith('argument'):
                arguments.append(parameter)
            else:
                other.append(parameter)
        return triggers, arguments, other

    def forward(self, graph, epoch):

        if graph is None:
            loss = 0
            if self.training:
                if len(self.losses) > 0:
                    self.optimizer.zero_grad()
                    loss = torch.cat(self.losses).mean()
                    loss.backward()
                    clip_grad_norm(self.parameters(), 2)
                    self.optimizer.step()
                    self.losses = []
                    loss = loss.data.cpu().numpy()[0]
            return loss
        loss = 0

        # we encode the sentence
        sentence, word_feature_abstractions = self.sentence_encoder(graph)
        argument_memory = null_var((graph.argumentCandidatesSize(), self.N_events), self.cudaFlag)
        # then, we perform trigger detection
        for i in range(graph.sentenceSize()):

            if i == 0:
                graph.predicted_triggers = [0] * graph.sentenceSize()
                graph.predicted_arguments_by_trigger = {}

            if not graph.isPotentialTrigger(i):
                # if graph.goldTriggerIsEvent(i):
                    # logger.warn("Gold trigger skipped: " + self.index.getWord(graph.words[i]) + ":" + self.index.getPos(graph.pos_tags[i]))
                continue

            if self.training and not graph.goldTriggerIsEvent(i) and random.uniform(0.0, 1.0) < self.skipnull:
                continue

            trigger_prediction_input = self.trigger_encoder(i, graph, sentence, word_feature_abstractions)
            trigger_prediction_output = F.log_softmax(self.trigger_ultimate(self.penultimate_trigger_dropout(F.sigmoid(self.trigger_penultimate(trigger_prediction_input)))), 1)
            trigger_loss = self.criterion(trigger_prediction_output, Variable(singleValueToLongTensor(graph.getGoldTrigger(i), self.cudaFlag)))
            trigger_prediction = torch.max(trigger_prediction_output, 1)[1]
            graph.predicted_triggers[i] = trigger_prediction.data.cpu()[0]
            if graph.predictionIsEvent(i) and graph.getEventSupertype(i) == graph.getGoldEventSupertype(i):
                # print(self.index.getEventType(graph.predicted_triggers[i]), self.index.getEventType(graph.gold_triggers[i]))
                trigger_loss = trigger_loss * 0.5
            loss += trigger_loss

            # if there is an gold event, or if we predicted an event, we have
            # to predict arguments
            if (self.training and graph.goldTriggerIsEvent(i)) or graph.predictionIsEvent(i):
                # now we need to predict arguments
                for j in range(graph.argumentCandidatesSize()):

                    if graph.mentions[j][2] not in self.index.restriction_masks[graph.predicted_triggers[i]]:
                        # if the entity subtype is not in the allowed types, continue with the next mention
                        continue
                    argument_h = self.argument_encoder(i, j, graph, sentence)
                    argument_h = torch.cat([argument_h, argument_memory[j].view(1, self.N_events)], 1)
#                     argument_prediction_state = self.argument_penultimate(argument_h)@argument_ultimate + ultimate_argument_bias
                    argument_prediction_state = self.argument_ultimate(self.penultimate_argument_dropout(F.sigmoid(self.argument_penultimate(argument_h))))
                    mask = Variable(self.index.restriction_masks[graph.predicted_triggers[i]][graph.mentions[j][2]])
                    if self.cudaFlag:
                        mask = mask.cuda()
                    argument_prediction_output = F.log_softmax(argument_prediction_state + mask, 1)
                    argument_prediction = torch.max(argument_prediction_output, 1)[1].data.cpu()[0]
                    if argument_prediction != 0:
                        if i not in graph.predicted_arguments_by_trigger:
                            graph.predicted_arguments_by_trigger[i] = {}
                        assert j not in graph.predicted_arguments_by_trigger[i]
                        graph.predicted_arguments_by_trigger[i][j] = [argument_prediction]
                        argument_memory[j, graph.predicted_triggers[i]] = 1
                    if self.training:
                        # it may be that the argument prediction was built on a wrong mask because the trigger type was not correct. In such a case we have to recompute it without a mask
                        argument_prediction_output = F.log_softmax(argument_prediction_state, 1)
                        for gold_type in graph.getGoldArgumentsByIds(i, j):
                            argument_loss = self.criterion(argument_prediction_output, Variable(singleValueToLongTensor(gold_type, self.cudaFlag)))
                            # if gold_type == 0:
                                # argument_loss = argument_loss * 0.1
                            # if argument_loss.data.cpu().numpy()[0] == float('inf'):
                                # if self.training:
                                    # raise BaseException()
                                # else:
                                    # logger.warn("Mask forbids gold role (gold role %s, event %s)")
                            if argument_prediction == 0 and gold_type != 0:
                                argument_loss = argument_loss * (self.arg_loss_penalty + 1)
                            loss += argument_loss

#         if epoch == 2 and not self.training:
#             graph.rememberTriggerPrediction()
        # graph.compactifyArguments()
        if self.training and not isinstance(loss, int):
            self.losses.append(loss)
            if len(self.losses) >= self.batch_size:
                self.optimizer.zero_grad()
                loss = torch.cat(self.losses).mean()
                loss.backward()
                clip_grad_norm(self.parameters(), 2)
                self.optimizer.step()
                self.losses = []
                return loss.data.cpu().numpy()[0]
        return 0
