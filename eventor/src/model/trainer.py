from tqdm import tqdm
import torch
import numpy
from torch.nn.utils.clip_grad import clip_grad_norm
from sklearn.metrics.classification import accuracy_score, f1_score, \
    precision_score, recall_score
import logging
from visual.visualize import visualize_dependencies, visualize_backprop_graph
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LambdaLR, \
    StepLR, MultiStepLR
import dill as pickle
# import pickle
from itertools import groupby
from bio.TensorboardLogger import TensorboardLogger
from datetime import datetime
import math
from builtins import set, sum
from boto.beanstalk.response import Trigger
import sys
from torch.nn.modules.loss import _Loss
from math import isnan
from torch import optim
import random
from random import shuffle


logger = logging.getLogger("Eventor.trainer")


def accuracy(name, dataset, index, loss, train_trigger=True, print_numbers=False,
             compute_wrong_decisions=False, print_wrong_decisions=False,
             unreachable_triggers=None, unreachable_arguments=None):
    '''
    Given a dataset. computes, prints and returns all relevant evaluation
    metrics.
    :param name:
    :param dataset:
    :param index:
    :param loss:
    :param train_trigger:
    '''

    predicted_triggers = 0
    gold_triggers = 0
    correct_triggers = 0
    correct_triggers_identification = 0

    predicted_arguments = 0
    gold_arguments = 0
    correct_arguments = 0
    correct_arguments_identification = 0
    correct_arguments_lenient = 0
    processed = {}
    entity_frequencies = {}
    arg_frequencies = {}
    accessible_triggers = 0

    argument_collection = {}

    graphs_by_document = groupby(dataset, lambda x: x.sid)

    def collect_arguments(argument_dict, triggers, mentions):
        collected = []
        for i, subdict in argument_dict.items():
            etype = triggers[i]
            for j, roles in subdict.items():
                mention = mentions[j]
                for role in roles:
                    collected.append((etype, role, mention[8]))
        return collected

    for did, graphs in graphs_by_document:
        graphs = list(graphs)
        # collect all the gold arguments
        _gold_arguments = []
        _pred_arguments = []
        for graph in graphs:
            _gold_arguments += collect_arguments(graph.gold_arguments_by_trigger, graph.gold_triggers, graph.mentions)
            _pred_arguments += collect_arguments(graph.predicted_arguments_by_trigger, graph.predicted_triggers, graph.mentions)
        predicted_arguments += len(_pred_arguments)
        gold_arguments += len(_gold_arguments)
        for _pred_argument in _pred_arguments:
            if _pred_argument in _gold_arguments:
                correct_arguments += 1

    for i in range(len(dataset)):
        graph = dataset[i]
        sid = graph.sid
        # the processed dict will be used to compute which gold triggers or arguments were missed
        if sid not in processed:
            processed[sid] = {}

#         if graph.predicted_triggers != graph.gold_triggers:
#             print(graph.predicted_triggers, graph.gold_triggers, [index.getWord(x) for x in graph.words], graph.sid, sep='\n')
        if graph.predicted_triggers is not None:
            for word_id in range(len(graph.predicted_triggers)):
                if graph.predictionIsEvent(word_id):
                    predicted_triggers += 1
                    if graph.triggerPredictionIsCorrect(word_id):
                        correct_triggers += 1
                    if graph.predicted_triggers[word_id] > 0 and graph.gold_triggers[word_id] > 0:
                        correct_triggers_identification += 1
#                     predicted_argument_instances = graph.getPredictedArgumentsOf(word_id)
#                     if predicted_argument_instances is not None:
#                         predicted_arguments += len(predicted_argument_instances)
#                         if graph.triggerPredictionIsCorrect(word_id):
#                             for mention_id, role_id in predicted_argument_instances.items():
#                                 if graph.correctArgumentPrediction(word_id, mention_id, role_id):
#                                     correct_arguments += 1

    for sid, data in dataset.eval_triggers.items():
        gold_triggers += len(data)

#     for sid, data in dataset.eval_arguments.items():
#         gold_arguments += len(data)
# 
#         if sid in argument_collection:
#             for predicted_argument in argument_collection[sid]:
#                 # data format:
#                 # begin offset    end offset    trigger type    argument type    trigger begin offset    trigger end offset
#                 matching_gold = [x for x in data if x[2] == predicted_argument.trigger.output and x[3] == predicted_argument.output and x[0] == predicted_argument.begin and x[1] == predicted_argument.end]
#                 if len(matching_gold) > 0:
#                     correct_arguments_lenient += 1
# 
#         if print_wrong_decisions:
#             if sid not in processed:
#                 logger.warn('Did not process anything in %s' % sid)
#                 continue
#             for _processed_arg in processed[sid]['arguments']:
#                 if _processed_arg not in data:
#                     logger.debug('Predicted not in eval data: %s in %s' % (str(_processed_arg), sid))
#             for _gold_arg in data:
#                 if _gold_arg not in processed[sid]['arguments']:
#                     logger.debug('Gold not in predicted data: %s in %s\n\tPredicted data%s\n\tGold data%s' % (str(_gold_arg), sid, processed[sid]['arguments'], data))

    p_arguments = (correct_arguments / predicted_arguments) if predicted_arguments > 0 else 0
    r_arguments = correct_arguments / gold_arguments if gold_arguments > 0 else 0
    f1_arguments = ((2 * p_arguments * r_arguments) / (p_arguments + r_arguments)) if p_arguments > 0 or r_arguments > 0 else 0

    p_arguments_lenient = (correct_arguments_lenient / predicted_arguments) if predicted_arguments > 0 else 0
    r_arguments_lenient = correct_arguments_lenient / gold_arguments if gold_arguments > 0 else 0
    f1_arguments_lenient = ((2 * p_arguments_lenient * r_arguments_lenient) / (p_arguments_lenient + r_arguments_lenient)) if p_arguments_lenient > 0 or r_arguments_lenient > 0 else 0

    p_arguments_ident = (correct_arguments_identification / predicted_arguments) if predicted_arguments > 0 else 0
    r_arguments_ident = correct_arguments_identification / gold_arguments if gold_arguments > 0 else 0
    f1_arguments_ident = ((2 * p_arguments_ident * r_arguments_ident) / (p_arguments_ident + r_arguments_ident)) if p_arguments_ident > 0 or r_arguments_ident > 0 else 0

    p_triggers_ident = (correct_triggers_identification / predicted_triggers) if predicted_triggers > 0 else 0
    r_triggers_ident = correct_triggers_identification / gold_triggers if gold_triggers > 0 else 0
    f1_triggers_ident = ((2 * p_triggers_ident * r_triggers_ident) / (p_triggers_ident + r_triggers_ident)) if p_triggers_ident > 0 or r_triggers_ident > 0 else 0

    p_triggers = (correct_triggers / predicted_triggers) if predicted_triggers > 0 else 0
    r_triggers = correct_triggers / gold_triggers if gold_triggers > 0 else 0
    f1_triggers = ((2 * p_triggers * r_triggers) / (p_triggers + r_triggers)) if p_triggers > 0 or r_triggers > 0 else 0

    if print_numbers:
        logger.info("identification %s trigger p %.3f r %.3f f1 %.3f argument p %.3f r %.3f f1 %.3f loss %f" % (name, p_triggers_ident, r_triggers_ident, f1_triggers_ident, p_arguments_ident, r_arguments_ident, f1_arguments_ident, loss if loss is not None else -1.0))
        logger.info("classification %s trigger p %.3f r %.3f f1 %.3f argument p %.3f r %.3f f1 %.3f loss %f" % (name, p_triggers, r_triggers, f1_triggers, p_arguments, r_arguments, f1_arguments, loss if loss is not None else -1.0))
        logger.info("lenient classification %s argument p %.3f r %.3f f1 %.3f loss %f" % (name, p_arguments_lenient, r_arguments_lenient, f1_arguments_lenient, loss if loss is not None else -1.0))
        logger.info("trigger correct %d predicted %d gold %d accessible %d argument correct %d predicted %d gold %d" % (correct_triggers, predicted_triggers, gold_triggers, accessible_triggers, correct_arguments, predicted_arguments, gold_arguments))
    return p_triggers, r_triggers, f1_triggers, p_arguments, r_arguments, f1_arguments, (correct_triggers, predicted_triggers, gold_triggers), (correct_arguments, predicted_arguments, gold_arguments), (None, None)


def flatten_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()], 0)


def load_params(model, flattened):
    offset = 0
    for param in model.parameters():
        param.data.copy_(flattened[offset:offset + param.nelement()].view_as(param))
        offset += param.nelement()


def set_to_null(data, index):
    '''
    Sets prediction values of all tree nodes to their gold values
    :param data:
    '''
    for x in data:
        x.predicted_triggers = None
        x.predicted_arguments_by_trigger = None
    return data


def set_to_gold(data):
    '''
    Sets prediction values of all tree nodes to their gold values
    :param data:
    '''
    def settogold(x):
        x.output = x._type
    for x in data:
        x.setPredictionsToGold()
    return data


def train(args, model, optimizer, model_save_path, train_dataset, dev_dataset, test_dataset=None):

    # create trainer object for training and testing

    results = []
    max_dev_f1 = float('-inf')
    max_dev_trigger_f1 = float('-inf')
    max_dev_trigger_p = float('-inf')
    max_dev_trigger_r = float('-inf')
    max_dev_argument_f1 = float('-inf')
    max_dev_argument_p = float('-inf')
    max_dev_argument_r = float('-inf')
    include_triggers = True
    writer = TensorboardLogger("runs/" + str(datetime.now().ctime()))
    # evaluate train performance given that every possible prediction would be correct
    train_p_trigger_perf, train_r_trigger_perf, train_f1_trigger_perf, train_p_arg_perf, train_r_arg_perf, train_f1_arg_perf, train_predictions_trigger_perf, train_predictions_argument_perf, train_frequencies = accuracy("train_perf", set_to_gold(train_dataset), model.index, -1, include_triggers, True, True)
    logger.debug("Train entiy mention frequencies: %s" % str(train_frequencies[0]))
    logger.debug("Train argument mention frequencies: %s" % str(train_frequencies[1]))

    # remove the correct predictions again, just to be safe
    set_to_null(train_dataset, model.index)

    if dev_dataset is not None:
        # evaluate dev performance given that every possible prediction would be correct
        dev_p_trigger_perf, dev_r_trigger_perf, dev_f1_trigger_perf, dev_p_arg_perf, dev_r_arg_perf, dev_f1_arg_perf, dev_predictions_trigger_perf, dev_predictions_argument_perf, dev_frequencies = accuracy("dev_perf", set_to_gold(dev_dataset), model.index, -1, include_triggers, True, True, None, None)
        logger.debug("Development entity mention frequencies: %s" % str(dev_frequencies[0]))
        logger.debug("Development argument mention frequencies: %s" % str(dev_frequencies[1]))
        # remove the correct predictions again, just to be safe
        set_to_null(dev_dataset, model.index)
#     scheduler = ReduceLROnPlateau(trigger_optimizer, mode='max', patience=1)  # MultiStepLR(trigger_optimizer, milestones=[10])
    for epoch in range(args.epochs):

        logger.info("epoch %d" % (epoch + 1))
        train_loss = train_batched(train_dataset, model, epoch + 1)
        train_p_trig, train_r_trig, train_f1_trig, train_p_arg, train_r_arg, train_f1_arg, train_predictions_trigger, train_predictions_argument, _ = accuracy("train", train_dataset, model.index, train_loss, include_triggers, True, True)
        set_to_null(train_dataset, model.index)
        # model.avg_param = 0.9 * model.avg_param + 0.1 * flatten_params(model) if epoch > 0 else flatten_params(model)
        # ============ TensorBoard logging ============
        # (1) Log the scalar values
        info = {
            'loss': train_loss,
            "train_p_trig": train_p_trig,
            "train_r_trig": train_r_trig,
            "train_f1_trig": train_f1_trig,
            "train_p_arg": train_p_arg,
            "train_r_arg": train_r_arg,
            "train_f1_arg": train_f1_arg
        }

        for tag, value in info.items():
            writer.scalar_summary(tag, value, epoch + 1)

#         (2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            #             print(tag, "\n>", value)
            tag = tag.replace('.', '/')
            writer.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
            if value.grad is not None:
                writer.histo_summary(tag + '/grad',
                                     value.grad.data.cpu().numpy(), epoch + 1)

        if dev_dataset is None:
            logger.info("No devset")
        else:
            model.avg_param = 0.9 * model.avg_param + 0.1 * flatten_params(model) if not isinstance(model.avg_param, int) else flatten_params(model)
            dev_loss = test(dev_dataset, model, epoch + 1)
            dev_p_trig, dev_r_trig, dev_f1_trig, dev_p_arg, dev_r_arg, dev_f1_arg, dev_predictions_trigger, dev_predictions_argument, _ = accuracy("dev", dev_dataset, model.index, dev_loss, True, True, True)
            set_to_null(dev_dataset, model.index)

            # (3) Log the scalar dev values
            info = {
                'loss': dev_loss,
                "dev_p_trigger": dev_p_trig,
                "dev_r_trigger": dev_r_trig,
                "dev_f1_trigger": dev_f1_trig,
                "dev_p_arg": dev_p_arg,
                "dev_r_arg": dev_r_arg,
                "dev_f1_arg": dev_f1_arg,
                "dev_correct_trigger": dev_predictions_trigger[0],
                "dev_predicted_trigger": dev_predictions_trigger[1],
                "dev_correct_argument": dev_predictions_argument[0],
                "dev_predicted_argument": dev_predictions_argument[1],
                "train_correct_trigger": train_predictions_trigger[0],
                "train_predicted_trigger": train_predictions_trigger[1],
                "train_correct_argument": train_predictions_argument[0],
                "train_predicted_argument": train_predictions_argument[1]
            }

            if test_dataset is not None:
                test_loss = test(test_dataset, model, epoch + 1)
                test_p_trig, test_r_trig, test_f1_trig, test_p_arg, test_r_arg, test_f1_arg, test_predictions_trigger, test_predictions_argument, _ = accuracy("test", test_dataset, model.index, test_loss, True, True)
                set_to_null(test_dataset, model.index)

                info_test = {
                    "test_p_trigger": test_p_trig,
                    "test_r_trigger": test_r_trig,
                    "test_f1_trigger": test_f1_trig,
                    "test_p_arg": test_p_arg,
                    "test_r_arg": test_r_arg,
                    "test_f1_arg": test_f1_arg,
                    "test_correct_trigger": test_predictions_trigger[0],
                    "test_predicted_trigger": test_predictions_trigger[1],
                    "test_correct_argument": test_predictions_argument[0],
                    "test_predicted_argument": test_predictions_argument[1]
                }

                info = {**info, **info_test}

            print("max dev trigger f1 %.2f, max dev argument f1 %.2f" % (dev_f1_trigger_perf, dev_f1_arg_perf))

            for tag, value in info.items():
                writer.scalar_summary(tag, value, epoch + 1)

            results.append((epoch, (train_f1_trig, train_f1_arg),
                            (dev_f1_trig, dev_f1_arg)))

            dev_f1 = numpy.average([dev_f1_trig, dev_f1_arg])

            if dev_f1 > max_dev_f1:
                max_dev_f1 = numpy.average([dev_f1_trig, dev_f1_arg])
                max_dev_trigger_f1 = dev_f1_trig
                max_dev_trigger_p = dev_p_trig
                max_dev_trigger_r = dev_r_trig
                max_dev_argument_f1 = dev_f1_arg
                max_dev_argument_p = dev_p_arg
                max_dev_argument_r = dev_r_arg

                # save the model
                if model_save_path:
                    original_param = flatten_params(model)
                    logger.debug("saving model to " + model_save_path)
                    torch.save(model, model_save_path)
                    load_params(model, model.avg_param)
                    logger.debug("saving averaged model to " + model_save_path + ".avg")
                    torch.save(model, model_save_path + ".avg")
                    load_params(model, original_param)
                    logger.debug("pickling index")
                    with open(model_save_path + "_index", "wb") as f:
                        pickle.dump(model.index, f)
    if dev_dataset:
        logger.info("Maximum averaged dev f1: %.2f, trigger p %.2f r %.2f f1 %.2f, argument p %.2f r %.2f f1 %.2f" % (max_dev_f1, max_dev_trigger_p, max_dev_trigger_r, max_dev_trigger_f1, max_dev_argument_p, max_dev_argument_r, max_dev_argument_f1))
    else:
        # if we didn't set a devset but want to save the model, we save the latest version
        if model_save_path:
            torch.save(model, model_save_path)
            with open(model_save_path + "_index", "wb") as f:
                pickle.dump(model.index, f)

    return max_dev_f1, max_dev_trigger_f1, max_dev_argument_f1, results, model.avg_param


def clear_tree_variables(tree):
    '''
    Clears all variables which hold a point to the PyTorch graph. Do this at the end of an epoch to avoid GPU memory overflow. Some variables are kept as numpy arrays because their value may be needed further down the road.
    :param tree:
    '''
    tree.__dict__.pop("h_down", None)
    tree.__dict__.pop("c_down", None)
    tree.__dict__.pop("h_up", None)
    tree.__dict__.pop("c_up", None)
    tree.__dict__.pop("h", None)
    tree.__dict__.pop("ph", None)
    if "raw_output" in tree.__dict__:
        tree.__dict__.pop("raw_output", None)
    if "sentence" in tree.__dict__:
        tree.__dict__.pop("sentence", None)
    if "loss" in tree.__dict__:
        tree.__dict__.pop("loss", None)
    if "binary_loss" in tree.__dict__:
        tree.__dict__.pop("binary_loss", None)
    if "decoder_state" in tree.__dict__:
        tree.__dict__.pop("decoder_state", None)
    if "x" in tree.__dict__:
        tree.__dict__.pop("x", None)
    if "sentence_states" in tree.__dict__:
        tree.__dict__.pop("sentence_states", None)
    if "sentence_input_trigger" in tree.__dict__:
        tree.__dict__.pop("sentence_input_trigger", None)
    if "sentence_input_argument" in tree.__dict__:
        tree.__dict__.pop("sentence_input_argument", None)


def train_batched(dataset, model, epoch):
    '''
    Performs 'batched' training. The training is batched in the sense that
    losses are accumulated over multiple training points (sometimes an
    efficient way to overcome local minima). However, it is not batched
    in the sense that any parallel computation takes place. It is highly
    non-trivial to efficiently batch tree-shaped RNNs.
    '''
    model.train()
    loss = 0.0
    indices = [x for x in numpy.arange(len(dataset))]
    shuffle(indices)
    for i in tqdm(range(int(len(indices))), desc='Training'):

        idx = indices[i]
        graph = dataset[idx]
        r = model(graph, epoch)
        graph.reset()
        loss += r
    # this final call is to make an final update if the final batch is smaller
    # than the batch size
    r += model(None, epoch)
    return loss / len(dataset)


def test(dataset, model, epoch):
    model.eval()
    original_param = flatten_params(model)  # save current params
    load_params(model, model.avg_param)
    loss = 0
    for idx in tqdm(range(len(dataset)), desc='Testing '):
        graph = dataset[idx]
        loss += model(graph, epoch)
        graph.reset()
    load_params(model, original_param)
    return loss
