'''
Created on Apr 28, 2017

@author: judeaax
'''
from torch.utils import data
import logging
from model import graph, config
from model.graph import SentenceGraph, _handle_role
import torch

logger = logging.getLogger("Eventor.load")


class Data(data.Dataset):
    def __init__(self):
        super(Data, self).__init__()
        self.graphs = []

    def __len__(self):
        return len(self.graphs)

    def add(self, graph):
        self.graphs.append(graph)

    def keep(self, n):
        self.graphs = self.graphs[:n + 1]

    def index(self, obj):
        return self.graphs.index(obj)

    def delete(self, index):
        del self.graphs[index]

    def delete_obj(self, obj):
        index = self.index(obj)
        self.delete(index)

    def __getitem__(self, index):
        return self.graphs[index]


def load_eval(infile_eval, index):
    '''
    Loads the evaluation data with the following format: eval_(trigger|argument)_data = {sid=[eval_tuple]}, eval_tuple=[begin offset, end offset,gold type, gold type or event type FOR ARGUMENTS, begin offset, end offset]. eval_tuple contains redundant information because it has to match the format of evaluation tuples later on.
    :param infile_eval:
    :param index:
    '''
    eval_trigger_data = {}
    eval_arg_data = {}
    triggers = 0
    arguments = 0
    with open(infile_eval, "r") as file:
        # get the lines
        lines = file.read().splitlines()
        for line in lines:
            # get the values
            values = line.split("|")
            # get the sentence id
            sid = values[0]
            if sid not in eval_trigger_data:
                eval_trigger_data[sid] = []
                eval_arg_data[sid] = []
            # first we deal with triggers
            persentence_trigger = eval_trigger_data[sid]
            trigger_values = values[1].split(" ")
            trigger = (int(trigger_values[0]), int(trigger_values[1]), index.addEventType(trigger_values[3]), index.addEventType(trigger_values[3]), int(trigger_values[0]), int(trigger_values[1]))
            triggers += 1
            persentence_trigger.append(trigger)
            # now we add the arguments
            persentence_arguments = eval_arg_data[sid]
            for i in range(2, len(values)):
                val = values[i].split(" ")
                enid = index.addEntityType(val[3])
                # begin offset    end offset    trigger type    argument type    trigger begin offset    trigger end offset
                argument = (int(val[0]), int(val[1]), trigger[2], index.addArgumentType(_handle_role(val[4])), trigger[0], trigger[1])
                arguments += 1
                persentence_arguments.append(argument)
    logger.debug("Loaded evaluation data for %d triggers and %d arguments" % (triggers, arguments))
    return (eval_trigger_data, eval_arg_data)


def process_line(line, index):
    parts = line.split("\t")
    assert len(parts) == 7
    graph = SentenceGraph(parts, index)
    return graph


def load(infile, infile_eval, index):
    data = Data()

    with open(infile, 'r') as f:
        lines = f.read().splitlines()

        for i in range(len(lines)):
            line = lines[i]
            graph = process_line(line, index)

            if graph is None:
                continue

            data.add(graph)
            position = ((i + 1) / len(lines)) * 100
            if position % 25 == 0:
                print("\rLoading data (%d)" % (int(position)), end='')

    data.eval_triggers, data.eval_arguments = load_eval(infile_eval, index)

    N_classes_argument = len(index.argument_index)
    N_classes_trigger = len(index.event_index)

    return data, N_classes_argument, N_classes_trigger


def buildRestrictions(data, index, N_classes_argument):
    # here we build the entity type restrictions for argument prediction
    restrictions = {}
    for graph in data:
        for wid, subdict in graph.gold_arguments_by_trigger.items():
            event_type = graph.gold_triggers[wid]
            if event_type not in restrictions:
                restrictions[event_type] = {}
            for mid, rids in subdict.items():
                mention = graph.mentions[mid]
                # mention[5] hold the entity subtype index
                if mention[2] not in restrictions[event_type]:
                    restrictions[event_type][mention[2]] = set()
                restrictions[event_type][mention[2]].update(rids)
    index.restrictions = restrictions
    restriction_masks = {}
    for event_id, subdict in restrictions.items():
        if event_id not in restriction_masks:
            restriction_masks[event_id] = {}
        for subtype_id, roles in subdict.items():
            vector = torch.zeros(1, N_classes_argument).fill_(float('-inf'))
            for r in roles:
                vector[0, r] = 0.0
            # we also want to allow the null type
            vector[0, 0] = 0.0
            restriction_masks[event_id][subtype_id] = vector
    # the null event type does not allow for any arguments
    restriction_masks[0] = {x: torch.zeros(1, N_classes_argument).fill_(float('-inf')) for x in index.inverse_entity_index.keys()}
    index.restriction_masks = restriction_masks
