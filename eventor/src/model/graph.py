'''
Created on Feb 23, 2018

@author: judeaax
'''
from utils.utils import toLongTensor, print_sizes, singleValueToLongTensor, \
    null_var
from torch.autograd import variable
from torch.autograd.variable import Variable
import torch
import logging
import re
import mmh3
from model import config
import pydot
from itertools import groupby


def _handle_role(role):
    '''
    A common hub to handle role labels.
    :param role: The label
    '''
    if role.startswith("Time"):
        return "Time"
    else:
        return role


class SentenceGraph(object):
    '''
    classdocs
    '''

    logger = logging.getLogger("Eventor:graph")

    potential_trigger_re = re.compile("")

    def __init__(self, values, index):
        '''
        Constructor
        '''
        # #1 document id, #2 sentence id, #3 genre, #4 words, #5 triggers, #6 arguments, #7 mentions
        self.did = values[0]
        self.sid = values[1]
        self.genre = index.addGenre(values[2])
        self._make_words(values[3], index)
        self._make_triggers(values[4], index)
        self._make_mentions(values[6], index, self.sentenceSize())
        self._make_arguments(values[5], index)
        self.index = index
        self.valid_trigger_pos = re.compile("IN|JJ|RB|VBG|VBD|NN|NNPS|VB|VBN|NNS|VBP|NNP|PRP|VBZ", re.I)

        def potential_trigger(i):
            '''
            Returns true if the trigger at the specified position is a potential trigger.
            :param i: Some word index.
            '''
            return i not in self.mention_indices and re.match(self.valid_trigger_pos, self.index.getPos(self.pos_tags[i])) != None

        self.potential_trigger_list = [potential_trigger(i) for i in range(self.sentenceSize())]
        for i in range(len(self.potential_trigger_list)):
            for j in range(len(self.mentions)):
                features, indices = self.mention_features[j]
                features[i] = None
                if not potential_trigger(i):
                    indices[i] = None

    def _make_mention_features(self, feature_indices):
        values = feature_indices.split("/#/")
        feature_indices = []
        features = []
        for i in range(len(values)):
            feature_indices.append(torch.LongTensor([self._hash(x) for x in values[i].split("/+/")]))
            features.append([x for x in values[i].split("/+/")])
        return (features, feature_indices)

    def _make_mentions(self, data, index, sentence_len):
        '''
        Given data, constructs all mentions therein.
        :param data:
        :param index:
        :param sentence_len:
        '''

        values = data.split("|")
        self.mentions = []
        self.mention_indices = set()
        self.mention_features = []
        for value in values:
            if len(value) == 0:
                continue
            begin, end, idx, _type, _subtype, features, mention_id, entity_id = value.split(",")
            assert len(self.mentions) == int(idx), "%s: %s!=%s\n%s" % (value, len(self.mentions), idx, str(self.mentions))
            self.mentions.append((int(begin), int(end), index.addEntityType(_type), _type, int(idx), index.addEntityType(_subtype), _subtype, mention_id, entity_id))
            if _type != "Crime" and _type != 'Job-Title' and _type != 'Sentence' and not _type.startswith("Time") and _type != 'Numeric':
                for span_index in range(int(begin), int(end) + 1):
                    # the mention indices will be used to skip trigger prediction at those word indices
                    self.mention_indices.add(span_index)
            mention_features = self._make_mention_features(features)
            assert len(mention_features[0]) == sentence_len
            assert len(mention_features[0]) == len(mention_features[1])
            self.mention_features.append(mention_features)
            assert len(self.mentions) == len(self.mention_features)

    def _make_arguments(self, data, index):
        values = data.split("|")
        self.gold_arguments_by_trigger = {}
        for value in values:
            if len(value) == 0:
                continue
            mention_index, trigger_index, role = value.split(",")
            mention_index = int(mention_index)
            trigger_index = int(trigger_index)
            if trigger_index not in self.gold_arguments_by_trigger:
                self.gold_arguments_by_trigger[trigger_index] = {}
            if mention_index not in self.gold_arguments_by_trigger[trigger_index]:
                self.gold_arguments_by_trigger[trigger_index][mention_index] = set()
            self.gold_arguments_by_trigger[trigger_index][mention_index].add(index.addArgumentType(_handle_role(role)))

        entitites = {x: list(y) for x, y in groupby(self.mentions, lambda x: x[8])}
        for wid, subdict in self.gold_arguments_by_trigger.items():
            nargs = {}
            for mid, roles in subdict.items():
                mentions = [x for x in entitites[self.mentions[mid][8]] if self.mentions[mid][4] != mid]
                for mention in mentions:
                    nargs[mention[4]] = roles
            ndict = {**nargs, **subdict}
            self.gold_arguments_by_trigger[wid] = ndict

    def _make_triggers(self, data, index):
        triggers = data.split("|")
        self.gold_triggers = [0] * len(self.words)

        for trigger in triggers:
            if len(trigger) == 0:
                continue
            word_id, _type = trigger.split(":")
            self.gold_triggers[int(word_id)] = index.addEventType(_type)

    def _hash_node(self, feature):
        usigned = mmh3.hash(feature, signed=False)
        return usigned % config.trigger_hash_dimensions

    def _hash(self, feature):
        usigned = mmh3.hash(feature, signed=False)
        return usigned % config.hash_dimensions

    def _make_feature_vector(self, idx):
        vector = torch.zeros(config.trigger_hash_dimensions)
        if len(self.feature_indices[idx]) > 0:
            indices = torch.LongTensor(self.feature_indices[idx])
            vector[indices] = 1
        return vector

    def _make_words(self, data, index):
        values = data.split(">>")
        self.words = []
        self.lemmas = []
        self.pos_tags = []
        self.bilou_entity_tags = []
        self.features = []
        self.feature_indices = []
        self.onehot_node_feature_vectors = torch.zeros(len(values), config.trigger_hash_dimensions)
        self.adj_matrix = torch.eye(len(values), len(values))
        # self.adj_matrix_labels = torch.zeros(len(values), len(values))
        self.adj_matrix_dependencies = torch.eye(len(values), 1000)
        for i in range(len(values)):
            value = values[i]
            if len(value.split('|')) != 6:
                print('\n'.join(value.split('|')))
            word, lemma, pos, relations, bilou_entity_tag, features = value.split("|")
            self.words.append(index.addWord(word.lower().replace(',', '.')))
            self.lemmas.append(index.addWord(lemma.lower()))
            self.pos_tags.append(index.addPos(pos))
            self.bilou_entity_tags.append(index.addEntityType(bilou_entity_tag))
            self.features.append([x for x in features.split("/+/")])  # if x.startswith("NodeHU")
            self.feature_indices.append([self._hash_node(x) for x in self.features[-1]])
            self.onehot_node_feature_vectors[i] = self._make_feature_vector(i)
            relation_values = relations.split("=")

            for relation_value in relation_values:
                if relation_value != 'null':
                    neighbor_id, relation = relation_value.split("+")
                    self.adj_matrix[i][int(neighbor_id)] = 1
                    dep_id = index.addDependency(relation.replace('(-', '').replace('-)', ''))
                    self.adj_matrix_dependencies[i][dep_id] = 1

    def getEventPredictions(self):
        return '%s\t%s' % (self.sid, self._prepare_event_predictions())

    def _prepare_event_predictions(self):
        predictions = []
        for i in range(self.sentenceSize()):
            etype = self.index.getEventType(self.predicted_triggers[i])
            if self.predictionIsEvent(i) and i not in self.predicted_arguments_by_trigger:
                predictions.append("%s+%d" % (etype, i))
        for wid, subdict in self.predicted_arguments_by_trigger.items():
            etype = self.index.getEventType(self.predicted_triggers[wid])
            arguments = []
            for mid, roles in subdict.items():
                mention = self.mentions[mid]
                for role in roles:
                    arguments.append("%s:%s" % (mention[7], self.index.getArgumentType(role)))
            predictions.append("%s+%d+%s" % (etype, wid, "|".join(arguments)))
        return "\t".join(predictions)

    def setPredictionsToGold(self):
        # set trigger predictions to gold
        self.predicted_triggers = [x for x in self.gold_triggers]
        # set argument predictions to gold
        self.predicted_arguments_by_trigger = {}
        for trigger, a in self.gold_arguments_by_trigger.items():
            self.predicted_arguments_by_trigger[trigger] = {}
            for mention, role in a.items():
                self.predicted_arguments_by_trigger[trigger][mention] = role

    def triggerPredictionIsCorrect(self, i):
        return self.predicted_triggers[i] == self.gold_triggers[i]

    def getPredictedArgumentsOf(self, i):
        if self.predicted_arguments_by_trigger is None:
            return None
        return self.predicted_arguments_by_trigger[i] if i in self.predicted_arguments_by_trigger else None

    def getGoldArgumentsOf(self, i):
        return self.gold_arguments_by_trigger[i] if i in self.gold_arguments_by_trigger else None

    def getGoldArgumentsByIds(self, i, j):
        gold_arguments = self.getGoldArgumentsOf(i)
        if gold_arguments is not None:
            if j in gold_arguments:
                return gold_arguments[j]
        return [0]

    def getGoldArgumentsByPredictedEventTypes(self, i, j):
        if self.triggerPredictionIsCorrect(i):
            return self.getGoldArgumentsByIds(i, j)
        else:
            return [0]

    def goldTriggerIsEvent(self, i):
        return self.gold_triggers[i] > 0

    def predictionIsEvent(self, i):
        return self.predicted_triggers[i] > 0

    def getGoldTrigger(self, i):
        return self.gold_triggers[i]

    def correctArgumentPrediction(self, wid, mention_id, role):
        if isinstance(role, set) or isinstance(role, list):
            for r in role:
                assert r > 0, "%s trigger %d mention %d roles %s" % (self.sid, wid, mention_id, str(role))
        else:
            assert role > 0
            role = [role]
        golds = self.getGoldArgumentsOf(wid)
        if golds is not None and mention_id in golds:
            gold_relations = golds[mention_id]
            if isinstance(gold_relations, int):
                gold_relations = [gold_relations]
            for r in role:
                if r in gold_relations:
                    return True
        return False

    def sentenceSize(self):
        return len(self.words)

    def isPotentialTrigger(self, i):
        return self.potential_trigger_list[i]

    def argumentCandidatesSize(self):
        return len(self.mentions)

    def getEntityTypeForMentionIndex(self, j):
        return self.getEntityTypeForMention(self.mentions[j])

    def getEntityTypeForMention(self, mention):
        return mention[3]

    def getTextForMentionIndex(self, j):
        return self.getTextForMention(self.mentions[j])

    def getPredictedEventTypeForWordIndex(self, i):
        return self.index.getEventType(self.predicted_triggers[i])

    def getTextForMention(self, mention):
        return " ".join([self.index.getWord(x) for x in self.words[mention[0]:mention[1] + 1]])

    def reset(self):
        if hasattr(self, 'word_feature_abstractions'):
            del self.word_feature_abstractions

    def toGraph(self, index):
        '''
        Transforms this graph instance into a PyDot graph
        :param index:
        '''
        graph = pydot.Dot(graph_type='graph')
        node_list = []
        for i in range(self.sentenceSize()):
            node_label = "%s (%s)" % (index.getWord(self.words[i]), index.getPos(self.pos_tags[i]))
            # add the word as a node
            node = pydot.Node(node_label)
            graph.add_node(node)
            node_list.append(node)
        already_added_edges = set()
        for i in range(self.sentenceSize()):
            # TODO this has to be replaced, it won't work with the new
            # structure
            # then add the dependency edges as edges
            for neighbor in self.adj_matrix[i].nonzero():
                nid = int(neighbor[0])
                depid = int(self.adj_matrix_labels[i][nid])
                if (i, nid) in already_added_edges:
                    continue
                label = index.getDependency(depid).replace("(-", " ").replace("-)", " ").replace(":", "-")
                edge = pydot.Edge(node_list[i], node_list[nid], label=label)
                already_added_edges.add((i, nid))
                already_added_edges.add((nid, i))
                graph.add_edge(edge)
        return graph

    def rememberTriggerPrediction(self):
        self.remembered_trigger_prediction = list(self.predicted_triggers)

    def hasRememberedTriggerPrediction(self):
        return hasattr(self, 'remembered_trigger_prediction')

    def setTriggerPredictionToRemembered(self):
        self.predicted_triggers = list(self.remembered_trigger_prediction)

    def _distance_to_trigger(self, i, mention):
        begin, end = mention[:2]
        if begin > i:
            return begin - i
        elif end < i:
            return i - end
        else:
            return 0

    def _get_event_supertype(self, triggers, i):
        et = self.index.getEventType(triggers[i])
        return et.split('.')[0]

    def getEventSupertype(self, i):
        return self._get_event_supertype(self.predicted_triggers, i)

    def getGoldEventSupertype(self, i):
        return self._get_event_supertype(self.gold_triggers, i)

    def compactifyArguments(self):
        newdict = {}
        for wid, subdict in self.predicted_arguments_by_trigger.items():
            mentions_and_roles = [(self.mentions[x], subdict[x]) for x in subdict.keys()]
            group_by_id = groupby(mentions_and_roles, lambda x: x[0][8])
            for eid, m_r in group_by_id:
                # coreferent mentions
                m_r = list(m_r)
                if len(m_r) > 1:
                    # sort coreferent mentions by their distance to the trigger
                    m_r = sorted(m_r, key=lambda x: self._distance_to_trigger(wid, x[0]))
                    for mention, roles in m_r:
                        if len([x for x in roles if x > 0]) > 0:
                            if wid not in newdict:
                                newdict[wid] = {}
                            newdict[wid][mention[4]] = roles
                            break
                else:
                    if wid not in newdict:
                        newdict[wid] = {}
                    newdict[wid][m_r[0][0][4]] = m_r[0][1]
        self.predicted_arguments_by_trigger = newdict