import pydot
import torch
from torch.autograd.variable import Variable
from utils.utils import singleValueToLongTensor


class Tree(object):
    '''
    This class is the backbone of Eventor. It represents a tree node and holds
    its children, along with additional information for the node.
    '''
    def __init__(self, idx=None, word=None, word_index=None,
                 pos=None, pos_index=None, entity_type=None,
                 entity_type_index=None, _type=None, _type_name=None,
                 _id=None, begin=None, end=None, sentence=None,
                 distances=None):
        self.id = idx
        self.word = word
        self.word_index = word_index
        self.entity_type = entity_type
        self.entity_type_index = entity_type_index
        self._type = _type
        self.parent = None
        self.num_children = 0
        self._type_name = _type_name
        self.children = list()
        self.dependencies = list()
        self.dependency_names = list()
        self.is_root = False
        self.id = _id
        if self.entity_type_index > 0:
            self.output = 1
        else:
            self.output = None
        self.begin = begin
        self.end = end
        self.sentence = sentence
        self.distances = distances
        self.pos = pos
        self.pos_index = pos_index

    def add_child(self, child, dependency, dependency_name):
        if child != self:
            child.parent = self
            child.parent_dependency = dependency
        self.num_children += 1
        self.children.append(child)
        self.dependencies.append(dependency)
        self.dependency_names.append(dependency_name)

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def getBiasVector(self, ent_emb, ge_emb, pos_emb, cudaFlag):
        entity_embedding = ent_emb(Variable(singleValueToLongTensor(
            0 if self.entity_type_index is None else self.entity_type_index,
            cudaFlag)))
        genre_embedding = ge_emb(Variable(singleValueToLongTensor(self.genre,
                                                                  cudaFlag)))
        return torch.cat([entity_embedding, genre_embedding], 1)

    def distanceToRoot(self):
        if hasattr(self, '_dist_to_root'):
            return self._dist_to_root
        self._dist_to_root = 0
        parent = self.parent
        while parent:
            self._dist_to_root += 1
            parent = parent.parent
        return self._dist_to_root

    def getTrigger(self):
        return self.trigger

    def getMaxDepth(self):
        if getattr(self, "maxDepth", None):
            return self.maxDepth
        else:

            def gd(tree, md):
                '''
                Recursively computes the depth of a tree.
                :param tree:
                :param md:
                '''
                md += 1
                if tree.num_children > 0:
                    for i in range(tree.num_children):
                        md = max([md, gd(tree.children[i], md)])
                return md

            self.maxDepth = gd(self, -1)
            return self.maxDepth

    def getAllTrees(self):
        return [x for x in self._traverse(self, lambda x:x)]

    def getArgumentNodes(self):
        return [x for x in self._traverse(self, lambda x:x) if not x.is_trigger and x._type > 1]

    def getMentionNodes(self):
        return [x for x in self._traverse(self, lambda x:x) if not x.is_trigger and x._type > 0]

    def getNonNATrees(self):
        return [x for x in self._traverse(self, lambda x:x) if x.is_trigger or x._type != 0]

    def getTreesWithPredictions(self):
        # all nodes which have no "n/a" type (intermediate nodes) have to have outputs and are returned here
        return [x for x in self._traverse(self, lambda x:x) if x.is_trigger or x._type != 0]

    def getFlatPredictions(self):
        return [int(x.output) for x in self.getTreesWithPredictions()]

    def getPredictionTuples(self, index):
        def assert_mention_has_outputs(tree):
            if tree.entity_type_index > 0:
                if not hasattr(tree, "output"):
                    if tree.is_trigger:
                        tree.output = index.addEventType("No.Event")
                    else:
                        tree.output = index.addArgumentType("null")
                assert hasattr(tree, "output")
                assert tree.output is not None
        self._traverse_from_top(assert_mention_has_outputs)
        trigger_output = int(self.trigger.output)
        return [(x.begin, x.end, trigger_output, int(x.output) if x.output
                 is not None else -1,
                 self.trigger.begin, self.trigger.end, x.is_trigger) for x in
                self.getTreesWithPredictions()]

    def getLabels(self):
        return [x._type for x in self.getTreesWithPredictions() if x._type > 0]

    def getEntityTypes(self):
        return [x.entity_type for x in self.getTreesWithPredictions() if x.entity_type != "null"]

    def _traverse_from_top(self, f):
        return self._traverse(self, f)

    def leads_to_argument(self):
        if not hasattr(self, 'leads_to_arg'):
            leads_to_arg = {x for x in self._traverse(self, lambda x: x is not x.trigger and x._type > 1)}
            self.leads_to_arg = True in leads_to_arg
        return self.leads_to_arg

    def toGraph(self, index):
        graph = pydot.Dot(graph_type='graph')

        def add_to_graph(tree):

            if hasattr(tree, "output") and tree.output is not None:
                argm = int(tree.output)
                if tree.is_trigger:
                    prediction = index.getEventType(argm)
                else:
                    prediction = index.getArgumentType(argm)
            else:
                prediction = "n/a"
            label = tree._type_name
            fillcolor = "blue" if not tree.forprediction else ("green" if label == prediction else "red")
            pencolor = "blue" if tree.entity_type_index > 1 else "black"
            tree.pydot_node = pydot.Node("%s_%d-%s-%s" % (tree.word, tree.id, prediction, label), fillcolor=fillcolor, pencolor=pencolor, style="filled")
            graph.add_node(tree.pydot_node)
            for cid in range(tree.num_children):
                child = tree.children[cid]
                dependency_name = tree.dependency_names[cid].replace(":", "-")
                add_to_graph(child)
                edge = pydot.Edge(tree.pydot_node, child.pydot_node, label=dependency_name)
                graph.add_edge(edge)
        add_to_graph(self)
        return graph

    def _traverse(self, tree, f):
        _list = []
        _list += [f(tree)]
        for child in tree.children:
            child_list = self._traverse(child, f)
            _list += child_list
        return _list

    def __str__(self, *args, **kwargs):
        return "%s (%d|%d)" % (self.word, self._type, self.output if hasattr(self, 'output') else -1)
