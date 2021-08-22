import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
import dominate
import datetime
from dominate.tags import script, span, div, style, br, p, img
from dominate.dom_tag import attr
import os
rcParams.update({'figure.autolayout': True})

from graphviz import Digraph
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import logging


logger = logging.getLogger("EventorVisualizeUtilities")


def visualize_backprop_graph(var, params, limit):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
        limit: A limit on how much the farthest node in the graph will be from
        the root (=loss) node
    """
    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var, limit, count):
        if limit is not None and count >= limit:
            return
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map[id(u)] if id(u) in param_map else "n/a", size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0], limit, count + 1)
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t, limit, count + 1)
    add_nodes(var.grad_fn, limit, 0)
    return dot


def visualize_one_change(matrix, tree, index, path, addPdfPostfix=False):

    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')

    heatmap = ax.pcolor(matrix, cmap="hot")

    # legend
    fig.colorbar(heatmap)

    def collect_labels(t):
        return (t.word, t.id, str(t.raw_output[0, t.output].data.cpu().exp().
                                  numpy()[0]) if
                hasattr(t, "raw_output") else "n/a")

    # axes
    labels = [*{*tree._traverse_from_top(collect_labels)}]

    labels = sorted(labels, key=lambda l:l[1])
    ax.set_xticks([x[1] + 0.5 for x in labels])
    ax.set_yticks([x[1] + 0.5 for x in labels])
    ax.set_xticklabels([x[0] + ":" + x[2] for x in labels], rotation=90)
    ax.set_yticklabels([x[0] + ":" + x[2] for x in labels])
    logger.debug("Writing to %s" % path)
    plt.savefig(path if not addPdfPostfix else path + ".pdf")
    g = tree.toGraph(index)
    g.write_svg(path + ".tree.svg")

    def write_raw_output(t, f):
        '''
        Writes the class probabilities of the node to the specified file 
        handler. Omits a node if it is not an argument node.
        :param t:
        :param f:
        '''
        if t.entity_type_index > 1:
            if hasattr(t, 'raw_output'):
                values = [(index.getArgumentType(x), t.raw_output[0, x].
                           data.cpu().exp().numpy()[0]) for x
                          in range(len(index.argument_index))]
            else:
                values = [[-1, -1]]
            values = sorted(values, key=lambda x: x[1], reverse=True)
            f.write("\n" + str(t) + "\n" + str(values))

    with open(path + ".probs.txt", "w") as f:
        tree._traverse_from_top(lambda t: write_raw_output(t, f))


def visualize_dependencies(weights, index, suf):
    sim = numpy.zeros((len(weights), len(weights)))
    for i in range(len(weights)):
        for j in range(i, len(weights)):
            a = weights[i].view(1, -1)
            b = weights[j].view(1, -1)
            cos = cosine_similarity(a, b)
            sim[i, j] = cos.squeeze()
    sim = sim.transpose()
    plt.close('all')

    # Make plot with vertical (default) colorbar
    fig, ax = plt.subplots()

    cax = ax.imshow(sim, interpolation="None", cmap=cm.coolwarm)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')

    # axes
    labels = [index.getDependency(x).replace('(', '<').replace(')', '>') for x in range(len(weights))]

    ax.set_xticks([x for x in range(len(weights))])
    ax.set_yticks([x for x in range(len(weights))])
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])

    plt.savefig("dependencies_%s" % (suf))


def write_trees(data, index, postfix):

    os.makedirs("trees/", exist_ok=True)

    for i in range(len(data)):
        if i > 100:
            break
        tree = data[i][0]
        event_type = tree.trigger._type_name
        graph = tree.toGraph(index)
        os.makedirs("trees/%s/%s" % (event_type, postfix), exist_ok=True)
        graph.write_svg("trees/%s/%s/%d.svg" % (event_type, postfix, i))


def htmlout(points, index, prf1, suf):
    '''
    :param points: The data points
    :param index:
    :param prf1:
    '''
    name = str(datetime.datetime.now())
    doc = dominate.document(name)
    css = '''
    .sentence {
        display: table;
    }

    .word {
        display: table-cell;
        padding-right:10px;
    }

    .correct {
        color: seagreen;
    }

    .wrong {
        color: crimson;
    }
    '''
    with doc.head:
        style(css)
    with doc:
        p('Trigger P %.3f R %.3f F1 %.3f' % (prf1[0], prf1[1], prf1[2]))
        p('Argument P %.3f R %.3f F1 %.3f' % (prf1[3], prf1[4], prf1[5]))
        for point in points:
            br()
            p(point.sid)
            _htmlout(point, index, doc, suf)
            br()
            br()
    return doc


def _htmlout(point, index, doc, suf):
    with div(cls='sentence'):
        with div(style='display:table-row;'):
            for i in range(point.sentenceSize()):
                with div("%s/%s" % (index.getWord(point.words[i]), index.getEntityType(point.bilou_entity_tag_indices[i])), cls='word'):
                    if not point.isPotentialTrigger(i):
                        cls = 'correct'
                        _is = 'n'
                        _should = 'n'
                        rel = '='
                    elif point.predicted_triggers is not None:
                        if point.predicted_triggers[i] == point.gold_triggers[i]:
                            cls = 'correct'
                            rel = '='
                        else:
                            cls = 'wrong'
                            rel = '!='
                        _is = index.getEventType(point.predicted_triggers[i]) if point.predicted_triggers[i] > 0 else "o"
                        _should = index.getEventType(point.gold_triggers[i]) if point.gold_triggers[i] > 0 else "o"
                    else:
                        cls = "correct"
                        _is = 'n/a'
                        _should = 'n/a'
                        rel = '='
                    div("%s%s%s" % (_is, rel, _should), cls=cls)

    with div("Arguments:", cls='arguments'):
        if point.predicted_arguments_by_trigger is not None:
            for wid in range(point.sentenceSize()):
                word = index.getWord(point.words[wid])

                predicted_arguments = point.predicted_arguments_by_trigger[wid] if wid in point.predicted_arguments_by_trigger else {}
                gold_arguments = point.gold_arguments_by_trigger[wid] if wid in point.gold_arguments_by_trigger else {}
                if len(predicted_arguments) == 0 and len(gold_arguments) == 0:
                    continue
                else:
                    with div("trigger %d (%s)" % (wid, word), cls="correct" if point.triggerPredictionIsCorrect(wid) else "wrong"):
                        mids = list(predicted_arguments.keys()) + list(gold_arguments.keys())
                        for mid in set(mids):
                            mention = point.mentions[mid]
                            predicted_roles = [index.getArgumentType(x) for x in predicted_arguments[mid]] if mid in predicted_arguments else [index.getArgumentType(0)]
                            gold_roles = [index.getArgumentType(x) for x in gold_arguments[mid]] if mid in gold_arguments else [index.getArgumentType(0)]
                            cls = 'wrong'
                            rel = '!=' if point.triggerPredictionIsCorrect(wid) else '?='
                            for role in predicted_roles:
                                if role in gold_roles:
                                    cls = 'correct'
                                    rel = '='
                                    break
                            div("%s (%s): p %s %s g %s" % (str(mention), str(point.getTextForMention(mention)), str(predicted_roles), rel, str(gold_roles)), cls=cls, style="padding-left:50px;")
#                             div("%s" % (str(point.mention_features[mid][0][wid])))
    graph = point.toGraph(index)
    os.makedirs("graphs", exist_ok=True)
    svg_name = "graphs/%s.svg" % (point.sid)
    graph.write_svg(svg_name)
    br()
    br()
    br()
    img(src=svg_name)

