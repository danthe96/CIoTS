from itertools import permutations, combinations, product
import numpy as np
from sklearn.metrics import accuracy_score, \
                            matthews_corrcoef, \
                            f1_score, \
                            precision_score, \
                            recall_score, \
                            confusion_matrix
from copy import deepcopy
import pandas as pd


def _equalize_nodeset(true_graph, pred_graph):
    true_nodes = set(true_graph.nodes())
    pred_nodes = set(pred_graph.nodes)
    if true_nodes.issubset(pred_nodes):
        node_diff = pred_nodes.difference(true_nodes)
        true_graph.add_nodes_from(list(node_diff))
    elif pred_nodes.issubset(true_nodes):
        node_diff = true_nodes.difference(pred_nodes)
        pred_graph.add_nodes_from(list(node_diff))
    else:
        raise Exception("incomparable graphs")
    return true_graph, pred_graph


def _fpr(truth, prediction):
    truth = np.array(truth).astype(bool)
    prediction = np.array(prediction).astype(bool)
    fp = prediction & (~truth)
    return fp.sum()/(~truth).sum()


def evaluate_edges(true_g, pred_g, directed=True):
    true_graph, pred_graph = _equalize_nodeset(deepcopy(true_g),
                                               deepcopy(pred_g))
    truth = []
    prediction = []
    if not directed:
        true_graph = true_graph.to_undirected()
        pred_graph = pred_graph.to_undirected()
        node_pairs = [(u, v) for (u, v) in combinations(true_graph.nodes(), 2)]
    else:
        node_pairs = [(u, v) for (u, v) in permutations(true_graph.nodes(), 2)]
    for u, v in node_pairs:
        truth.append(true_graph.has_edge(u, v))
        prediction.append(pred_graph.has_edge(u, v))
    return {'accuracy': accuracy_score(truth, prediction),
            'f1-score': f1_score(truth, prediction),
            'precision': precision_score(truth, prediction),
            'FDR': 1 - precision_score(truth, prediction),
            'TPR': recall_score(truth, prediction),
            'FPR': _fpr(truth, prediction),
            'matthews_corrcoef': matthews_corrcoef(truth, prediction)}


def _graph_confusion_matrix(true_graph, pred_graph, node_pairs=None):
    if node_pairs is None:
        node_pairs = [(u, v) for (u, v) in permutations(true_graph.nodes(), 2)]
    truth = []
    prediction = []
    for u, v in node_pairs:
        truth.append(true_graph.has_edge(u, v))
        prediction.append(pred_graph.has_edge(u, v))
    c_matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    return {'tp': c_matrix[1, 1], 'tn': c_matrix[0, 0],
            'fp': c_matrix[0, 1], 'fn': c_matrix[1, 0]}

def evaluate_edge_deletion(true_g, iterations, dim):
    confusion = []
    confusion_delta = []
    # for each iteration calculate confusion
    prev_nodes = set()
    for iteration in iterations:
        # extend original graph
        if len(true_g.nodes) < len(iteration['graph'].nodes):
            true_graph, graph = _equalize_nodeset(deepcopy(true_g),
                                                  deepcopy(iteration['graph']))
        # shrink original graph
        else:
            graph = deepcopy(iteration['graph'])
            true_graph = true_g.subgraph(graph.nodes)
        # only consider existence of edge, not direction
        true_graph = true_graph.to_undirected()
        graph = graph.to_undirected()
        # connections with new nodes
        current_nodes = list(graph.nodes)
        nodes_delta = list(set(current_nodes[0:dim]).union(set(current_nodes).difference(prev_nodes)))
        # possible node pairs
        node_pairs = product(current_nodes[:dim], current_nodes[dim:])
        node_pairs_delta = product(nodes_delta[:dim], nodes_delta[dim:])
        # write results
        iter_confusion = _graph_confusion_matrix(true_graph, graph, node_pairs)
        iter_confusion['p_iter'] = iteration['p_iter']
        confusion.append(iter_confusion)
        iter_confusion_delta = _graph_confusion_matrix(true_graph.subgraph(nodes_delta),
                                                       graph.subgraph(nodes_delta),
                                                       node_pairs_delta)
        iter_confusion_delta['p_iter'] = iteration['p_iter']
        confusion_delta.append(iter_confusion_delta)
        # update prev nodes for delta calculation
        prev_nodes = set(deepcopy(graph.nodes))
    return pd.DataFrame(confusion), pd.DataFrame(confusion_delta)
