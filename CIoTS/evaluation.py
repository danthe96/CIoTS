import numpy as np
import pandas as pd
import networkx as nx
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score, \
                            matthews_corrcoef, \
                            f1_score


def _equalize_nodeset(true_graph, pred_graph):
    true_nodes = set(true_graph.nodes())
    pred_nodes = set(pre_graph.nodes)
    if true_nodes.issubset(pred_nodes):
        node_diff = pred_nodes.difference(true_nodes)
        true_graph.add_nodes_from(list(node_diff))
    elif pred_nodes.issubset(true_nodes):
        node_diff = true_nodes.difference(pred_nodes)
        pre_graph.add_nodes_from(list(node_diff))
    else:
        raise StandardError("incomparable graphs")
    return true_graph, pred_graph


def evaluate_edges(true_graph, pred_graph, directed=True):
    true_graph, pred_graph = _equalize_nodeset(true_graph, pred_graph)
    truth = []
    prediction = []
    if directed:
        for u, v in permutations(true_graph.nodes(), 2):
            truth.append(true_graph.has_edge(u, v))
            prediction.append(prediction.has_edge(u, v))
    else:
        for u, v in combinations(true_graph.nodes(), 2):
            truth.append(true_graph.has_edge(u, v) or true_graph.has_edge(v, u))
            prediction.append(prediction.has_edge(u, v) or prediction.has_edge(v, u))
    return {'accuracy': accuracy_score(truth, prediction),
            'f1-score': f1_score(truth, prediction),
            'matthews_corrcoef': matthews_corrcoef(truth, prediction)}
