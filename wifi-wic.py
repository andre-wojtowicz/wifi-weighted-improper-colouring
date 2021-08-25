#!/usr/bin/env python3

import argparse
import copy
import numpy as np
import pandas as pd
import openpyxl
import networkx as nx
import matplotlib.pyplot as plt
import time


def total_cost(G):

    return sum(G[e[0]][e[1]]["weight"] for e in G.edges if G.nodes[e[0]]["channel"] == G.nodes[e[1]]["channel"])
    
def max_edge_cost(G):

    try:
        return max(G[e[0]][e[1]]["weight"] for e in G.edges if G.nodes[e[0]]["channel"] == G.nodes[e[1]]["channel"])
    except ValueError:
        return 0


def interference(G, v):

    return sum(G[v][u]["weight"] for u in G.neighbors(v) if G.nodes[v]["channel"] == G.nodes[u]["channel"])


def potential_interference(G, v, i):

    return sum(G[v][u]["weight"] for u in G.neighbors(v) if G.nodes[u]["channel"] == i)
    
    
def total_potential_interference(G, v, k):

    return sum(potential_interference(G, v, i) for i in range(k))


def levelling_heuristic(G, k, t, static_aps=None, forbidden_channels=None):

    G = copy.deepcopy(G)
    nx.set_node_attributes(G, dict(zip(G, [None]*len(G))), "channel")
    if static_aps:
        for ap in static_aps:
            G.nodes[ap]["channel"] = static_aps[ap]
    else:
        static_aps = {}
    
    for _ in range(len(G)-len(static_aps)):

        uncolored = [v for v in G if G.nodes[v]["channel"] is None]
        tpis = dict(zip(uncolored, [total_potential_interference(G, v, k) for v in G if v in uncolored]))
        min_tpi = min(tpis.values())
        vs = [v for v, val in tpis.items() if val==min_tpi]
        if len(vs) == 1:
            v = vs[0]
        else:
            wds = dict(zip(uncolored, [len(G[v])*sum(e["weight"] for e in G[v].values()) for v in uncolored]))
            v = max(wds, key=wds.get)
        
        possible_colors = range(k)
        if forbidden_channels and v in forbidden_channels:
            possible_colors = list(set(possible_colors) - set(forbidden_channels[v]))
        pis = dict(zip(possible_colors, [potential_interference(G, v, i) for i in possible_colors]))
        xs = sorted(pis, key=pis.get)
        
        for x in xs:
            if max(potential_interference(G, v, x) for v in G) < t:
                G.nodes[v]["channel"] = x
                break
                
        if G.nodes[v]["channel"] is None:
            return None
            
    new_t = max(interference(G, v) for v in G)
        
    return G, new_t


def main(channels, static_aps, forbidden_channels, input_file, max_time, seed):

    M = pd.read_excel(input_file, index_col=0).fillna(0)
    apnames = M.columns.values.tolist()
    
    if type(channels) == str:
        channels = eval(channels)
        
    if type(static_aps) == str:
        static_aps = eval(static_aps)
        for ap in static_aps:
            static_aps[ap] = [i for i,v in enumerate(channels) if v==static_aps[ap]][0]
            
    if type(forbidden_channels) == str:
        forbidden_channels = eval(forbidden_channels)
        for ap in forbidden_channels:
            forbidden_channels[ap] = [[i for i,v in enumerate(channels) if v==c][0] for c in forbidden_channels[ap]]

    k = len(channels)
    G = nx.from_numpy_matrix(np.matrix(M))
    G = nx.relabel_nodes(G, dict(zip(G, apnames)))

    if seed:
        np.random.seed(seed)

    start_time = time.time()

    t = float("Inf")
    while True:

        result = levelling_heuristic(G, k, t, static_aps, forbidden_channels)
        if result is None:
            break
        G, t = result
        print("t = {}".format(t))
        
        if time.time() - start_time > max_time or t == 0:
            break

    print("\nResults:\n")

    for v in G:
        print("{} - {}".format(v, channels[G.nodes[v]["channel"]]))

    print("\nMax edge cost: {}".format(max_edge_cost(G)))
    print("Max node cost: {}".format(t))
    print("Total cost: {}".format(total_cost(G))) 


if __name__ == "__main__":

    def_channels_24 = (1, 6, 11)
    def_channels_50 = (36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128)

    parser = argparse.ArgumentParser(description='Calculates Wi-Fi channels setup through Weighted Improper Colouring',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t24', '--type-24', action='store_true', help='Calculate for 2.4 GHz')
    group.add_argument('-t50', '--type-50', action='store_true', help='Calculate for 5.0 GHz')
    parser.add_argument('-c24', '--channels-24', default=def_channels_24, help='Available channels for 2.4 GHz')
    parser.add_argument('-c50', '--channels-50', default=def_channels_50, help='Available channels for 5.0 GHz')
    parser.add_argument('-s', '--static-aps', help='Static APs channels as dictionary "{\'ap_name\': no, ...}"')
    parser.add_argument('-b', '--forbidden-channels', help='Forbidden channels per AP as dictionary "{\'ap_name\': [no1, no2], ...}"')
    parser.add_argument('-f', '--input-file', required=True, help='Weighted adjacency matrix')
    parser.add_argument('-t', '--max-time', type=int, default=30, help='Max execution time in seconds')
    parser.add_argument('-r', '--seed', type=int, help='Seed for randomized procedures')

    a = parser.parse_args()
    
    channels = a.channels_24 if a.type_24 else a.channels_50

    main(channels, a.static_aps, a.forbidden_channels, a.input_file, a.max_time, a.seed)
    