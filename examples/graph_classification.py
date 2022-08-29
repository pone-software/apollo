from unittest import result
import dgl
import torch as th
import networkx as nx
from multiprocessing import Pool, cpu_count
import time


import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F

# import matplotlib.pyplot as plt
import numpy as np
import os
from apollo.graph.generators import TimeSeriesGraphGenerator, PerEventGraphGenerator
from apollo.graph.builders import CompleteGraphBuilder

from olympus.event_generation.data import EventCollection
from olympus.event_generation.detector import make_triang, Detector, make_line

if __name__ == "__main__":
    g = nx.complete_graph(20)
    # nx.draw(g, with_labels=True, font_weight='bold')
    # plt.show()
    test = np.arange(20)
    for x in range(20):
        g.nodes[x]["hmmm"] = test[x]

    filename = os.path.join(
        os.path.dirname(__file__), "../../data/events_track_0.pickle"
    )

    event_collection = EventCollection.from_pickle(filename)

    rng = np.random.RandomState(31338)
    oms_per_line = 20
    dist_z = 50  # m
    dark_noise_rate = 16 * 1e4 * 1e-9  # 1/ns

    pmts_per_module = 16
    pmt_cath_area_r = 75e-3 / 2  # m
    module_radius = 0.21  # m

    # Calculate the relative area covered by PMTs
    efficiency = (
        pmts_per_module
        * (pmt_cath_area_r) ** 2
        * np.pi
        / (4 * np.pi * module_radius**2)
    )
    det = Detector(
        make_line(0, 0, 20, 50, rng, dark_noise_rate, 0, efficiency=efficiency)
    )
    # det = make_triang(100, 20, dist_z, dark_noise_rate, rng, efficiency)

    event_collection.detector = det

    base_graph = CompleteGraphBuilder(event_collection).build_graph()
    graph_builder = TimeSeriesGraphGenerator(base_graph, event_collection)

    graphs = graph_builder.generate_graphs()

    # print(graphs)

    # graph_builder = PerEventGraphGenerator(base_graph, events)
