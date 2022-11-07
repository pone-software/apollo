import os

import networkx as nx
import numpy as np

from apollo.data.events import EventCollection
from apollo.graph.builders import CompleteGraphBuilder
from apollo.graph.generators import TimeSeriesGraphGenerator
from apollo.utils.detector_helpers import get_line_detector

if __name__ == "__main__":
    g = nx.complete_graph(20)
    # nx.draw(g, with_labels=True, font_weight='bold')
    # plt.show()
    test = np.arange(20)
    for x in range(20):
        g.nodes[x]["mmm"] = test[x]

    filename = os.path.join(
        os.path.dirname(__file__), "../../data/tracks/events_track_0.pickle"
    )

    event_collection = EventCollection.from_pickle(filename, detector=get_line_detector())

    base_graph = CompleteGraphBuilder(event_collection).build_graph()
    graph_builder = TimeSeriesGraphGenerator(base_graph, event_collection)

    graphs = graph_builder.generate_graphs()

    print("test")

    # print(graphs)

    # graph_builder = PerEventGraphGenerator(base_graph, events)
