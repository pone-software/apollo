import matplotlib.pyplot as plt

from apollo.data.configs import HistogramConfig
from apollo.data.events import EventCollection, EventTimeframeMode
from apollo.dataset.generators import NoisedHistogramGenerator
from apollo.utils.detector_helpers import get_line_detector

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

histogram = HistogramConfig(start=0, end=1000, bin_size=10)

det = get_line_detector()

# event_collection = EventCollection.from_folder('../../dataset/training/single_line_all_events')
event_collection = EventCollection.from_pickle('../../data/cascades/events_cascade_0.pickle', detector=det)
event_collection.detector = det
event_collection.redistribute(interval=histogram, is_in_timeframe_mode=EventTimeframeMode.CONTAINS_PERCENTAGE, percentile=20)
dataset = NoisedHistogramGenerator(event_collection=event_collection, histogram_config=histogram)
#
# dataset.save('./dataset/all_single_line_events_0_10000000_50')
# # new_set = HistogramDataset.load('./dataset/test4')
# new_set2 = HistogramDataset.load('./dataset/test4', np.sum, {
#     'axis': 0
# })
#
# print(new_set.histogram)
dataset.generate('data/test')
print('test')
