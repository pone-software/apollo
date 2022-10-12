from apollo.dataset.datasets import HistogramDataset
import numpy as np
from olympus.event_generation.data import EventCollection

# event_collection = EventCollection.from_folder('../../dataset/training/single_line_all_events')
event_collection = EventCollection.from_folder('../../dataset/all')
event_collection.redistribute(0,1000000)

dataset = HistogramDataset.from_event_collection(event_collection, start_time=0,end_time=1000000, number_of_modules=20)

dataset.save('./dataset/all_single_line_events_0_10000000_50')
# new_set = HistogramDataset.load('./dataset/test4')
new_set2 = HistogramDataset.load('./dataset/test4', np.sum, {
    'axis': 0
})
#
# print(new_set.histogram)
