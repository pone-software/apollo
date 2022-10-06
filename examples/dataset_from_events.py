from apollo.data.dataset import HistogramDataset
import numpy as np
from olympus.event_generation.data import EventCollection

# event_collection = EventCollection.from_folder('../../data/training/single_line_all_events')
event_collection = EventCollection.from_folder('../../data/all')
event_collection.redistribute(0,1000000)

dataset = HistogramDataset.from_event_collection(event_collection, start_time=0,end_time=1000000, number_of_modules=20)

dataset.save('./data/all_single_line_events_0_10000000_50')
# new_set = HistogramDataset.load('./data/test4')
new_set2 = HistogramDataset.load('./data/test4', np.sum, {
    'axis': 0
})
#
# print(new_set.histogram)
