import os
import logging
import pickle

import numpy as np

from olympus.event_generation.data import EventCollection


logging.getLogger().setLevel(logging.INFO)

data_base_path = os.path.join('../../data/training')
histogram_path = os.path.join(data_base_path, 'histograms')



event_collection = EventCollection.from_folder(os.path.join(data_base_path, 'single_line_all_events'))

for i in range(1, 21):
    step_size = i * 50
    histogram, results = event_collection.generate_histogram(50, 0, 10000000)
    with open(os.path.join(histogram_path, 'single_line_histo_{index}.npy'.format(index=step_size)), 'wb') as f:
        np.save(f, histogram)
        f.close()
    with open(os.path.join(histogram_path, 'single_line_results_{index}.pickle'.format(index=step_size)), 'wb') as f:
        pickle.dump(results, f)
        f.close()