import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from mne import find_events

from utils.loader import unicorn_channels, convert_to_mne

matplotlib.use("Qt5Agg")


def parse_element(element):
    if isinstance(element, int) and element == 0:
        return [0]
    elif isinstance(element, str) and element == '0':
        return [0]
    else:
        return [int(x) for x in str(element).split(',')]


data_folder = '../data'

filepath = os.path.join(data_folder, 'speller_test.txt')

header = ["timestamp", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "id", "trigger"]

df = pd.read_csv(filepath, skiprows=5, sep='\t', names=header)

# This colum is tricky because it contains a list of integers separated by commas
array_of_ids = list(df.iloc[:, 9].values)

# Apply the function to each element in the data list
parsed_data = list(map(parse_element, array_of_ids))

# Replace the data column with the parsed data
df['id'] = parsed_data
print(df.head())

# Convert trigger value to 1 if it is 0 and to 2 if it is 1
trigger = [1 if x == '0' else 2 for x in df['trigger'].to_numpy()]

raw_array = convert_to_mne(df.iloc[:, 1:9], df['trigger'].to_numpy(), 250, chs=unicorn_channels, recompute=True)
ev_ids = {'NT': 1, 'T':2}
event_colors = {1:'r', 2:'g'}
stim_channel = 'STI'
events = find_events(raw_array, stim_channel=stim_channel)
raw_array.save('../data/speller_test-raw.fif', overwrite=True)
raw_array.plot(events=events, event_id=ev_ids, color=event_colors, n_channels=9)
plt.show()
