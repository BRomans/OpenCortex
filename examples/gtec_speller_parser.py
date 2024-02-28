import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

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

raw_array = convert_to_mne(df.iloc[:, 1:9], df['trigger'].to_numpy(), 250, chs=unicorn_channels, recompute=False)

raw_array.save('../data/speller_test-raw.fif', overwrite=True)
raw_array.plot()
plt.show()
