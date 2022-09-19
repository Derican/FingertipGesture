import json
import os
import numpy as np
from Plot import STUDY2_DIR, valid_study2_data

zero_data_filenames = np.load('study2/zero.npy')
overlap_data_filenames = np.load('study2/overlap.npy')

confusion_data_filenames = []

for person in valid_study2_data:
    for file in os.listdir('study2/{person}'.format(person=person)):
        if '.gif' in file:
            confusion_data_filenames.append(
                os.path.join(STUDY2_DIR, person, file))

error_dict = {}
for pers in valid_study2_data:
    error_data = []
    for t in zero_data_filenames:
        if pers in t:
            error_data.append(t)
    for o in overlap_data_filenames:
        if pers in o:
            error_data.append(o)
    for c in confusion_data_filenames:
        if pers in c:
            error_data.append(c)
    error_dict[pers] = error_data

with open('error_config.json', 'w') as f:
    json.dump(error_dict, f)
