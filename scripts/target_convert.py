import clip
import torch
import json
from tqdm import tqdm as tqdm_func
from PIL import Image
import os
import pickle
import numpy as np
from tqdm import tqdm

input_seq = "./labelselect/vinvl_rouge.pkl"
input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')

def fill_blank(original_array):
    desired_shape = (1, 17)
    if original_array.shape != desired_shape:
        # Calculate how many additional '-1' elements need to be added
        elements_to_add = desired_shape[1] - original_array.shape[1]

        # Create an array of '-1' elements to add
        additional_elements = np.ones((1, elements_to_add), dtype=np.int64) * -1

        # Concatenate the arrays to reach the desired shape
        final_array = np.concatenate((original_array, additional_elements), axis=1)
    else:
        final_array = original_array
    return final_array

modified_data = {}
for id, sent in tqdm(input_seq.items()):
    sent = sent[:, 1:]
    first_zero_index = np.where(sent[0] == 0)[0]

    # Check if '0' was found
    if first_zero_index.size > 0:
        # Get the index of the element after the first '0'
        target_test = (sent[0][:first_zero_index[0]+1]).reshape(1,-1)

    result_array = fill_blank(target_test)
    modified_data[id] = result_array

with open('./labelselect/vinvl_rouge_target.pkl', 'wb') as pickle_file:
    pickle.dump(modified_data, pickle_file)



target_seq1 = "./labelselect/vinvl_rouge_target.pkl"
target_seq1 = pickle.load(open(target_seq1, 'rb'), encoding='bytes')

#print(target_seq1['15335'],input_seq['15335'])


