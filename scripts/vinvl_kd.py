import pickle
import json

import numpy
import numpy as np

input_seq = "mscoco/sent/coco_train_input_kd.pkl"
input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')


target_seq = "mscoco/sent/coco_train_target_kd.pkl"
target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')

print(target_seq['15335'],target_seq['15335'].shape)


max_caption_length = 17

# Specify the path to your JSON file
json_file_path = 'vinvl140.json'  # Replace with the actual path to your JSON file

# Open and read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 'data' now contains the contents of the JSON file as a Python dictionary
#print(data)

vocabulary = {}
modified_data = {}
with open('coco_vocabulary.txt', 'r') as vocab_file:
    for index, word in enumerate(vocab_file):
        word = word.strip()
        vocabulary[word] = index

# Function to tokenize a caption and convert words to indexes
def tokenize_and_index(caption, vocab):
    words = caption.split()  # Split the caption into words
    if words:
        last_word = words[-1]
        if last_word.endswith('.'):
            words[-1] = last_word[:-1]
            words.append('.') # Remove '.' from the last word
    indexed_caption = [vocab.get(word, vocab['UNK']) for word in words]
    while len(indexed_caption) >= max_caption_length:
        return numpy.zeros(1)
    while len(indexed_caption) < max_caption_length:
        indexed_caption.append('-1')


    indexed_caption = np.array(indexed_caption, dtype=np.int64)
    return indexed_caption.reshape(1, -1)   # Replace words with indexes

# Replace captions with tokenized and indexed captions
for item in data:
    image_id = item['image_id']
    caption = item['caption']
    indexed_caption = tokenize_and_index(caption, vocabulary)
    if indexed_caption.all()==0:
        modified_data[image_id] = target_seq[image_id]
        print(image_id)
    else:
        modified_data[image_id] = indexed_caption

#print(modified_data)
# Save the updated data back to a JSON file
#with open('updated_json_file.json', 'w') as updated_json_file:
#   json.dump(modified_data, updated_json_file, indent=4)
with open('target.pkl', 'wb') as pickle_file:
    pickle.dump(modified_data, pickle_file)

target_seq = "target.pkl"
target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')

print(target_seq['15335'],target_seq['15335'].shape)


##################################
max_caption_length = 17

# Specify the path to your JSON file
json_file_path = 'vinvl140.json'  # Replace with the actual path to your JSON file

# Open and read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 'data' now contains the contents of the JSON file as a Python dictionary
#print(data)

vocabulary = {}
modified_data = {}
with open('coco_vocabulary.txt', 'r') as vocab_file:
    for index, word in enumerate(vocab_file):
        word = word.strip()
        vocabulary[word] = index

# Function to tokenize a caption and convert words to indexes
def tokenize_and_index(caption, vocab):
    words = caption.split()  # Split the caption into words
    if words:
        last_word = words[-1]
        if last_word.endswith('.'):
            words[-1] = last_word[:-1]
            # Remove '.' from the last word
    indexed_caption = [vocab.get(word, vocab['UNK']) for word in words]
    indexed_caption.insert(0, '0')
    if len(indexed_caption) >= max_caption_length:
        return numpy.zeros(1)
    while len(indexed_caption) < max_caption_length:
        indexed_caption.append('0')
    indexed_caption = np.array(indexed_caption, dtype=np.int64)
    return indexed_caption.reshape(1, -1)    # Replace words with indexes

# Replace captions with tokenized and indexed captions
for item in data:
    image_id = item['image_id']
    caption = item['caption']
    indexed_caption = tokenize_and_index(caption, vocabulary)
    if indexed_caption.all() == 0:
        modified_data[image_id] = input_seq[image_id]
    else:
        modified_data[image_id] = indexed_caption

#print(modified_data)
# Save the updated data back to a JSON file
#with open('updated_json_file.json', 'w') as updated_json_file:
#   json.dump(modified_data, updated_json_file, indent=4)
with open('input_vinvl.pkl', 'wb') as pickle_file:
    pickle.dump(modified_data, pickle_file)

target_seq = "input_vinvl.pkl"
target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')

#print(target_seq['15335'].shape)
