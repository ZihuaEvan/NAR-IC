from pycocoevalcap.cider.cider import Cider
import pickle
import json
from tqdm import tqdm
import numpy as np
# Ground truth and candidate (to-be-evaluated) captions
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor

rouge_scorer = Rouge()


vocabulary = {}
with open('coco_vocabulary.txt', 'r') as vocab_file:
    for index, word in enumerate(vocab_file):
        word = word.strip()
        vocabulary[index] = word

input_seq = "mscoco/sent/coco_train_input.pkl"
input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')

kd_seq = "input_vinvl.pkl"
kd_seq = pickle.load(open(kd_seq, 'rb'), encoding='bytes')

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

def lang2index(indexes_array):
    indexes_array = indexes_array[:, 1:]

    # Slice the array to keep only the meaningful indexes before the first '0'
    sent_list=[]
    for sent in indexes_array:
        try:
            first_zero_index = np.where(sent == 0)[0][0]
            meaningful_indexes = sent[:first_zero_index]
            word_list = [vocabulary[index] for index in meaningful_indexes]
            sentence = ' '.join(word_list)
            sent_list.append(sentence)
        except:
            word_list = [vocabulary[index] for index in sent]
            sentence = ' '.join(word_list)
            sent_list.append(sentence)
    return sent_list

def trainset_build(scorer,save_name):
    new = {}
    for id, sent in tqdm(input_seq.items()):
        kd_sent = kd_seq[id]
        kd_lang =lang2index(kd_sent)
        input_lang = lang2index(sent)
        best_score = 0

        for i in range(5):
            senti = {id: [str(input_lang[i])]}

            new_dict = {id: kd_lang}
            rouge_score, _ = scorer.compute_score(new_dict, senti)
            #rouge_score, _ = rouge_scorer.compute_score(new_dict, senti)

            # Update best score and best id if needed
            if rouge_score > best_score:
                best_score = rouge_score

            # Print the best score and best id for each input sequence
                new[id] = kd_sent

    with open(save_name+'.pkl', 'wb') as pickle_file:
        pickle.dump(new, pickle_file)

def datalist_build(scorer, ifsave=None):
    datalist = []

    for i in range(5):
        kd_lang = {}
        input_lang={}
        for id, sent in tqdm(input_seq.items()):
            kd_sent = kd_seq[id]
            input_lang[id] = [lang2index(sent)[i]]
            kd_lang[id] = lang2index(kd_sent)

        print("setting up cider.")
        #scorer = Cider()
        #score, scores = scorer.compute_score(input_lang,kd_lang)
        score, scores = rouge_scorer.compute_score(input_lang,kd_lang)
        datalist.append(scores)

    result_indices = [max(range(len(datalist)), key=lambda i: datalist[i][j]) for j in range(len(datalist[0]))]

    flag = 0
    new = {}
    for id, sent in tqdm(input_seq.items()):
        if flag < len(result_indices):
            new[id] = sent[result_indices[flag]].reshape(1,-1)
            flag += 1

    file_name = "vinvl_meteor.pkl"

    # Save the list to a file using pickle
    if ifsave == True:
        with open(file_name, "wb") as file:
            pickle.dump(new, file)
    return new
'''

with open("vinvl_cider.pkl", "rb") as file:
    vinvl_list = pickle.load(file)
with open("vinvl_rouge.pkl", "rb") as file:
    swin_list = pickle.load(file)
count_common = sum(1 for x, y in zip(vinvl_list, swin_list) if x == y)

print(f"The number of common elements is: {count_common}")'''

scorer = Meteor()
print(datalist_build(scorer=scorer,ifsave=False))

#trainset_build(scorer=scorer,save_name="vinvl_cider")