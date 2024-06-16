import clip
import torch
import json
from tqdm import tqdm as tqdm_func 
from PIL import Image
import os
import pickle
import numpy as np
from tqdm import tqdm
# input:
# json_file_path:your json file path
# device: gpu device, e.g.: "cuda:0"


def get_clip_score(json_file_path, device):
    if device == None:
        device = "cuda:0"
    device = device if torch.cuda.is_available() else "cpu"
    
    clip_type = "img&text"
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    model.eval()

    with open(json_file_path, "r") as f:
        generated_captions = json.load(f)
    image_root = "D:/data/val2014"

    # FOR Image and text
    if clip_type == "img&text":
        clip_scores = []
        for entry in tqdm_func(generated_captions, desc="Calculating CLIP Scores"):
            image_id = entry["image_id"]
            generated_caption = entry["caption"]
            image_path = os.path.join(image_root, f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
            score = calculate_clip_score_img_text(generated_caption, image_path, model, preprocess, device)
            clip_scores.append(score)
        average_clip_score = sum(clip_scores) / len(clip_scores)
        print("Average CLIP Score:", average_clip_score)

    # FOR Text and text
    if clip_type == "text&text":
        with open("/data/wyl/coco_data/annotations/captions_val2014.json", "r") as f:
            mscoco_test_annotations = json.load(f)
        progress_bar = tqdm_func(total=len(generated_captions), desc="Calculating CLIP Scores")
        clip_scores = []
        for entry in tqdm_func(generated_captions, desc="Calculating CLIP Scores"):
            image_id = entry["image_id"]
            generated_caption = entry["caption"]
            reference_captions = [ann["caption"] for ann in mscoco_test_annotations["annotations"] if ann["image_id"] == image_id]
            scores = 0
            for i in range(len(reference_captions)):
                score = calculate_clip_score_text_and_text(reference_captions[i], generated_caption, model, device)
                scores = scores + score 
            clip_scores.append(scores/len(reference_captions))
        average_clip_score = sum(clip_scores) / len(clip_scores)
        print("Average CLIP Score:", average_clip_score)

def calculate_clip_score_img_text(reference_caption, image_path, model, preprocess, device):
    text_input = clip.tokenize(reference_caption).to(device)
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        image_features = model.encode_image(image_input)
        similarity_score = (text_features @ image_features.T).mean()
    return similarity_score.item()

def calculate_clip_score_text_and_text(reference_caption, generated_caption, model, device):
   text_inputs = torch.cat([clip.tokenize(reference_caption), clip.tokenize(generated_caption)]).to(device)
   with torch.no_grad():
       text_features = model.encode_text(text_inputs)
       text_features = text_features.chunk(2)
       similarity_score = (text_features[0] @ text_features[1].T).mean()
   return similarity_score.item()


#get_clip_score('captions_val2014_kdy_results_1.json',device="cuda:0")


input_seq = "D:\EENAIC\mscoco\sent/coco_train_input.pkl"
input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')

#3print(input_seq['15335'],input_seq['15335'].shape)
vocabulary = {}
modified_data = {}
with open('coco_vocabulary.txt', 'r') as vocab_file:
    for index, word in enumerate(vocab_file):
        word = word.strip()
        vocabulary[index] = word

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

#a = lang2index(input_seq['15335'])
#print(a)
model, preprocess = clip.load("ViT-L/14", device="cuda", jit=False)
model.eval()
image_root = "D:/data/val2014"
image_root_train =  "D:/data/train2014"
modified_data = {}

for id, sent in tqdm(input_seq.items()):
    input_lang = lang2index(sent)
    best_score = 0
    best_id = 0
    for sent_id in range(len(input_lang)):
        if os.path.exists(os.path.join(image_root, f"COCO_val2014_{str(id).zfill(12)}.jpg")):
            image_path = os.path.join(image_root, f"COCO_val2014_{str(id).zfill(12)}.jpg")
        else:
            image_path = os.path.join(image_root_train, f"COCO_train2014_{str(id).zfill(12)}.jpg")
        score = calculate_clip_score_img_text(input_lang[sent_id], image_path, model, preprocess, "cuda")
        if score>best_score:
            best_score = score
            best_id = sent_id
    best_sent = input_lang[best_id]
    modified_data[id] = sent[best_id].reshape(1,-1)
        
with open('input_clip.pkl', 'wb') as pickle_file:
    pickle.dump(modified_data, pickle_file)
