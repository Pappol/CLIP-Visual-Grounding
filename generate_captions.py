local_path = 'refcocog/images/'
local_annotations = 'refcocog/annotations/'

# imports

# imports

import json
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import clip
import numpy as np
from clip import tokenize

import stanza
from tqdm import tqdm

from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel
from transformers import RobertaTokenizerFast

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

stanza.download('en',package='partut')
nlp = stanza.Pipeline(lang='en')

def remove_of(sentence):
 if "the side of" in sentence:
    index=sentence.find("the side of")
    sentence=sentence[index+11:]
    return sentence
 if "the handle of" in sentence:
    index=sentence.find("the handle of")
    sentence=sentence[index+13:]
 if "the bunch of" in sentence:
    index=sentence.find("the bunch of")
    sentence=sentence[index+12:]
    return sentence   
 if "the corner of" in sentence:
    index=sentence.find("the corner of")
    sentence=sentence[index+13:]
    return sentence
 if "the end of" in sentence:
    index=sentence.find("the end of")
    sentence=sentence[index+10:]
    return sentence
 if "the half of" in sentence:
    index=sentence.find("the half of")
    sentence=sentence[index+11:]
    return sentence    
 if "the edge of" in sentence:
    index=sentence.find("the edge of")
    sentence=sentence[index+11:]    
    return sentence
 if "the back of" in sentence:
    index=sentence.find("the back of")
    sentence=sentence[index+11:]
    return sentence   
 if "the smaller of" in sentence:
    index=sentence.find("the smaller of")
    sentence=sentence[index+14:]
    return sentence    
 if "the piece of" in sentence:
    index=sentence.find("the piece of")
    sentence=sentence[index+16:]
    return sentence 
 if "the wing of" in sentence:
    index=sentence.find("the wing of")
    sentence=sentence[index+11:]    

 if "the front of" in sentence:
    index=sentence.find("the front of")
    sentence=sentence[index+12:]   
    return sentence 
 if "the back side of" in sentence:
    index=sentence.find("the back side of")
    sentence=sentence[index+16:]   
    return sentence
 if "the front side of" in sentence:
    index=sentence.find("the front side of")
    sentence=sentence[index+17:]   
    return sentence 
 if "the left side of" in sentence:
    index=sentence.find("the left side of")
    sentence=sentence[index+16:]   
    return sentence
 if "the right side of" in sentence:
    index=sentence.find("the back side of")
    sentence=sentence[index+17:]   
    return sentence
  
 if "the pile of" in sentence:
    index=sentence.find("the pile of")
    sentence=sentence[index+11:]    
    return sentence
 if "the pair of" in sentence:
    index=sentence.find("the pair of")
    sentence=sentence[index+11:] 
    return sentence   
 if "the pieces of" in sentence:
    index=sentence.find("the pieces of")
    sentence=sentence[index+13:]   
    return sentence 
 if "the intersection of" in sentence:
    index=sentence.find("the intersection of")
    sentence=sentence[index+19:]  
    return sentence  
 if "the middle of" in sentence:
    index=sentence.find("the middle of")
    sentence=sentence[index+13:]
    return sentence    
 if "the patch of" in sentence:
    index=sentence.find("the patch of")
    sentence=sentence[index+12:]    
    return sentence
 if "the couple of" in sentence:
    index=sentence.find("the couple of")
    sentence=sentence[index+12:]    
    return sentence
 if "the slice of" in sentence:
    index=sentence.find("the slice of")
    sentence=sentence[index+12:]    
    return sentence
 if "the tallest of" in sentence:
    index=sentence.find("the tallest of")
    sentence=sentence[index+14:]    
    return sentence
 if "the kind of" in sentence:
    index=sentence.find("the kind of")
    sentence=sentence[index+11:]
    return sentence    
 if "that is" in sentence:
    index=sentence.find("that is")
    sentence=sentence[:index]
    return sentence
 if "the part of" in sentence:
    index=sentence.find("the part of")
    sentence=sentence[index+11:]
    return sentence
 if "the corner of" in sentence:
    index=sentence.find("the corner of")
    sentence=sentence[index+13:]
    return sentence
 if "the half of" in sentence:
    index=sentence.find("the half of")
    sentence=sentence[index+11:]
    return sentence
 if "the top of" in sentence:
    index=sentence.find("the top of")
    sentence=sentence[index+10:]
    return sentence
 return sentence

def sent_stanza_processing(sentence):
    sentence = sentence.lower()
    if sentence.startswith('there is '):
        sentence = 'the ' + sentence[9:]
    if sentence.startswith('this is '):
        sentence = sentence[9:]
    sentence = remove_of(sentence)

    nlp_sent = sentence.lower()

    # put the sentence in lower case
    # if the sentence does not start with "a" or "the" insert it
    x = nlp_sent.split(" ")
    if (x[0] != "the" and x[0] != "a"):
        nlp_sent = "the " + nlp_sent

    doc = nlp(nlp_sent)
    # print nlp dependencies
    # doc.sentences[0].print_dependencies()
    # print(input["sentences"]["raw"])
    root = ''
    phrase_upos = []
    # get heads of words
    heads = [sent.words[word.head -
                        1].text for sent in doc.sentences for word in sent.words]
    for sent in doc.sentences:
        for word in sent.words:
            # if it is a verbal phrase then take the nominal subject of the phrase
            if (word.deprel == 'nsubj' or word.deprel == 'nsubj:pass'):
                root = word.text
                return word.text
                # print(word.text)
                break
            # print(word)
            phrase_upos.append(word)
            # else take the root of the phrase
            if (word.head == 0):
                # print(word.text)
                return word.text
                # root=word.text
                # if the root is a verb
                if (word.upos == 'VERB'):
                    for w in reversed(phrase_upos):
                        # go back until you get a noun
                        if (w.upos == 'NN'):
                            return word.text
                            # print(w.text)

def get_root(yolo_output, sentence, model, yolo):
    root = sent_stanza_processing(sentence)
    # print(root)
    prompt_tokens = tokenize(
        root, context_length=77, truncate=True).cuda()
    with torch.no_grad():
        prompt_features = model.encode_text(prompt_tokens).float()

    names = []
    for a in range(len(yolo_output.xyxy[0])):
        class_index = int(yolo_output.pred[0][a][5])
        label = yolo.names[class_index]
        names.append(label)
    tokens = tokenize(names, context_length=77, truncate=True).cuda()
    with torch.no_grad():
        classes_features = model.encode_text(tokens).float()
    prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
    classes_features /= classes_features.norm(dim=-1, keepdim=True)
    prompt_similarity = classes_features.cpu().numpy() @ prompt_features.cpu().numpy().T
    if prompt_similarity.shape[0] == 0:
        return "empty"
    rappresentation = np.argmax(prompt_similarity)

    interested_class = names[rappresentation]
    return interested_class



def clear_caption(caption):
    caption = caption.replace('<s>', '')
    caption = caption.replace('</s>', '')
    return caption

def crop_yolo(yolo_output, img, index):
    x1 = yolo_output.xyxy[0][index][0].cpu().numpy()
    x1 = np.rint(x1)
    y1 = yolo_output.xyxy[0][index][1].cpu().numpy()
    y1 = np.rint(y1)
    x2 = yolo_output.xyxy[0][index][2].cpu().numpy()
    x2 = np.rint(x2)
    y2 = yolo_output.xyxy[0][index][3].cpu().numpy()
    y2 = np.rint(y2)

    cropped_img = img.crop((x1, y1, x2, y2))

    return cropped_img
# remove the id in the image name string
def split_string(string):
    string = string.split("_")
    string = string[:-1]
    string = "_".join(string)
    append = ".jpg"
    string = string + append

    return string

# dataset class definition
class Coco(Dataset):
    def __init__(self, path_json, path_pickle, train=True):
        self.path_json = path_json
        self.path_pickle = path_pickle
        self.train = train

        # load images and annotations
        with open(self.path_json) as json_data:
            data = json.load(json_data)
            self.ann_frame = pd.DataFrame(data['annotations'])
            self.ann_frame = self.ann_frame.reset_index(drop=False)

        with open(self.path_pickle, 'rb') as pickle_data:
            data = pickle.load(pickle_data)
            self.refs_frame = pd.DataFrame(data)

        # separate each sentence in dataframe
        self.refs_frame = self.refs_frame.explode('sentences')
        self.refs_frame = self.refs_frame.reset_index(drop=False)

        self.size = self.refs_frame.shape[0]

        # merge the dataframes
        self.dataset = pd.merge(
            self.refs_frame, self.ann_frame, left_on='ann_id', right_on='id')
        # drop useless columns for cleaner and smaller dataset
        self.dataset = self.dataset.drop(columns=['segmentation', 'id', 'category_id_y', 'ref_id', 'index_x',
                                         'iscrowd', 'image_id_y', 'image_id_x', 'category_id_x', 'ann_id', 'sent_ids', 'index_y', 'area'])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset.iloc[idx]

    def get_annotation(self, idx):
        return self.ann_frame.iloc[idx]

    def get_imgframe(self, idx):
        return self.img_frame.iloc[idx]

    def get_validation(self):
        return self.dataset[self.dataset['split'] == 'val']

    def get_test(self):
        return self.dataset[self.dataset['split'] == 'test']

    def get_train(self):
        return self.dataset[self.dataset['split'] == 'train']
    

# dataset load
dataset = Coco(local_annotations + 'instances.json', local_annotations + "refs(umd).p")

dataframe = dataset.get_validation()
#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=20)
text_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
text_model = VisionEncoderDecoderModel.from_pretrained('/home/pappol/Scrivania/deepLearning/Image_Captioning_VIT_Roberta_final_4')
text_model.to(device)

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

clip, preprocess = clip.load('ViT-B/32')

with open('image_captioning_testset.json', 'w') as f:
    entries = []
    for i in tqdm(range(len(dataframe))):
        input = dataframe.iloc[i]
        image_path = split_string(input["file_name"])
        sentence = input["sentences"]["raw"]
        original_img = Image.open(local_path + image_path).convert("RGB")
        yolo_output = yolo(original_img)

        root = get_root(yolo_output, sentence, clip, yolo)

        frase = []
        for j in range(len(yolo_output.xyxy[0])):
            if root != "empty" and yolo.names[int(yolo_output.pred[0][i][5])] != root:
               continue
            cropped = crop_yolo(yolo_output, original_img, j)
            features = text_feature_extractor(cropped, return_tensors="pt").pixel_values.to(device)
            generated = text_model.generate(features)[0].to(device)
            caption = text_tokenizer.decode(generated)
            caption = clear_caption(caption)
            frase.append(caption)

        entry = {'sentences': frase,
                'index': i}
        entries.append(entry)
        
    json.dump(entries, f, indent=4)
