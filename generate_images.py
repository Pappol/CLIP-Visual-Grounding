#imports
import json
import pickle
import torch
from torch.utils.data import Dataset
import pandas as pd
import clip
import torch
from tqdm import tqdm
import os
from diffusers import StableDiffusionPipeline
  

local_path = 'refcocog/images/'
local_annotations = 'refcocog/annotations/'

#dataset class definition
class Coco(Dataset):
    def __init__(self, path_json, path_pickle, train=True):
        self.path_json = path_json
        self.path_pickle = path_pickle
        self.train = train

        #load images and annotations
        with open(self.path_json) as json_data:
            data = json.load(json_data)
            self.ann_frame = pd.DataFrame(data['annotations'])
            self.ann_frame = self.ann_frame.reset_index(drop=False)

        with open(self.path_pickle, 'rb') as pickle_data:
            data = pickle.load(pickle_data)
            self.refs_frame = pd.DataFrame(data)

        #separate each sentence in dataframe
        self.refs_frame = self.refs_frame.explode('sentences')
        self.refs_frame = self.refs_frame.reset_index(drop=False)

        self.size = self.refs_frame.shape[0]

        #merge the dataframes
        self.dataset = pd.merge(self.refs_frame, self.ann_frame, left_on='ann_id', right_on='id')
        #drop useless columns for cleaner and smaller dataset
        self.dataset = self.dataset.drop(columns=['segmentation',  'iscrowd', 'image_id_y', 'image_id_x', 'sent_ids', 'index_y', 'area'])
        #self.dataset = self.dataset.drop(columns=['segmentation', 'id', 'category_id_y','ref_id', 'index_x', 'iscrowd', 'image_id_y', 'image_id_x', 'category_id_x', 'ann_id', 'sent_ids', 'index_y', 'area'])

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

def split_string(string):
    string = string.split("_")
    string = string[:-1]
    string = "_".join(string)
    append = ".jpg"
    string = string + append
    
    return string

dataset = Coco(local_annotations + 'instances.json', local_annotations + "refs(umd).p")

#define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#pass image into yolo
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

class_names=yolo.names

model, preprocess = clip.load("ViT-B/32")

model.eval()
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker = None) 
pipe = pipe.to(device)

dataset=dataset.get_validation()


for i in tqdm(range(1, len(dataset))):
  #if they already exist continue
    if os.path.exists("stable_diffusion/2/stable_diffusion_"+str(i)+"_1.jpg") and os.path.exists("stable_diffusion/2/stable_diffusion_"+str(i)+"_2.jpg") and os.path.exists("stable_diffusion/2/stable_diffusion_"+str(i)+"_3.jpg"):
        continue
    print(i)
    prompt= 'Use deep learning algorithms to generate a hyper-realistic portrait of '+   dataset.iloc[i]["sentences"]["raw"] +' Use advanced image processing techniques to make the image appear as if it were a photograph'

    stable_input = pipe(prompt,num_inference_steps=50).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
    stable_input2 = pipe(prompt,num_inference_steps=50).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
    stable_input3 = pipe(prompt,num_inference_steps=50).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)


    stable_input.save("stable_diffusion/2/stable_diffusion_"+str(i)+"_1.jpg") # Image saving to another directory
    stable_input2.save("stable_diffusion/2/stable_diffusion_"+str(i)+"_2.jpg") # Image saving to another directory
    stable_input3.save("stable_diffusion/2/stable_diffusion_"+str(i)+"_3.jpg") # Image saving to another directory
