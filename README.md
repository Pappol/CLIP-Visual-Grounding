# Visual grounding

## Introduction
This projects aim to study the visual grounding task witch consists in given a image and a caption founding the portion of the image in the form of a bounding box that best describes the caption. The visual grounding task is a very important task in the computer vision field and has many applications in the real world. For example, it can be used to help blind people to understand the world around them.

## Abstract
We tackled the problem from different prospective using as a baseline to compare the performances. We used different deep learning techniques to solve the problem. 
## Dataset
We used as refcocog+ witch is an annotated dataset specifically for this task. It is composed by 142210 images and 141564 captions. The captions are annotated with the bounding box that best describes the caption. The dataset is divided in 3 parts: train, validation and test. The train set is composed by 120000 images, the validation set by 10000 images and the test set by 22210 images. The dataset is available at this [link](https://drive.google.com/uc?id=1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq).

## Environment
Inside the jupiter notebook the first cell is dedicated to the environment setup. The environment is composed by the following libraries:
- torch
- https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt 
- ftfy
- regex
- tqdm
- git+https://github.com/openai/CLIP.git
- transformers == 4.28.0
- rouge-metric
- stanza

## How to run the project
To run the project you have to download the dataset and put it in the root of the project. All the project run within a single jupiter notebook witch is in the root of the project. The notebook is called "Visual grounding.ipynb". The notebook is divided in 3 parts: the first part is the data preparation, the second part is the training and the third part is the evaluation. The notebook is self explanatory and all the code is commented.
