import json
import pickle
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import datasets
import transformers
import pandas as pd
import torch
from pathlib import Path
from transformers import RobertaTokenizerFast # After training tokenizern we will wrap it so it can be used by Roberta model
from transformers import default_data_collator
import argparse

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import ViTFeatureExtractor
from transformers import TrainerCallback
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


import requests

#declarations
TRAIN_BATCH_SIZE = 58  # input batch size for training (default: 64)
VALID_BATCH_SIZE = 5   # input batch size for testing (default: 1000)

TRAIN_EPOCHS = 40       # number of epochs to train (default: 10)
VAL_EPOCHS = 1 

LEARNING_RATE = 1e-4   # learning rate (default: 0.01)
SEED = 42              # random seed (default: 42)
MAX_LEN = 128          # Max length for product description
SUMMARY_LEN = 20       # Max length for product names
WEIGHT_DECAY = 0.01    # Weight decay (default: 1e-4)

class IAMDataset(Dataset):
    def __init__(self, df, tokenizer,feature_extractor, decoder_max_length = 20):
        self.df = df
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.decoder_max_length = decoder_max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        img_path = self.df['images'][idx]
        caption = self.df['captions'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        
        # add labels (input_ids) by encoding the text
        labels = self.tokenizer(caption, truncation = True,
                                          padding="max_length", 
                                          max_length=self.decoder_max_length).input_ids
        
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def compute_metrics(pred):
    rouge = datasets.load_metric("rouge")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

# class SaveModelCallback(TrainerCallback):
#     def __init__(self, save_interval, trainer):
#         self.save_interval = save_interval
#         self.trainer = trainer
#     def on_epoch_end(self, args, state, control, **kwargs):
#         if (state.epoch + 1) % self.save_interval == 0:
#             self.trainer.save_model(f"Image_Captioning_VIT_Roberta_iter_epoch{state.epoch + 1}")


def main(args):
    df = pd.read_csv("RefCOCOg_cropped.csv")
    df['cropped'] = df['cropped'].str.replace('refcocog/', '')
    df = df.rename(columns={'cropped': 'images', 'raw': 'captions'})
    df['captions'] = df['captions'].str.lower()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=MAX_LEN)

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    batch_size=TRAIN_BATCH_SIZE

    train_dataset = IAMDataset(df=train_df.sample(frac=1,random_state=2).iloc[:].reset_index().drop('index',axis =1),
                            tokenizer=tokenizer,
                            feature_extractor= feature_extractor)

    test_dataset = IAMDataset(df=test_df.sample(frac=1,random_state=2)[:].reset_index().drop('index',axis =1),
                            tokenizer=tokenizer,feature_extractor= feature_extractor)

    # set encoder decoder tying to True
    model = VisionEncoderDecoderModel.from_pretrained(args.path)# could be VisionEncoderDecoderModel

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 20
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # load rouge for validation
    rouge = datasets.load_metric("rouge")

    captioning_model = 'VIT_Captioning'


    training_args = Seq2SeqTrainingArguments(
        output_dir=captioning_model,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        #evaluate_during_training=True,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=1024,  
        save_steps=2048, 
        warmup_steps=1024,  
        num_train_epochs = TRAIN_EPOCHS, #TRAIN_EPOCHS
        overwrite_output_dir=True,
            save_strategy="epoch",
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        tokenizer=feature_extractor,
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
        #save strategy
    )

    trainer.train()

    trainer.save_model('Image_Captioning_VIT_Roberta_final')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="Image_Cationing_VIT_Roberta_iter2")
    args = parser.parse_args()
    main(args)