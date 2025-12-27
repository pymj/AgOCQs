#supporting import
import pandas as pd
import numpy as np
import os
import sys
import copy
from datasets import load_dataset
from pprint import pprint
import re
import spacy
import nltk
from sklearn.utils import shuffle
import argparse
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from nltk.corpus import stopwords
from model import *
from operator import itemgetter
from tqdm import tqdm
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")
from transformers import T5ForConditionalGeneration,T5Tokenizer 
import PyPDF2
import traceback
from typing import Tuple, Union, List, Optional
import gc
# neuralcoref.add_to_pipe(nlp)
import coreferee
nlp.add_pipe("coreferee")

from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import pytorch_lightning as pl
# from termcolor import colored
import textwrap
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
pl.seed_everything(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)


#load default dataset and splits for pretraining- already saved in folder
train_data = load_dataset('squad', split='train')
validate_data = load_dataset('squad', split='validation')
sample_validation_dataset = next(iter(validate_data))
df_train = pd.DataFrame(columns = ['context','answers', 'question'])
df_validation = pd.DataFrame(columns = ['context', 'answers', 'question'])
counter_train= 0
for index,val in enumerate(tqdm(train_data)):
    passage = val['context']
    question = val['question']
    answer = val['answers']['text'][0]
    no_of_words = len(answer.split())
    if no_of_words > 5:
        df_train.loc[counter_train]= [passage] + [answer] + [question]
        counter_train = counter_train + 1

counter_val = 0
df_train = pd.DataFrame(columns=['passage','answer','question'])
for index,val in enumerate(tqdm(validate_data)):
    passage = val['context']
    question = val['question']
    answer = val['answers']['text'][0]
    no_of_words = len(answer.split())
    if no_of_words > 5:
        df_validation.loc[counter_val]= [passage] + [answer] + [question]
        counter_val = counter_val + 1

# save the dataset
df_train.to_csv("training_data.csv", index = False)
df_validation.to_csv("val_data.csv", index = False)


class QuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_len_inp=512,max_len_out=96):
        self.path = filepath

        self.passage_column = "context"
        #self.answer = "answer"
        self.question = "question"

        # self.data = pd.read_csv(self.path)
        self.data = pd.read_csv(self.path,nrows=1000)

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount =0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        labels = copy.deepcopy(target_ids)
        labels [labels==0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,"labels":labels}

    def _build(self):
        for idx in tqdm(range(len(self.data))):
            passage,target = self.data.loc[idx, self.passage_column], self.data.loc[idx, self.question]

            input_ = "context: %s </s>" % (passage)
            target = "question: %s </s>" % (str(target))

            # get encoding length of input. If it is greater than self.max_len skip it
            test_input_encoding = self.tokenizer.encode_plus(input_,
                                        truncation=False,
                                        return_tensors="pt")

            length_of_input_encoding = len(test_input_encoding['input_ids'][0])


            if length_of_input_encoding > self.max_len_input:
              self.skippedcount = self.skippedcount + 1
              continue

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len_input, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len_output, pad_to_max_length=True,return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

# LoRA Configuration for CPU
def peft_model(model_name='t5-base'):
    """Create T5 with LoRA for CPU training"""
    
    print("Loading base model...")
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # FP32 for CPU
        low_cpu_mem_usage=True
    )
    
    # LoRA configuration - optimized for CPU
    lora_config = LoraConfig(
        r=16,  # Rank - higher = better quality, more memory
        lora_alpha=32,  # Scaling parameter
        target_modules=["q", "v"],  # Which layers to adapt
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.QUESTION_ANS,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Output: trainable params: 2,359,296 || all params: 223,020,032 || trainable%: 1.06
    
    return model

# Training class for CPU
class CPUTrainer:
    def __init__(self, model, tokenizer, hparams, train_dataset, validation_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams
        self.train_data = train_dataset
        self.val_data = validation_dataset
        self.device = torch.device('cpu')

    def forward( self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
         outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

         return outputs


    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        # loss = outputs.loss
        # total_loss += loss.item()
        loss = outputs[0]
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss",loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=0)


    # optimizer changed for CPU training to torch.optim.SGD
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=3e-4, eps=1e-8)
        return optimizer
    
    def train(self, epochs=3, batch_size=1, learning_rate=1e-3):
        """Custom training loop optimized for CPU"""
        
        # DataLoader with minimal memory footprint
        train_loader = self.train_dataloader()
        
        validate_loader = self.val_dataloader()
            
        optimizer = self.configure_optimizers()
        
            
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
           
            for batch_idx, batch in enumerate(train_progress_bar):    
                # Forward pass
                train_outputs = self.training_step(batch, batch_idx)
                total_loss += train_outputs.item()
                # Backward pass
                optimizer.zero_grad()
                train_outputs.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                # Update progress bar
                train_progress_bar.set_postfix({'loss': train_outputs.item()})
                
                # Aggressive memory cleanup every 10 steps
                if batch_idx % 10 == 0:
                    gc.collect()
            val_loss= 0
            self.model.eval()
            validate_progress_bar = tqdm(validate_loader, desc=f'Epoch {epoch+1}/{epochs}')
            with torch.no_grad():
                for batch_idx, batch in enumerate(validate_progress_bar):  
                    # Forward pass
                    val_outputs = self.validation_step(batch, batch_idx)
                    
                    # Update progress bar
                    validate_progress_bar.set_postfix({'val_loss': val_outputs.item()})
                    
                    # Aggressive memory cleanup every 10 steps
                    if batch_idx % 10 == 0:
                        gc.collect()
                    
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(validate_loader)
            print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
            print(f"Epoch {epoch+1} -  Validation Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch)

def save_checkpoint(self, epoch):
    """Save model checkpoint"""
    self.model.save_pretrained(f'./lora_checkpoint_epoch_{epoch}')
    print(f"Checkpoint saved for epoch {epoch+1}")