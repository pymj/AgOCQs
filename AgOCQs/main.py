
#supporting import
import pandas as pd
import numpy as np
from datasets import load_dataset
from pprint import pprint
import re
from model import *
from AgOCQs.process_data import *
from generate_cqs import *
import spacy
import nltk
from sklearn.utils import shuffle
import argparse
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from nltk.corpus import stopwords
from tqdm import tqdm
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")
from transformers import T5ForConditionalGeneration,T5Tokenizer 
import traceback
from typing import Tuple, Union, List, Optional
import gc
import coreferee
nlp.add_pipe("coreferee")
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import pytorch_lightning as pl
import os
from argparse import Namespace
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

def main(file_paths):
    # Initialize
    trained_tokenizer = os.getcwd() + '/model/t5/tokenizer/'
    #load stored trained squad dataset
    train_data_path = os.getcwd() + '/model/t5/dataset/squad_t5_train.csv'
    val_data_path = os.getcwd() + '/model/t5/dataset/squad_t5_val.csv'
    tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)
    hparams= Namespace(
        epochs=3,
        batch_size=1,  # Keep at 1 for 8GB RAM
        learning_rate=1e-3  # Higher LR works well with LoRA
    )
    train_dataset = QuestionGenerationDataset(tokenizer,train_data_path)
    validation_dataset = QuestionGenerationDataset(tokenizer,val_data_path)
    
    # Create LoRA model
    model = peft_model('t5-base')
    
    # Initialize trainer
    trainer = CPUTrainer(model, tokenizer, hparams, train_dataset, validation_dataset)
    
    # Train - adjust parameters based on your RAM
    trainer.train(epochs=hparams.epochs)

    # Save final model
    model.save_pretrained('./final_lora_model')
    tokenizer.save_pretrained('./final_lora_model')
    print("---------Training complete!----------")
    list_text_files = []
    text_file_path= get_files(file_paths)
    for textFile in text_file_path:
        if textFile.endswith(".txt"):
            CQs_file_name = Path(textFile).stem
            result_name = CQs_file_name.join("_CQs")
            Similar_CQs_filter, distinct_patterns, result_name  = generateCQs(textFile)
            list_text_files.append(CQs_file_name)
    return list_text_files, Similar_CQs_filter, distinct_patterns, result_name 
if __name__ == "__main__":
    main()