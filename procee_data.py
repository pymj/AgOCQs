#supporting import
import pandas as pd
import numpy as np
import os
from pprint import pprint
import re
import spacy
import nltk
from sklearn.utils import shuffle
import argparse
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")
import PyPDF2
import traceback
from typing import Tuple, Union, List, Optional
import coreferee
nlp.add_pipe("coreferee")

from pathlib import Path

def extract_pdf(pdf_path):
    print("pdf_path:", pdf_path)
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print("Number of pages",num_pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

base_path = os.getcwd() + "/inputText/request/"
workdir=os.listdir(base_path)
if '.DS_Store' in workdir:
  workdir.remove('.DS_Store')
print (base_path)

def get_files(path):
    file_names = [file_name for file_name in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, file_name))]
    file_paths = [os.path.join(base_path, file_name) for file_name in file_names]
    return file_paths

file_paths=get_files(base_path)

for file_path in file_paths:
    if file_path.endswith('.pdf'):
        key_name = Path(file_path).stem
        filePath= os.path.join(base_path,key_name)
        outFile = os.path.join(base_path,f"{key_name}.txt")
        # outFile = ''.join(base_path,f"/{key_name}.txt")
        print(f"Processing {filePath}...")
        with open(file_path, 'rb') as f:
            text = extract_pdf(f)
            with open(outFile, "w", encoding="utf-8") as extract:
                extract.write(text)
            print(f"Extracted text from {file_path} and saved to {outFile}")

    elif file_path.endswith('.txt'):
        outFile = file_path
    

def readTextFile(path):   
    outText = []
    for file_path in path:
        if file_path.endswith('.txt'):
                try:
                    txtFile= open(file_path, "r")
                    mypdf = txtFile.readlines()
                    outText.append(mypdf)
                except FileNotFoundError:
                    print("File not found.")
                    print(traceback.format_exc())
                except Exception as e:
                    print("Error:", e)
                finally:
                    if 'txtFile' in locals():
                        txtFile.close()

    print("raw text has been read")
    return outText

def extract_sentences(file):
        content = nlp(str(file))
        # content_resolved = content._.coref_resolved
        content_resolved = content._.coref_chains
        # extract sentences from the resolved content
        sentences=[x.text for x in content_resolved.sents]
        return sentences

# formatting and cleaning text
def format(outText):
        """Fix syntax issues in incoming text fields, before any further processing
        """
        extracted_sentences= extract_sentences(outText)
        texts= ' '.join(extracted_sentences)
        print("sample sentences", extracted_sentences[:10])
        return texts



class TextPreprocessor:
    
    def __init__(self):
        """Initialize the preprocessor with common patterns."""
        # Patterns for different types of numbering
        self.numbering_patterns = [
            # Section numbers like 1.1, 1.2.3, etc.
            r'^\s*\d+(?:\.\d+)*\s+',
            # Simple line numbers at start
            r'^\s*\d+\s+',
            # Parenthetical numbers like (1), (2), etc.
            r'^\s*\(\d+\)\s*',
            # Letter numbering like a), b), etc.
            r'^\s*[a-zA-Z]\)\s*',
            # Roman numerals
            r'^\s*[IVXLCDM]+\.\s*',
            r'^\s*[ivxlcdm]+\.\s*',
            # Bullet points with numbers
            r'^\s*[-•*]\s*\d+\.\s*',
            # Sub-section numbers like 5.14.2
            r'^Sub-Section\s+\d+(?:\.\d+)*\s*',
        ]
        
        # Document identifiers and headers
        self.document_patterns = [
            r'IW-CDS-\d+-\d+\s*\([^)]+\)',  
            r'^\s*\d+\s*$',  
        ]
        
    def remove_numbering(self, text: str) -> str:
        """
        Remove various types of numbering from the text.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                cleaned_lines.append('')
                continue
                
            # Remove document identifiers if they're standalone
            skip_line = False
            for pattern in self.document_patterns:
                if re.match(pattern + r'\s*$', line):
                    skip_line = True
                    break
            
            if skip_line:
                continue
            
            # Remove numbering patterns from the beginning of lines
            cleaned_line = line
            for pattern in self.numbering_patterns:
                cleaned_line = re.sub(pattern, '', cleaned_line)
            
            # Remove inline section references like "Section 5.14"
            cleaned_line = re.sub(r'\bSection\s+\d+(?:\.\d+)*\b', 'Section', cleaned_line)
            
            # Remove inline sub-section references
            cleaned_line = re.sub(r'\bSub-Section\s+\d+(?:\.\d+)*\b', 'Sub-Section', cleaned_line)
            
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_page_headers_footers(self, text: str) -> str:
        """
        Remove common page headers and footers.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # Skip document ID lines that appear as headers/footers
            if re.search(r'IW-CDS-\d+-\d+.*Revision.*\d{4}', line):
                continue
            
            # Skip standalone page numbers
            if re.match(r'^\s*\d{1,4}\s*$', line):
                # Check if it's likely a page number (isolated number line)
                if i > 0 and i < len(lines) - 1:
                    prev_empty = not lines[i-1].strip()
                    next_empty = not lines[i+1].strip()
                    if prev_empty or next_empty:
                        continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in the text.
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with maximum of two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        
        return '\n'.join(lines)
    
    def fix_word_breaks(self, text: str) -> str:
        """
        Fix words that are broken across lines with hyphens.
        """
        # Fix words broken with hyphens at line ends
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    
    def clean_special_characters(self, text: str, preserve_punctuation: bool = True) -> str:
        """
        Clean special characters while preserving readability.
        """
        # Replace common special characters
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            '\u00a0': ' ',  # Non-breaking space
            '\u200b': '',   # Zero-width space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove control characters but preserve newlines and tabs
        text = ''.join(char for char in text if char in '\n\t' or not ord(char) < 32)
        
        return text
    
    def merge_broken_sentences(self, text: str) -> str:
        """
        Merge sentences that are incorrectly broken across lines.
        """
        lines = text.split('\n')
        merged_lines = []
        
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                merged_lines.append('')
                i += 1
                continue
            
            # Check if current line doesn't end with sentence-ending punctuation
            # and the next line doesn't start with capital letter (likely continuation)
            if (i < len(lines) - 1 and 
                current_line and 
                not current_line[-1] in '.!?:;' and
                lines[i + 1].strip() and 
                not re.match(r'^[A-Z\d\(\[]', lines[i + 1].strip())):
                
                # Merge with next line
                next_line = lines[i + 1].strip()
                merged_lines.append(current_line + ' ' + next_line)
                i += 2
            else:
                merged_lines.append(current_line)
                i += 1
        
        return '\n'.join(merged_lines)
    
    def extract_sections(self, text: str) -> dict:
        """
        Extract sections from the document based on headers.
        """
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a section header (usually in title case or all caps)
            if (len(line.strip()) > 0 and 
                len(line.strip()) < 100 and
                (line.isupper() or 
                 re.match(r'^[A-Z][A-Za-z\s]+$', line.strip()) or
                 'Section' in line)):
                
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def preprocess(self, 
                  text: str,
                  remove_numbers: bool = True,
                  remove_headers: bool = True,
                  normalize_spaces: bool = True,
                  fix_breaks: bool = True,
                  clean_chars: bool = True,
                  merge_sentences: bool = False) -> str:
        """
        Apply all preprocessing steps to the text.
            text: Input text
            remove_numbers: Remove numbering
            remove_headers: Remove page headers/footers
            normalize_spaces: Normalize whitespace
            fix_breaks: Fix word breaks
            clean_chars: Clean special characters
            merge_sentences: Merge broken sentences
        """
        if remove_numbers:
            text = self.remove_numbering(text)
        
        if remove_headers:
            text = self.remove_page_headers_footers(text)
        
        if fix_breaks:
            text = self.fix_word_breaks(text)
        
        if clean_chars:
            text = self.clean_special_characters(text)
        if merge_sentences:
            text = self.merge_broken_sentences(text)
        if normalize_spaces:
            text = self.normalize_whitespace(text)
        return text
    
    def process_file(self, 
                    text: str,
                    output_path: Optional[str] = None,
                    **kwargs) -> str:
        """
        Process text file.
        """
        # Read input file
        # with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        #     text = f.read()
        output_path = "/AgOCQs/output/processed_text.txt"
        # Process text
        processed_text = self.preprocess(text, **kwargs)
        df= pd.DataFrame([processed_text], columns=['sentences'])
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(df)
            print(f"Processed text saved to: {output_path}")
        
        return processed_text