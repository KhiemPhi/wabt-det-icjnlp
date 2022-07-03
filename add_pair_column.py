from builtins import breakpoint
import pandas as pd
import numpy as np
from langdetect import detect
from utils import preprocess_clean
import re
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def get_comment_tokens(sentences, device, tokenizer):
    with torch.no_grad():         
        # ToDO: Smart batching   
        inputs = tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200, truncation=True)            
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
    
    return inputs

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def is_en(txt):
    try:
        return detect(txt)=='en'
    except:
        return False

def clean(txt):
    return preprocess_clean(txt)


labeled_df = pd.read_csv('annotations_1645.csv')


device = "cuda:2"
sent_transformer = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")    
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")  



embeddings = []
sent_transformer.to(device)
for i in tqdm( labeled_df["Comments"] ):
    tokens = get_comment_tokens([i], device, tokenizer)
    embs = sent_transformer(**tokens)["pooler_output"].detach().cpu().numpy()
    embeddings.append(embs)

embeddings = np.vstack(embeddings)
sim_mat = cosine_similarity(embeddings)

most_sim_idx_all = []

for idx, i in enumerate(sim_mat):    
    most_sim_idx = np.argsort(i)[::-1][1:] # most similar 
    most_sim_idx_all.append(list(most_sim_idx)) 

labeled_df["Index"] = most_sim_idx_all

labeled_df.to_csv('annotations_1645_sim_idx.csv', index=False)
