import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class WhataboutismDataset(Dataset):

    def __init__(self, comments, labels, topics, titles, ids, context):

        self.labels = torch.tensor(labels, dtype=torch.long)
       
        self.comments = comments
        self.topics = topics
        self.titles = titles
        self.ids = ids

        self.on_topic_related = 0
        self.on_topic_whataboutism = 0

        self.pos_counts = len( np.where(labels==1)[0] )
        self.neg_counts = len( np.where(labels==0)[0] )

        self.context = context
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")  
    
    def get_comment_embedding(self, sentences):
            with torch.no_grad():            
                inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200)            
                inputs['input_ids'] = inputs['input_ids'].squeeze(1)
                inputs['attention_mask'] = inputs['attention_mask'].squeeze(1)
                inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(1)

    def __getitem__(self, idx:int):

        
        label = self.labels[idx]
        sent_emb = self.get_comment_embedding(self.comments[idx])
     

        return sent_emb, label

    def __len__(self):

        return len(self.titles)

