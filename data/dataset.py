import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

import random
import tqdm

class WhataboutismDatasetUnlabeled(Dataset):
    def __init__(self, comments, comments_to_related):
        self.comments = comments 
        self.comments_to_related = comments_to_related
    
    def __getitem__(self, idx:int):
        
        comment = self.comments[idx]
        context = list(np.random.choice(self.comments_to_related[comment], size=5))

        return comment, context 

    def __len__(self):
        return len(self.comments)        


class WhataboutismDataset(Dataset):

    def __init__(self, comments, labels, topics, titles, ids, context, df, test, idx, test_comments=None, aug_to_idx=None, random=False, agnostic=False, title=False):

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
        
      
        self.df = df 
    
        
        self.test = test
        self.idx = idx
        self.context_comments = []
   
        if self.context:
            self.aug_to_idx = aug_to_idx
            
            self.test_comments=test_comments
            self.comments_to_idx = {}
            for idx, comment in enumerate(tqdm.tqdm(self.comments)):            
                sim_idx = eval(self.df.iloc[idx]["Index"]) # get sim idx for top
                topic = self.topics[idx]
                if len(sim_idx) == 0:
                    sim_idx = eval(self.aug_to_idx[comment])

                sim_idx_intersect = []
                topic_at_idx = self.df.iloc[sim_idx].index.values
                label_at_idx = self.df.iloc[sim_idx]["Label"].values
                
                
                for i, topic_idx, label_idx in zip(sim_idx, topic_at_idx, label_at_idx): 
                    if i in self.idx  :  
                        ratio = i / len(sim_idx)
                        if agnostic:
                            if label_idx != self.labels[idx] and ratio <= 0.1 and topic_idx != topic: # MOST SIMILAR COMMENT WITH A DIFFERENT LABEL, HARD-EXAMPLE MINER  
                                sim_idx_intersect.append(i)       
                        else:
                            if label_idx != self.labels[idx] and ratio <= 0.1: # MOST SIMILAR COMMENT WITH A DIFFERENT LABEL, HARD-EXAMPLE MINER  
                                sim_idx_intersect.append(i)            
                       
                self.comments_to_idx[comment] = self.df.iloc[sim_idx_intersect]["Comments"].values[0:5]
                self.context_comments.extend( self.df.iloc[sim_idx_intersect]["Comments"].values[0:5] )
            
            self.select_indices = np.zeros_like(self.titles)
          
            intersect = len(np.intersect1d(self.context_comments, self.test_comments))
            print(intersect)
        self.random = random
        self.title = title
  
    def __getitem__(self, idx:int):

        label = self.labels[idx]
        comment = self.comments[idx]
     
        topic = self.topics[idx]
        

        # Generate another random comment
        if self.context:
            context_comments = self.comments_to_idx[comment]
         
            if len(context_comments) > 0 and self.random == False:
              
                context=  context_comments[0] # need to deterministically generate better context, maybe use current embeddings instead for paraings
            else: 
                
                context_comments = self.df.iloc[self.idx]["Comments"].values 
                context = np.random.choice(context_comments, size=1)[0]
            
            if not self.title:
                return comment, label, context
            else:
                return comment, label, self.titles[idx]


        else:
            return comment, label
    def __len__(self):

        return len(self.titles)

