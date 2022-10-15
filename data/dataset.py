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

class EmotionDataset(Dataset):
    def __init__(self, comments, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.comments = comments  
    def __getitem__(self, idx:int):
        comment = self.comments[idx]
        label = self.labels[idx]
        return comment, label
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
        self.num_context = 2
        
        self.random = random
        self.title = title
        self.df = df 

        
        self.test = test
        self.idx = idx
        self.context_comments = []

        #Remove Idx of Biden Afghanistan

        self.df = df.set_index('Title')

        
     
        if self.context:
            self.aug_to_idx = aug_to_idx            
            self.test_comments=test_comments
            _, test_idx , _ = np.intersect1d(self.df["Comments"], self.test_comments, return_indices=True)
            self.comments_to_idx = {}
            self.comments_to_context_label = {}
            
            for idx, comment in enumerate(tqdm.tqdm(self.comments)): 
                        
                sim_idx = range(len(df)) # get sim idx for top
                
                title = self.titles[idx]

                
                
                if len(sim_idx) == 0:
                    sim_idx = eval(self.aug_to_idx[comment])

                
                # sim_label_intersect = []
                # topic_at_idx = self.df.iloc[sim_idx].index.values
                # label_at_idx = self.df.iloc[sim_idx]["Label"].values
                
                topic_at_idx = self.df.iloc[sim_idx].index.values # values 
                topic = self.topics[idx]

                invalid_topics = np.where(self.topics != topic)[0]
                invalid_idx = np.hstack((test_idx, invalid_topics))

               
                sim_idx_train = np.setdiff1d(sim_idx, invalid_idx)


                
                zero_index = np.where( self.df.iloc[sim_idx_train]["Label"] == 0 )
                one_index =   np.where( self.df.iloc[sim_idx_train]["Label"] == 1 )
                
                zero_comment = self.df.iloc[sim_idx_train]["Comments"].values[ np.random.choice(zero_index[0], self.num_context)]
                one_comment = self.df.iloc[sim_idx_train]["Comments"].values[  np.random.choice(one_index[0], self.num_context) ]
                
                
                
                self.comments_to_idx[comment] = np.hstack(  ( zero_comment, one_comment)      )
                self.comments_to_context_label[comment] = np.array([0, 0, 1]  )#np.hstack(( self.df.iloc[sim_idx_train]["Label"].values[zero_index], self.df.iloc[sim_idx_train]["Label"].values[one_index]))
                self.context_comments.extend( np.hstack((zero_comment, one_comment, title)) )

            
            self.select_indices = np.zeros_like(self.titles)
          
            intersect = len(np.intersect1d(self.context_comments, self.test_comments))
            print(intersect)
        
  
    def __getitem__(self, idx:int):

        label = self.labels[idx]
        comment = self.comments[idx]
     
        topic = self.topics[idx]
        

        # Generate another random comment
        if self.context:
            context_comments = self.comments_to_idx[comment]
         
            if len(context_comments) > 0 and self.test:
              
                context=  list(context_comments) # need to deterministically generate better context, maybe use current embeddings instead for paraings
                context_label = list(self.comments_to_context_label[comment]       )     
            else: 
                
                context=  list(context_comments[0:self.num_context]) # need to deterministically generate better context, maybe use current embeddings instead for paraings
                context_label = list(self.comments_to_context_label[comment][0:self.num_context]       )     
            
            if not self.title:
                return comment, label, context, context_label
            else:
                return comment, label, self.titles[idx], self.titles[idx]


        else:
            return comment, label
    def __len__(self):

        return len(self.titles)
