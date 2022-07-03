from builtins import breakpoint
import pytorch_lightning as pl 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from loss_fn import CB_loss
from utils import scatter_tSNE
import csv

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report, precision_recall_curve,
PrecisionRecallDisplay)


class ContextSentenceTransformer(pl.LightningModule):

    def __init__(self, train_set, test_set, val_set, learning_rate=0.1, batch_size=8, beta=0.99, gamma=2.5, class_num=2, context=True, loss="focal", cross=False, unlabel_set=None):
        super().__init__()         
        
        self.sent_transformer = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")         
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")            
              
        self.train_set = train_set 
        self.test_set = test_set 
        self.val_set = val_set
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = 0

        self.val_preds = []
        self.val_labels = []
        self.best_f1 = 0

        self.beta = beta 
        self.gamma = gamma
        self.class_num = class_num
        
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.similarity_preds = []

        self.context = context 
        self.cross = False

        if self.cross:
            self.classifier = nn.Linear(384, 2) 
        else: 
            self.classifier = nn.Linear(768, 2) #MLP Classifier

        self.cross_entropy = nn.CrossEntropyLoss()

        self.unlabel_testing = False
        self.unlabel_set = unlabel_set 

        if self.unlabel_set is not None: 
            self.unlabel_loader = DataLoader(self.unlabel_set, batch_size=self.batch_size, shuffle=False)
      

        self.loss = loss
        self.best_epoch = 0
        self.train_loss = []
        self.val_loss = []
    
    def train_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        """
            Returns the test data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= False as this is the test_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def val_dataloader(self):
        """
            Returns the val data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the val_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  self.parameters()
        opt =  torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.1)
        
        return [opt], [sch]
    
    def get_comment_tokens(self, sentences, device):
        with torch.no_grad():         
            # ToDO: Smart batching   
            inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200, truncation=True)       
            for key in inputs.keys():    
                inputs[key] = inputs[key].to(device)
           
        
        return inputs

    def training_step(self, batch: dict, _batch_idx: int):
       
        comments, labels, opp_comment = batch


        if self.cross:
           
            cross_enc = list(zip(comments, opp_comment))
            context_tokens =  self.get_comment_tokens(cross_enc, labels.device)
            context_embs = self.sent_transformer(**context_tokens)["pooler_output"]
        else: 
            # 1. Push Comment
            comment_tokens = self.get_comment_tokens(comments, labels.device)
            comment_embs = self.sent_transformer(**comment_tokens)
            comment_embs = comment_embs["pooler_output"]
            
            #2. Push Context 
            transcript_tokens = self.get_comment_tokens(opp_comment, labels.device)
            transcript_embs = self.sent_transformer(**transcript_tokens)
            transcript_embs = transcript_embs["pooler_output"]

            #3. Push 2nd Context 
            context_embs = torch.hstack((comment_embs, transcript_embs))
            
    
        whataboutism_logits = self.classifier(context_embs) 
        
        labels_occurence = list(np.bincount(labels.cpu().numpy())) 
        
        if self.loss == "softmax" or self.loss == "focal":
            loss = CB_loss(labels, whataboutism_logits, labels_occurence, self.class_num, loss_type=self.loss, beta=self.beta, gamma=self.gamma, device=labels.device)
            self.train_loss.append(loss.cpu().item())
        else: 
            
            loss = self.cross_entropy(whataboutism_logits, labels)
           
            self.train_loss.append(loss.cpu().item())

        return loss
    
    def on_train_epoch_end(self):
        self.epochs += 1    
        self.train_loss = np.mean(self.train_loss)
        self.log("train-loss", self.train_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.train_loss = []

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.similarity_preds = []
        self.val_embs = []
        self.val_comments = []
        self.val_probs = []
    
    def validation_step(self,  batch: dict, _batch_idx: int):
        comments, labels, opp_comment = batch
      
        if self.cross:
            cross_enc = list(zip(comments, opp_comment))
            context_tokens =  self.get_comment_tokens(cross_enc, labels.device)
            context_embs = self.sent_transformer(**context_tokens)["pooler_output"]
            
        else: 
            comment_tokens = self.get_comment_tokens(comments, labels.device)
            comment_embs = self.sent_transformer(**comment_tokens)
            comment_embs = comment_embs["pooler_output"]
            
            transcript_tokens = self.get_comment_tokens(opp_comment, labels.device)
            transcript_embs = self.sent_transformer(**transcript_tokens)
            transcript_embs = transcript_embs["pooler_output"]
            
            similarity = self.similarity(comment_embs, transcript_embs)
            context_embs = torch.hstack((comment_embs, transcript_embs))
            smiliarity_preds = (similarity < 0.5).type(torch.int)
            self.similarity_preds.extend(smiliarity_preds.cpu().tolist())

        whataboutism_logits = self.classifier(context_embs) 
        preds = torch.argmax(whataboutism_logits, dim=1).flatten()
        probs = torch.softmax(whataboutism_logits, dim=1)[:, 1]
        self.val_preds.extend(preds.cpu().tolist())
        self.val_labels.extend(labels.cpu().tolist())        
        self.val_embs.extend(context_embs.cpu().tolist())
        self.val_comments.extend(comments)
        self.val_probs.extend(probs.cpu().tolist())

        labels_occurence = list(np.bincount(labels.cpu().numpy())) 
        if self.loss == "softmax" or self.loss == "focal":
            self.val_loss = CB_loss(labels, whataboutism_logits, labels_occurence, self.class_num, loss_type=self.loss, beta=self.beta, gamma=self.gamma, device=labels.device)
        else:
            self.val_loss = self.cross_entropy(whataboutism_logits, labels)
    
    def on_validation_epoch_end(self):
        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100
        
        if self.val_f1 >= self.best_f1: 

            self.csv_record = open('vis/validation_results_1615.csv', 'w')
            self.writer = csv.writer(self.csv_record)
            self.best_epoch = self.epochs

            report = classification_report(self.val_labels, self.val_preds, target_names=["Non-Whataboutism", "Whataboutism"])
            print(report)
            with open('{}.txt'.format('vis/validation_acc_tab'), 'w') as f:
                print(report, file=f)
        
            self.best_f1 = self.val_f1
            self.val_embs = np.array(self.val_embs)

            # Visualise the results when best is beaten
            path = "vis/tSNE/test-tSNE-epoch-" + str(self.epochs) + ".jpg"
            scatter_tSNE(self.val_embs, np.array(self.val_labels), file_path= path )

            # Visualize the wrong results
            wrong_comments = []
            self.writer.writerow(["Comment", "Label", "Predicted (0=Non-Wabt, 1=Wabt)", "Whataboutism Probability", "Error Type"])
            for test_comment, test_label, test_pred, test_prob in zip(self.val_comments, self.val_labels, self.val_preds, self.val_probs):                        
                if test_label == 1 and test_pred == 0:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Neg"])
                    wrong_comments.append(test_comment)
                elif test_label == 0 and test_pred == 1:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Pos"])
                    wrong_comments.append(test_comment)



        self.log("validation-acc", torch.tensor([self.val_accuracy]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("validation-f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-f1", self.best_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-epoch", self.best_epoch, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val-loss", self.val_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
   
    def test_step(self,  batch: dict, _batch_idx: int):
     
        if self.unlabel_testing:
            
            comments, related = batch 
            
            comment_tokens = self.get_comment_tokens(comments, self.sent_transformer.device)
            comment_embs = self.sent_transformer(**comment_tokens)
            comment_embs = comment_embs["pooler_output"]
            probs_list = []
            for item in related:
                context_tokens = self.get_comment_tokens(item, self.sent_transformer.device)
                context_embs = self.sent_transformer(**context_tokens)
                context_embs = context_embs["pooler_output"]   
                context_embs =  torch.hstack((comment_embs, context_embs))
                whataboutism_logits = self.classifier(context_embs) 
                #preds = torch.argmax(whataboutism_logits, dim=1).flatten()
                probs = torch.softmax(whataboutism_logits, dim=1)[:, 1]
                probs_list.append(probs)
          
            probs = torch.mean(torch.stack(probs_list), dim=0)
            preds = torch.mean(torch.stack(probs_list), dim=0)
            preds[torch.where(preds >= 0.7)] = 1.0 
            preds[torch.where(preds < 0.7)] = 0.0
            probs = probs.cpu().tolist()
            preds = preds.cpu().tolist()
          
            csvfile = open('vis/validation_results_unlabeled.csv', 'w')  
            csvwriter = csv.writer(csvfile) 

            csvwriter.writerow(["Transcript Sentence", "Predicted (0=Non-Wabt, 1=Wabt)", "Whataboutism Probability"])
            for comment, pred, prob in zip(comments, preds, probs):               
                csvwriter.writerow([comment, pred, prob])
    


class SentenceTransformer(pl.LightningModule):
    
    def __init__(self, train_set, test_set, val_set, learning_rate=0.1, batch_size=8, beta=0.99, gamma=2.5, class_num=2, context=True, loss="focal"):
        super().__init__()         
       
        
        self.sent_transformer = AutoModel.from_pretrained("roberta-base")         
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")             #cardiffnlp/twitter-roberta-base-irony
              
        self.train_set = train_set 
        self.test_set = test_set 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = 0

        self.val_preds = []
        self.val_labels = []
        self.best_f1 = 0

        self.beta = beta 
        self.gamma = gamma
        self.class_num = class_num
        
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.similarity_preds = []

        self.context = context 

       
        self.classifier = nn.Linear(768, 2) #MLP Classifier
        self.cross_entropy = nn.CrossEntropyLoss()
      

        self.loss = loss
    
    def train_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        """
            Returns the test data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= False as this is the test_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def val_dataloader(self):
        """
            Returns the val data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the val_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  self.parameters()
        opt =  torch.optim.Adam(params, lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
        
        return [opt], [sch]
    
    def get_comment_tokens(self, sentences, device):
        with torch.no_grad():     
           
            inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200, truncation=True)         
        
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            #inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        
        return inputs
    
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    

    def training_step(self, batch: dict, _batch_idx: int):
       
        comments, labels = batch

        comment_tokens = self.get_comment_tokens(comments, labels.device)
       
        comment_embs = self.sent_transformer(**comment_tokens)
        comment_embs = comment_embs["pooler_output"]
    
       
        whataboutism_logits = self.classifier(comment_embs) 
    
        
        labels_occurence = list(np.bincount(labels.cpu().numpy()))
        if self.loss == "softmax" or self.loss == "focal":
            loss = CB_loss(labels, whataboutism_logits, labels_occurence, self.class_num, loss_type=self.loss, beta=self.beta, gamma=self.gamma, device=labels.device)
        else: 
            loss = self.cross_entropy(whataboutism_logits, labels)

        return loss
    
    def on_train_epoch_end(self):
        self.epochs += 1    

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.similarity_preds = []
        self.val_embs = []
        self.val_comments = []
        self.val_probs = []
    
    def validation_step(self,  batch: dict, _batch_idx: int):
        comments, labels = batch
        comment_tokens = self.get_comment_tokens(comments, labels.device)
        comment_embs = self.sent_transformer(**comment_tokens)
        comment_embs = comment_embs["last_hidden_state"]
        comment_embs = self.max_pooling(comment_embs, comment_tokens['attention_mask'])
       
       
        logits = self.classifier(comment_embs) 
          
        preds = torch.argmax(logits, dim=1).flatten()
        probs = torch.softmax(logits, dim=1).flatten()
        self.val_preds.extend(preds.cpu().tolist())
        self.val_labels.extend(labels.cpu().tolist())        

        self.val_comments.extend(comments)
        self.val_probs.extend(probs.cpu().tolist())
        self.val_embs.extend(comment_embs.cpu().tolist())
    
    def on_validation_epoch_end(self):
        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100
        
        if self.val_f1 >= self.best_f1: 

            self.csv_record = open('vis/validation_results_1615.csv', 'w')
            self.writer = csv.writer(self.csv_record)
            self.best_epoch = self.epochs

            report = classification_report(self.val_labels, self.val_preds, target_names=["Non-Whataboutism", "Whataboutism"])
          
            with open('{}.txt'.format('vis/validation_acc_tab'), 'w') as f:
                print(report, file=f)
        
            self.best_f1 = self.val_f1
            self.val_embs = np.array(self.val_embs)

            # Visualise the results when best is beaten
            # path = "vis/tSNE/test-tSNE-epoch-" + str(self.epochs) + ".jpg"
            # scatter_tSNE(self.val_embs, np.array(self.val_labels), file_path= path )

            # Visualize the wrong results
            wrong_comments = []
            self.writer.writerow(["Comment", "Label", "Predicted (0=Non-Wabt, 1=Wabt)", "Whataboutism Probability", "Error Type"])
            for test_comment, test_label, test_pred, test_prob in zip(self.val_comments, self.val_labels, self.val_preds, self.val_probs):                        
                if test_label == 1 and test_pred == 0:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Neg"])
                    wrong_comments.append(test_comment)
                elif test_label == 0 and test_pred == 1:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Pos"])
                    wrong_comments.append(test_comment)



        self.log("validation-acc", torch.tensor([self.val_accuracy]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("validation-f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-f1", self.best_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-epoch", self.best_epoch, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        