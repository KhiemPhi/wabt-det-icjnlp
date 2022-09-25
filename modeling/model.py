from builtins import breakpoint
from random import shuffle
from turtle import onclick
from typing import final
import pytorch_lightning as pl 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from loss_fn import CB_loss, PrototypeCELoss
from utils import scatter_tSNE
import csv

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report, precision_recall_curve,
PrecisionRecallDisplay)

from datasets import load_dataset
from data import EmotionDataset

from pytorch_lightning.trainer.supporters import CombinedLoader
from timm.models.layers import trunc_normal_

from .proto import ProjectionHead, l2_normalize, momentum_update, distributed_sinkhorn

from einops import rearrange, repeat

class ContextSentenceTransformer(pl.LightningModule):

    def __init__(self, train_set, test_set, val_set, learning_rate=0.1, batch_size=8, beta=0.99, gamma=2.5, class_num=2, context=True, loss="focal", cross=False, unlabel_set=None):
        super().__init__()         
        
        self.sent_transformer = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")         
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")            
              
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
        
        self.ones_prototypes = []
        self.cross_entropy = nn.CrossEntropyLoss()
       
        self.similarity_preds = []

        self.context = context 
        self.cross = cross
      
        if self.cross:
            self.classifier = nn.Linear(384, 2) 
        else: 
            self.classifier = nn.Linear(768, 2)           
     
        self.loss = loss
        self.margin = 5
        
    
    def label_miner(self, context_candidates, candidates_labels):
        return 0
    
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
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        # loaders = {"train": train_loader, "test": test_loader}
        # combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return test_loader
    
    def val_dataloader(self):
        """
            Returns the val data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the val_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        # loaders = {"train": train_loader, "test": test_loader}
        # combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return test_loader
    
    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  self.parameters()
        opt =  torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
        
        return [opt], [sch]
    
    def get_comment_tokens(self, sentences, device):
        with torch.no_grad():         
            # ToDO: Smart batching   
            inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200, truncation=True)       
            for key in inputs.keys():    
                inputs[key] = inputs[key].to(device)
           
        
        return inputs   
    
    def max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embs(self, comments, labels):
        comment_tokens = self.get_comment_tokens(comments, labels.device)
        comment_embs = self.sent_transformer(**comment_tokens)
        sentence_embeddings = self.mean_pooling(comment_embs, comment_tokens['attention_mask'])
        
        return sentence_embeddings
    
    
    def inference(self, whatabout, train=True):
        
        comments, labels, context_comment, context_labels, transcript = whatabout
       
        final_sim_scores = []
        final_gt_scores = []
        context_embs = []

        
        if self.cross:
           
            cross_enc = list(zip(comments, context_comment))
            context_tokens =  self.get_comment_tokens(cross_enc, labels.device)
            context_embs = self.sent_transformer(**context_tokens)["pooler_output"]
        else: 
            # 1. Push Comment
            comment_embs = self.get_embs(comments, labels)
            
        
            
            context_embs = []
            other_context = []
            
            context_comment = np.vstack(context_comment)            
            context_labels = torch.vstack(context_labels)

            final_label = [] 
            consistent_logits = []
            consistent_label = []
            distance_loss = []         

            all_matrix = []

          
            EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)

            negative_context = self.get_embs(context_comment[0, :], labels)
            positive_context = self.get_embs(context_comment[1, :], labels)           
           
            

            for idx, comment_emb in enumerate(comment_embs): 
                
             
                distance_matrix_negative = EUCLIDEAN(comment_emb, negative_context)
                distance_matrix_positive = EUCLIDEAN(comment_emb, positive_context)
                distance_matrix = torch.hstack((distance_matrix_negative, distance_matrix_positive))

                if labels[idx] == 1:
                    sim_labels_wabt = torch.zeros_like(distance_matrix_negative)
                    sim_labels_non_wabt = torch.ones_like(distance_matrix_negative)
                    sim_labels = torch.hstack((sim_labels_wabt, sim_labels_non_wabt)).to(labels.device)
                else: 
                    sim_labels_wabt = torch.ones_like(distance_matrix_negative)
                    sim_labels_non_wabt = torch.zeros_like(distance_matrix_negative)    
                    sim_labels = torch.hstack((sim_labels_wabt, sim_labels_non_wabt)).to(labels.device)

             

                if train:                    
                    sim_loss = 0.5 * (sim_labels.float() * distance_matrix.pow(2) + (1 - sim_labels).float() * F.relu(self.margin - distance_matrix).pow(2)).mean()                    
                    distance_loss.append(sim_loss)

                most_same_emb_negative = negative_context[torch.max(distance_matrix_negative, dim=0)[1]]     
                most_diff_emb_negative = negative_context[torch.min(distance_matrix_negative, dim=0)[1]]
                most_same_emb_positive = positive_context[torch.max(distance_matrix_positive, dim=0)[1]]     
                most_diff_emb_positive = positive_context[torch.min(distance_matrix_positive, dim=0)[1]]               
                        
                final_label.append(sim_labels[torch.argmin(distance_matrix)].item()  )

            
                self.avg_distance_positive.append(torch.max(distance_matrix_positive, dim=0)[0].item())
                self.avg_distance_negative.append(torch.max(distance_matrix_negative, dim=0)[0].item())
                
                if train:
                    if labels[idx] == 0:
                        context_embs.append( most_same_emb_positive  )
                    
                    else: 
                        context_embs.append( most_same_emb_negative  )
                else: 
                    context_embs.append(most_same_emb_positive)
              
              
            
               
            context_embs = torch.stack(context_embs)
            
            classifier_embs = torch.hstack((comment_embs, context_embs ))

           
        
        
        whataboutism_logits = self.classifier(classifier_embs)
        whataboutism_labels = torch.argmax(whataboutism_logits, dim=1).cpu().tolist()
        
      

        if train:
            
            return whataboutism_logits, torch.stack(distance_loss).mean()
        else: 
            return whataboutism_logits, classifier_embs, final_label, whataboutism_labels

    def calculate_loss(self, whataboutism_logits, labels, labels_occurence):
        if self.loss == "softmax" or self.loss == "focal":
            loss = CB_loss(labels, whataboutism_logits, labels_occurence, self.class_num, loss_type=self.loss, beta=self.beta, gamma=self.gamma, device=labels.device)            
        else: 
            loss = self.cross_entropy(whataboutism_logits, labels)
        return loss
    
    def on_train_epoch_start(self):
        self.avg_sim_scores = []
        self.ones_prototypes = []
        self.zeros_prototypes = []
        self.avg_distance_positive = []
        self.avg_distance_negative = []


    def training_step(self, batch: dict, _batch_idx: int):        
        comments, labels, opp_comment, context_labels, transcript = batch     
        # one comment can have one of five contexts
        
        samples_per_cls = list(np.bincount(labels.cpu().numpy().astype('int64')))  
       
        
        whataboutism_logits, sim_loss = self.inference(batch)  
     

       
        
        classifier_loss = self.calculate_loss(whataboutism_logits, labels, samples_per_cls)
        #aux_loss = self.calculate_loss(aux_logits, labels, labels_occurence )
       
        return  classifier_loss + (sim_loss * 0.1)
    
    def on_train_epoch_end(self):        
        self.epochs += 1  
       
    

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.similarity_preds = []
        self.val_embs = []
        self.val_comments = []
        self.val_probs = []

        self.train_preds = []
        self.train_labels = []
      

        
    
    def validation_step(self,  batch: dict, _batch_idx: int):       
       
        comments_test, labels_test, opp_comment_test, context_labels_test, transcript = batch
      
        whataboutism_logits_test, context_embs, final_sim_labels_test, pred_labels = self.inference(batch, train=False)  
        #whataboutism_logits_train, aux_logits, context_embs_train, final_sim_labels_train = self.inference(batch["test"], train=False)  
        
        probs = torch.softmax(whataboutism_logits_test, dim=1)[:, 1].cpu().tolist()
        
        self.val_preds.extend(pred_labels)
        self.val_labels.extend(labels_test.cpu().tolist())        
        self.val_embs.extend(context_embs.cpu().tolist())
        self.val_comments.extend(comments_test)
        self.val_probs.extend(probs)

        self.similarity_preds.extend(final_sim_labels_test)

        
    def on_validation_epoch_end(self):
        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100

        #self.train_f1 = f1_score(self.train_labels, self.train_preds)*100

        self.sim_f1 = f1_score(self.val_labels, self.similarity_preds)*100
        
        if self.val_f1 > self.best_f1: 

            self.csv_record = open('vis/validation_results_1342.csv', 'w')
            self.writer = csv.writer(self.csv_record)
        
            self.best_f1 = self.val_f1
            self.val_embs = np.array(self.val_embs)

            # Visualise the results when best is beaten
            path = "vis/tSNE/test-tSNE-epoch-" + str(self.epochs) + ".jpg"
            scatter_tSNE(self.val_embs, np.array(self.val_labels), file_path= path )

            # Visualize the wrong results
            wrong_comments = []
            for test_comment, test_label, test_pred, test_prob in zip(self.val_comments, self.val_labels, self.val_preds, self.val_probs):                        
                if test_label == 1 and test_pred == 0:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Neg"])
                    wrong_comments.append(test_comment)
                elif test_label == 0 and test_pred == 1:
                    self.writer.writerow([test_comment, test_label, test_pred, test_prob,  "False Pos"])
                    wrong_comments.append(test_comment)



        self.log("validation-acc", torch.tensor([self.val_accuracy]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("validation-f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("sim-f1", self.sim_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("best-f1", self.best_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)



class ContextSentenceTransformerMultiTask(pl.LightningModule):

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
        self.regression_loss = nn.L1Loss()
        self.similarity_preds = []

        self.context = context 
        self.cross = False

        if self.cross:
            self.classifier = nn.Linear(384, 2) 
            
        else: 
            self.classifier = nn.Linear(384*2, 2) #MLP Classifier
            self.emotion_classifier = nn.Linear(384, 6)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.unlabel_testing = False
        self.unlabel_set = unlabel_set 

        if self.unlabel_set is not None: 
            self.unlabel_loader = DataLoader(self.unlabel_set, batch_size=self.batch_size, shuffle=False)
      
        emotion_dataset = load_dataset("emotion")
        self.emotion_train = EmotionDataset(emotion_dataset['train']['text'], emotion_dataset['train']['label'])
        self.emotion_test =  EmotionDataset(emotion_dataset['test']['text'], emotion_dataset['test']['label'])

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
        emotion_loader = DataLoader(self.emotion_train, batch_size=self.batch_size, shuffle=True)
        whatabout_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        
        return {"emotion": emotion_loader, "whatabout": whatabout_loader}
    
    def test_dataloader(self):
        """
            Returns the test data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= False as this is the test_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        emotion_loader = DataLoader(self.emotion_test, batch_size=self.batch_size, shuffle=False)
        whatabout_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        loaders = {"emotion": emotion_loader, "whatabout": whatabout_loader}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders 
    
    def val_dataloader(self):
        """
            Returns the val data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the val_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        emotion_loader = DataLoader(self.emotion_test, batch_size=self.batch_size, shuffle=False)
        whatabout_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        loaders = {"emotion": emotion_loader, "whatabout": whatabout_loader}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders 
    
    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  self.parameters()
        opt =  torch.optim.Adam(params, lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.1)
        
        return [opt], [sch]
    
    def get_comment_tokens(self, sentences, device):
        with torch.no_grad():         
            # ToDO: Smart batching   
            inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200, truncation=True)       
            for key in inputs.keys():    
                inputs[key] = inputs[key].to(device)
           
        
        return inputs
    
    def inference(self, whatabout, emotion, train=True):
       
        comments, labels, opp_comment, context_labels = whatabout
        comment_emotion, label_emotion = emotion

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
            
            opp_comments_compress = []
            for i in range(len(comments)):
                opp_comments_compress.append(opp_comment[0][i])
            
            
            context_tokens = self.get_comment_tokens(opp_comments_compress, labels.device)
            context_embs = self.sent_transformer(**context_tokens)
            context_embs = context_embs["pooler_output"]

            #3. Get Emotion Context
            comment_emotion_tokens = self.get_comment_tokens(comment_emotion, labels.device)
            comment_emotion_embs = self.sent_transformer(**comment_emotion_tokens)
            comment_emotion_embs = comment_emotion_embs["pooler_output"]


            #3. Push 2nd Context 
            context_embs = torch.hstack((comment_embs, context_embs ))
            
        whataboutism_logits =  self.classifier(context_embs)
       
        emotion_logits = self.emotion_classifier(comment_emotion_embs)

        if train:
            return whataboutism_logits,  emotion_logits
        else: 
            return whataboutism_logits, emotion_logits, context_embs

    def calculate_loss(self, whataboutism_logits, emotion_logits, labels, label_emotion, labels_occurence, labels_occurence_emotion):
        if self.loss == "softmax" or self.loss == "focal":
           
            loss = CB_loss(labels, whataboutism_logits, labels_occurence, self.class_num, loss_type=self.loss, beta=self.beta, gamma=self.gamma, device=labels.device)
            
            loss_emotion = self.cross_entropy(emotion_logits, label_emotion)
            
            total_loss = (0.95)*loss + (0.05)*loss_emotion          
            
        else: 
            loss = self.cross_entropy(whataboutism_logits, labels)
            
            loss_emotion = self.cross_entropy(emotion_logits, label_emotion)
            total_loss = (0.95)*loss + (0.05)*loss_emotion
            
        
        return total_loss

    def training_step(self, batch: dict, _batch_idx: int):

        
        comments, labels, opp_comment, context_labels = batch["whatabout"]
        comment_emotion, label_emotion = batch["emotion"]

        # one comment can have one of five contexts
        whataboutism_logits, emotion_logits = self.inference(batch["whatabout"], batch["emotion"])       
        
        labels_occurence = list(np.bincount(labels.cpu().numpy())) 

        
        labels_occurence_emotion = []
        for x in range(6):
            labels_occurence_emotion.append( len(torch.where(label_emotion == x)[0]) )
        
               
        total_loss = self.calculate_loss(whataboutism_logits, emotion_logits, labels, label_emotion, labels_occurence, labels_occurence_emotion)
        self.train_loss.append(total_loss.cpu().item())

        return total_loss
    
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
        
        self.emotion_preds = []
        self.emotion_labels = []
    
    def validation_step(self,  batch: dict, _batch_idx: int):
     
        comments, labels, opp_comment, context_labels = batch["whatabout"]
        comment_emotion, label_emotion = batch["emotion"]
        
        # one comment can have one of five contexts
        whataboutism_logits, emotion_logits, context_embs = self.inference(batch["whatabout"], batch["emotion"], train=False)    
        labels_occurence = list(np.bincount(labels.cpu().numpy())) 
        labels_occurence_emotion = list(np.bincount(label_emotion.cpu().numpy()))       
        
        preds = torch.argmax(whataboutism_logits, dim=1).flatten()     
        probs = torch.softmax(whataboutism_logits, dim=1)[:, 1]

       
        emotion_pred = torch.argmax(emotion_logits, dim=1).flatten()
        
        self.val_preds.extend(preds.cpu().tolist())
     
        self.emotion_preds.extend(emotion_pred.cpu().tolist())
        self.val_labels.extend(labels.cpu().tolist())        
        self.val_embs.extend(context_embs.cpu().tolist())
        self.val_comments.extend(comments)
        self.val_probs.extend(probs.cpu().tolist())
        self.emotion_labels.extend(label_emotion.cpu().tolist())

        labels_occurence = list(np.bincount(labels.cpu().numpy())) 
        
        self.val_loss = self.calculate_loss(whataboutism_logits, emotion_logits, labels, label_emotion, labels_occurence, labels_occurence_emotion)
    
    def on_validation_epoch_end(self):

        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100

        self.emotion_accuracy = accuracy_score(self.emotion_labels, self.emotion_preds)*100
        
        self.emotion_f1 = f1_score(self.emotion_labels, self.emotion_preds, average='macro')*100

        if self.val_f1 >= self.best_f1: 

            self.csv_record = open('vis/validation_results_1615.csv', 'w')
            self.writer = csv.writer(self.csv_record)
            self.best_epoch = self.epochs

            report = classification_report(self.val_labels, self.val_preds, target_names=["Non-Whataboutism", "Whataboutism"])
            #print(report)
            with open('{}.txt'.format('vis/validation_acc_tab'), 'w') as f:
                print(report, file=f)
        
            self.best_f1 = self.val_f1
            self.val_embs = np.array(self.val_embs)

            # Visualise the results when best is beaten
            path = "vis/tSNE/test-tSNE-epoch-" + str(self.epochs) + ".jpg"
            #scatter_tSNE(self.val_embs, np.array(self.val_labels), file_path= path )

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
        self.log("validation-acc-em", torch.tensor([self.emotion_accuracy]), prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("validation-f1-em", self.emotion_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
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
    

class ProtoTransformer(ContextSentenceTransformer): 

    def __init__(self, train_set, test_set, val_set, learning_rate=0.1, batch_size=8, beta=0.99, gamma=2.5, class_num=2, context=True, loss="focal", cross=False, unlabel_set=None, num_prototype=5):
        super().__init__(train_set, test_set, val_set, learning_rate, batch_size, beta, gamma, class_num, context, loss, cross, unlabel_set)

        
        
        self.sent_transformer = AutoModel.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")         
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")

        
               
        # Now I add prototypes 
        self.num_prototype = num_prototype
        self.embedding_dim = 768 # embedding size of Sentence-BERT
        self.num_classes = 2
        self.loss = PrototypeCELoss(self.beta, self.num_classes)

        self.prototypes =  nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, self.embedding_dim),
                                       requires_grad=True)
        self.cls_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), 
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.feat_norm = nn.LayerNorm(self.embedding_dim)
        self.class_norm = nn.LayerNorm(self.num_classes)
        self.proj_head = ProjectionHead(self.embedding_dim, self.embedding_dim)
        
        trunc_normal_(self.prototypes, std=0.02)
    
    def prototype_learning(self, embs, out_logits, labels, masks):
        pred_seg = torch.max(out_logits, 1)[1]
        mask = (labels == pred_seg.view(-1))
        
        cosine_similarity = torch.mm(embs, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = labels.clone().float()
        protos = self.prototypes.data.clone()

        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[labels == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[labels == k]

            c_k = embs[labels == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[labels == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=True)

        
        return proto_logits, proto_target


    def inference(self, whatabout, train=True):

        comments, labels, context_comment, context_labels = whatabout
        context_comment = np.vstack(context_comment)
                
        context_labels = torch.vstack(context_labels)
        
        comment_embs = self.get_embs(comments, labels)
       

        comment_embs = self.cls_head(comment_embs)
        comment_embs = self.proj_head(comment_embs)
        comment_embs = self.feat_norm(comment_embs)
        comment_embs = l2_normalize(comment_embs)
      
        masks = torch.einsum('nd,kmd->nmk', comment_embs, self.prototypes)
        out_logits = torch.amax(masks, dim=1) 
        

        self.prototypes.data.copy_(l2_normalize(self.prototypes))
      
    
        out_logits = self.class_norm(out_logits)  # this is the output logits
        out_probs = torch.softmax(out_logits, dim=1) # this is the output probs
        out_preds = torch.argmax(out_logits, dim=1)
        
        # now we do prototype learning         
        proto_logits, proto_target = self.prototype_learning(comment_embs, out_logits, labels, masks)
        
        return out_preds, out_logits, proto_logits, proto_target, out_probs

    def training_step(self, batch: dict, _batch_idx: int):        
        comments, labels, opp_comment, context_labels = batch     
        # one comment can have one of five contexts        
        
        out_preds, out_logits, proto_logits, proto_target, _ = self.inference(batch)  
        
        labels_occurence = list(np.bincount(labels.cpu().numpy()))        
        preds = { "pred_logits": out_logits, "logits": proto_logits, "target": proto_target }
        loss =  self.loss(preds, labels, labels_occurence)
      
        return loss
    
    def validation_step(self,  batch: dict, _batch_idx: int):
        
          
        comments_test, labels_test, opp_comment_test, context_labels_test = batch       
        preds_test, out_logits, proto_logits, proto_target, probs_test = self.inference(batch, train=False)  
        
        
        self.val_preds.extend(preds_test.cpu().tolist())
        self.val_labels.extend(labels_test.cpu().tolist())       
        self.val_comments.extend(comments_test)
        self.val_probs.extend(probs_test.cpu().tolist())

      
    
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