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
import random


class SelfSupervisedContextSentenceTransformer(pl.LightningModule):

    def __init__(self, pretrained_model, pretrained_tokenizer,  train_set, test_set, unlabel_test_set, learning_rate=0.1, batch_size=8, beta=0.99, gamma=2.5, class_num=2, context=True, loss="focal", cross=False, unlabel_set=None):
        super().__init__()         
        
        self.sent_transformer = pretrained_model       
        self.tokenizer = pretrained_tokenizer        
              
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
        self.cross = cross

        if self.cross:
            self.classifier = nn.Linear(384, 2) 
        else: 
            self.classifier = nn.Linear(768, 2) #MLP Classifier

        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.unlabel_testing = False
        self.unlabel_set = unlabel_set 
        self.unlabel_test_set = unlabel_test_set

        if self.unlabel_set is not None: 
            self.unlabel_loader = DataLoader(self.unlabel_set, batch_size=self.batch_size, shuffle=False)
      

        self.loss = loss
    
    def train_dataloader(self):
        """
            Returns the train data-loader, MANDATORY for PyTorch-Lightning to work.
                (1) Set num_workers equal to number of CPUs
                (2) Shuffle= True as this is the train_set
                (3) collate_fn: To gather all the batches
                (4) pin_memory: True to increase performance
        """
        unlabel = DataLoader(self.unlabel_test_set, batch_size=self.batch_size, shuffle=False)
        train = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        loaders = {"unlabel": unlabel, "train":train}
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loader # we do not need to use the train set anymore, we just need to use the unlabel set to train, since we do not have labels we need metric to establish the regression loss
    
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
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False) # we test on the test set to verify if self-supervised learning is working
    
    def configure_optimizers(self):   
        """
            Configure the optimizers and LR scheduler to better train the model
                (1) Optimizer: Can be Adam or SGD (etc)
                (2) Scheduler: Step-wise LR
        """     
        params =  self.parameters()
        opt =  torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-7)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
        
        return [opt], [sch]
    
    def get_comment_tokens(self, sentences, device):
        with torch.no_grad():            
            inputs = self.tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200)            
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        
        return inputs
    
    def get_sim(self, context_embs, context_embs_label, labels_labeled, topk=1):
        final = []
        whataboutism_logits_label = self.classifier(context_embs_label)
        whataboutism_softmax_labels = torch.softmax(whataboutism_logits_label, dim=1)
        for i in range(context_embs.shape[0]):
            ans = []
            for j in range(context_embs_label.shape[0]):
                similarity = torch.cosine_similarity(context_embs[i].view(1,-1), 
                                                    context_embs_label[j].view(1,-1)).item()   # novelty here, need to find better measurement to get better pseudo-labels, invariants calculations
                ans.append(similarity)
            final.append(ans)
        
        cos_sim_mat = torch.from_numpy(np.array(final))
        cos_sim_mat_top_k, cos_sim_mat_top_k_indices = torch.topk(cos_sim_mat, k=topk, largest=True)  
       
        desired_scores = []
        desired_labels = []
        total_flips = 0
        for sim_score, same_idx in zip(cos_sim_mat_top_k, cos_sim_mat_top_k_indices):
            softmax_scores = whataboutism_softmax_labels[same_idx]      
            desired_score = torch.mean(softmax_scores, dim=-2)
           
            
            if sim_score > 0.6:
                pseudo_label = torch.mode(labels_labeled[same_idx]).values
            else: 
                total_flips+=1
                flip = random.uniform(0, 1)
                pseudo_label = 1 - torch.mode(labels_labeled[same_idx]).values if flip <= 0.18 else torch.mode(labels_labeled[same_idx]).values
            

            desired_labels.append(pseudo_label)
            desired_scores.append(desired_score)

        desired_scores = torch.stack(desired_scores) 
        desired_labels = torch.stack(desired_labels)  

        return desired_scores, desired_labels, total_flips

    def training_step(self, batch: dict, _batch_idx: int):
        comments, labels, opp_comment = batch["unlabel"]        
        comments_label, labels_labeled, opp_comment_label = batch["train"]  
        
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
        
        #1. Build random context compares
        whataboutism_logits = self.classifier(context_embs)  

        #Now sample a few from the train-set and check similarity scores, we get the labels of those that are reasonable 
        comment_tokens = self.get_comment_tokens(comments_label, labels.device)
        comment_embs = self.sent_transformer(**comment_tokens)
        comment_embs = comment_embs["pooler_output"]
        
        #2. Push Context 
        transcript_tokens = self.get_comment_tokens(opp_comment_label, labels.device)
        transcript_embs = self.sent_transformer(**transcript_tokens)
        transcript_embs = transcript_embs["pooler_output"]

        #3. Push 2nd Context 
        context_embs_label = torch.hstack((comment_embs, transcript_embs))
        # Now we perform mse-loss between the whataboutism logits softmax score and the "desired score", which is the avg. of the scores by the sim
        desired_scores_inter, desired_labels_inter, total_flips = self.get_sim(context_embs, context_embs_label, labels_labeled) # label inheritance
        #desired_scores_intra, desired_labels_intra = self.get_sim(context_embs, context_embs, labels_labeled, topk=2)

        # Challenge: Pseduo-labels need to be good enough and Loss Function Need to Be Good Enough so that we can beat 91.10 F1 maybe get into the uppper-90s
     
        labels_occurence = list(np.bincount(desired_labels_inter.cpu().numpy()))     
        if self.loss == "softmax" or self.loss == "focal": # another novelty here, since the distribution we have is wrong, we need to make a correction somehow
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
        
        preds = torch.softmax(whataboutism_logits, dim=1)[:, 1]
        probs = torch.softmax(whataboutism_logits, dim=1)[:, 1].cpu()

        preds[torch.where(preds >= 0.5)] = 1.0 
        preds[torch.where(preds < 0.5)] = 0.0
        
        self.val_preds.extend(preds.cpu().tolist())
        self.val_labels.extend(labels.cpu().tolist())        
        self.val_embs.extend(context_embs.cpu().tolist())
        self.val_comments.extend(comments)
        self.val_probs.extend(probs.cpu().tolist())
    
    def on_validation_epoch_end(self):
        self.val_accuracy = accuracy_score(self.val_labels, self.val_preds)*100
        self.val_f1 = f1_score(self.val_labels, self.val_preds)*100
       
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
        self.log("best-f1", self.best_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
   
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