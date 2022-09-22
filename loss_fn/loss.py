from builtins import breakpoint
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


class PPC(nn.Module):
    def __init__(self):
        super(PPC, self).__init__()

        self.ignore_label = -1
       
    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc


class PPD(nn.Module):
    def __init__(self):
        super(PPD, self).__init__()

        self.ignore_label = -1
       

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd



class PrototypeCELoss(nn.Module):
    def __init__(self, beta, no_of_classes):
        super(PrototypeCELoss, self).__init__()

        self.ignore_index = -1
       
        self.loss_ppc_weight = 0.01
        self.loss_ppd_weight = 0.001

        self.beta = 0.9
        self.gamma = 1.5
        self.loss_type = "softmax"
        self.no_of_classes = no_of_classes

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()

    def forward(self, preds, labels, labels_occurence):
        
        pred_logits = preds['pred_logits']        
        contrast_logits = preds['logits']
        contrast_target = preds['target']
        
        loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
        loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

        loss = CB_loss(labels, pred_logits, labels_occurence, self.no_of_classes, self.loss_type, self.beta, self.gamma, pred_logits.device, p=0.8, q=5, eps=1e-2)

        return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd




def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    

    
    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    focal_loss = alpha * loss
    
    #focal_loss = torch.sum(weighted_loss)

    #focal_loss /= torch.sum(labels)

    
   

    
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device, p=0.8, q=5, eps=1e-2):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """


    effective_num = 1.0 - np.power(beta, samples_per_cls)
    samples_per_cls = torch.tensor(samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float().to(device)
    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    '''
    seesaw_weights = logits.new_ones(labels_one_hot.size()) 

    # mitigation factor
    if p > 0:       
        sample_ratio_matrix = samples_per_cls[None, :].clamp(min=1) / samples_per_cls[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]        
        seesaw_weights = seesaw_weights * (mitigation_factor.to(seesaw_weights.device))

    # compensation factor
    if q > 0:
        scores = F.softmax(logits.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * (compensation_factor.to(seesaw_weights.device))

    logits = logits + (seesaw_weights.log().to(labels_one_hot.device) * (1 - labels_one_hot))
    '''
   
    if loss_type == "focal":
        p = torch.sigmoid(logits)
       
        pt = labels_one_hot * p + (1 - labels_one_hot) * (1 - p)
        cb_loss = focal_loss(labels_one_hot, pt, weights, gamma)
        cb_loss = cb_loss + 2.0 * torch.pow(1-pt, gamma+1)
        cb_loss = torch.mean(cb_loss)
        
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, pos_weight = weights) 

    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
      
    return cb_loss