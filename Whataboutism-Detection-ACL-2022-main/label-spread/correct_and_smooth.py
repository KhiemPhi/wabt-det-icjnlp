import copy
import csv
import json
import random
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn import svm
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score)
from sklearn.metrics.pairwise import check_paired_arrays, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from transformers import (DebertaModel, DebertaTokenizer, DebertaV2Model,
                          DebertaV2Tokenizer)

from builtins import breakpoint
import pandas as pd
import numpy as np
from langdetect import detect

import re
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def get_tfidf_mapping(filename, ignore_words):
    df = pd.read_csv(filename)
    mapping = {}
    for index, row in df.iterrows():
        if index > 300:
            break
        if row['term'] in ignore_words:
            continue
        mapping[row['term']] = row['normalized_score']
    return mapping


def get_tfidf_scores(comments, mapping, norm=True):
    scores = []
    for comment in comments:
        comment = comment.translate(str.maketrans('', '', string.punctuation)).lower()  # strip punctuation
        comment_split = nltk.word_tokenize(comment)
        score = 0
        for word in comment_split:
            if word in mapping:
                score += mapping[word]
        if norm:
            scores.append(score / len(comment_split))
        else:
            scores.append(score)
    return scores

def load_comments(csv_path):
    df = pd.read_csv(csv_path)
    all_comments = df[['Comments']].values.squeeze()
    all_labels = df[['Label']].values.squeeze()
    all_topic = df[['Topic']].values.squeeze()
    positive_comments = df.loc[df['Label'] == 1][['Comments']].values.squeeze()
    negative_comments = df.loc[df['Label'] == 0][['Comments']].values.squeeze()
    return all_comments, all_labels, positive_comments, negative_comments, all_topic

def scale_scores(scores):
    scores = np.array(scores)
    standardized_scores = (scores - scores.mean()) / scores.std()  # Scale to standard Gaussian
    sigmoid_scores = 1 / (1 + np.exp(-standardized_scores))  # Apply sigmoid to convert to probabilities
    norm_scores = (sigmoid_scores - min(sigmoid_scores)) / (
                max(sigmoid_scores) - min(sigmoid_scores))  # Minimum prob appears to be 0.5 with sigmoid --> [0, 1]
    return norm_scores


def cheap_classifier(X_train, y_train, X_test, y_test):
    cheap_clf = svm.SVC(probability=True)
    cheap_clf.fit(X_train, y_train)
    y_pred = cheap_clf.predict(X_test)
    cheap_clf_acc = accuracy_score(y_test, y_pred)
    cheap_clf_f1 = f1_score(y_test, y_pred)
    return cheap_clf, cheap_clf_acc, cheap_clf_f1

def cheap_lp_model(features, labels, label_index):
    default_labels = copy.deepcopy(labels)
    all_index = np.array(range(len(labels)))
   
    unlabeled_indices = np.setdiff1d(all_index, label_index)
    new_labels = np.copy(labels)
    new_labels[unlabeled_indices] = -1

    lp_model = LabelSpreading(kernel='rbf', gamma=20, max_iter=100, alpha=0.7)

    lp_model.fit(features, new_labels)
    y_pred = lp_model.predict(features)
    y_proba = lp_model.predict_proba(features)

    test_labels = np.delete(default_labels, label_index)
    y_pred_unlabeled = np.delete(y_pred, label_index)

    acc = accuracy_score(y_pred_unlabeled, test_labels) * 100
    f1 = f1_score(y_pred_unlabeled, test_labels)*100
    
    return acc, f1, lp_model, label_index


def correct_and_smooth_label_propagation(features, labels, comments, num_labels=9, folds=1):
    default_labels = copy.deepcopy(labels)
    all_index = np.array(range(len(labels)))
    all_positive_labels = np.where(labels == 1)
    all_negative_labels = np.where(labels != 1)
    
    best_acc = 0
    best_labels = None
    best_fold = 0
    best_comments = None
    worst_comments = None
    worst_labels = None
    # Add in average accuracy

    # Form one-hot encoding for correct + smoothing
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(labels.reshape(-1,1))
    one_hot_encoded_labels = one_hot_encoder.transform(labels.reshape(-1,1)).toarray()
    
    

    for i in range(folds):
        supervised_positive_labels = np.random.choice(all_positive_labels[0], num_labels, replace=False)
        supervised_negative_labels = np.random.choice(all_negative_labels[0], num_labels, replace=False)        
        
        label_index = np.hstack((supervised_positive_labels, supervised_negative_labels)) 
        unlabeled_index = np.setdiff1d(all_index, label_index)

        # 1. Classify with cheap classifer        
        acc_cheap, f1_cheap, cheap_clf, labeled_index_cheap = cheap_lp_model(features, labels, label_index)
        

        # 2. Form Error Matrix + Spread Error Using Label Spreading
        cheap_proba = cheap_clf.predict_proba(features[labeled_index_cheap])
        cheap_proba_error = one_hot_encoded_labels[labeled_index_cheap] - cheap_proba
        error_matrix = np.zeros_like(one_hot_encoded_labels)
        error_matrix[labeled_index_cheap] = cheap_proba_error

        error_propagator_masked_labels = np.copy(labels)
        error_propagator_masked_labels[unlabeled_index] = -1
        
        error_propagator = LabelSpreading(kernel='rbf', gamma=20, max_iter=100, alpha=0.01)
        
        error_propagator.fit(error_matrix, error_propagator_masked_labels)          
        error_propagator_out = error_propagator.label_distributions_
        
        #Autoscale The Error with sigma        
        sigma = np.mean(np.abs(error_matrix[label_index] ))
        scale = sigma / np.sum(np.abs(error_propagator_out))
      
        
        cheap_proba_all = cheap_clf.label_distributions_ 
        
        #error_propagator_out[unlabeled_index] = cheap_proba_all[unlabeled_index] 
        correct_prediction = (cheap_proba_all + error_propagator_out) / 2
         
        #normalizer = np.sum(correct_prediction, axis=1)[:, np.newaxis]
        #normalizer[normalizer == 0] = 1
        #correct_prediction /= normalizer
       
        
        corrected_labels = np.argmax(correct_prediction,axis=1)
       
       
        #Smooth Out Errors with One More Label Propagation Step
        smooth_labels = np.copy(labels)
        smooth_labels[unlabeled_index] = corrected_labels[unlabeled_index]
        lp_model = LabelSpreading(kernel='rbf', gamma=200, max_iter=500, alpha=0.01)
        

        lp_model.fit(correct_prediction, smooth_labels)
        

        y_pred = lp_model.predict(correct_prediction)
        y_proba = lp_model.predict_proba(correct_prediction)
       

        test_labels = np.delete(default_labels, label_index)
        y_pred_unlabeled = np.delete(y_pred, label_index)

        acc = accuracy_score(y_pred_unlabeled, test_labels) * 100
        f1 = f1_score(y_pred_unlabeled, test_labels)*100
        
       
        if acc > best_acc:
            best_acc = acc
            best_labels = smooth_labels
            best_fold = i + 1
            worst_acc = acc
            worst_comments = comments[label_index]
            worst_labels = default_labels[label_index]
            best_comments = comments[label_index]
        if worst_acc > acc:
            worst_acc = acc
            worst_comments = comments[label_index]
            worst_labels = default_labels[label_index]

    return y_pred, y_proba, best_comments, best_acc, f1, worst_comments, worst_acc, best_labels, label_index, acc_cheap, f1_cheap, cheap_proba_all[:, 1]


def combine_metrics(label_probs, tfidf_probs, labels, alpha):
    combined_probs = alpha * label_probs + (1 - alpha) * tfidf_probs
    preds = []
    for prob in combined_probs:
        if prob > 0.5:
            preds.append(1)
        else:
            preds.append(0)
    preds = np.asarray(preds)
    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds) * 100
    return preds, acc, f1


data = "annotations_1500.csv"
similarity_file = "annotations_200_sim.csv"
all_comments, all_labels, positive_comments, negative_comments, all_topic = load_comments(data)


def get_comment_tokens(sentences, device, tokenizer):
    with torch.no_grad():         
        # ToDO: Smart batching   
        inputs = tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=200, truncation=True)            
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
    
    return inputs


#features = np.array(pd.read_csv(similarity_file, header=None))

device = "cuda:2"
sent_transformer = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")    
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")  

labeled_df = pd.read_csv('annotations_1500.csv')

embeddings = []
sent_transformer.to(device)
for i in tqdm( labeled_df["Comments"] ):
    tokens = get_comment_tokens([i], device, tokenizer)
    embs = sent_transformer(**tokens)["pooler_output"].detach().cpu().numpy()
    embeddings.append(embs)



features = np.vstack(embeddings)



all_data_index = random.shuffle(list(range(len(features))))
total_train_pts = int(0.8 * len(features))

initial_labels = 0.05
total_labels = int(initial_labels * len(all_comments) / 2)
print(total_labels)

K = 100  # consider experiments with 500, 1000, 2000, etc.

avg_acc = 0
avg_f1 = 0
avg_f1_cheap = 0
avg_acc_cheap = 0

best_acc_cheap = 0
best_f1_cheap = 0

best_acc = 0
best_f1 = 0
best_seed = []
best_label_seed = []
best_label_index_seed = []
best_y_pred = []
best_y_prob = []


for i in range(K):
    
    y_pred, y_proba, best_comments, acc, f1, worst_comments, worst_acc, best_labels, label_index, acc_cheap, f1_cheap, cheap_proba = correct_and_smooth_label_propagation(features, all_labels,all_comments, num_labels=total_labels, folds=1)
    avg_acc += acc
    avg_f1 += f1
    avg_f1_cheap += f1_cheap
    avg_acc_cheap += acc_cheap

    if (acc > best_acc_cheap) and f1 > best_f1_cheap:
        best_acc_cheap = acc_cheap
        best_f1_cheap = f1_cheap
        best_seed = best_comments
        best_label_seed = best_labels
        best_label_index_seed = label_index
        best_y_pred = y_pred
        best_y_prob = y_proba
        
    
    if (acc > best_acc) and f1 > best_f1:
        best_acc = acc
        best_f1 = f1
    
failures_cases_index_best = np.where(best_y_pred != all_labels)
failures_cases_comment = all_comments[failures_cases_index_best]
failures_cases_pred_labels = best_y_pred[failures_cases_index_best]
failures_cases_true_labels = all_labels[failures_cases_index_best]
wrong_y_prob = best_y_prob[failures_cases_index_best]
print(classification_report(all_labels, best_y_pred, target_names=["Not Whataboutism", "Whataboutism"]))
print(f1_score(all_labels, best_y_pred))

avg_acc_cheap /= K
avg_f1_cheap /= K
avg_acc_cheap = str(avg_acc_cheap)
avg_f1_cheap = str(avg_f1_cheap)


print("Label Spreading Avg Accuracy: " + avg_acc_cheap)
print("Label Spreading Avg F1: " + avg_f1_cheap)

avg_acc /= K
avg_f1 /= K
avg_acc = str(avg_acc)
avg_f1 = str(avg_f1)

print("Label Spreading C&S Avg Accuracy: " + avg_acc)
print("Label Spreading C&S Avg F1: " + avg_f1)


