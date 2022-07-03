from builtins import breakpoint
from utils import load_comments

import pandas as pd
import numpy as np
from scipy import stats

from langdetect import detect

'''
Created on Aug 1, 2016
@author: skarumbaiah
Computes Fleiss' Kappa 
Joseph L. Fleiss, Measuring Nominal Scale Agreement Among Many Raters, 1971.
'''

def checkInput(rate, n):
    """ 
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError 
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer" 
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"

def fleissKappa(rate,n):
    """ 
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category 
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters   
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
   
    #mean of the extent to which raters agree for the ith subject 
    PA = sum([(sum([i**2 for i in row])- n) / (n * (n - 1)) for row in rate])/N
    print("PA = ", PA)
    
    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j**2 for j in [sum([rows[i] for rows in rate])/(N*n) for i in range(k)]])
    print("PE =", PE)
    
    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)
    
    return kappa
paths = ["./dataset/annotations_1342.csv", '/users/kphi/wabt-det-emnlp2022/Ukraine Russia War_266.csv', '/users/kphi/wabt-det-emnlp2022/Biden Afghanistan_34.csv']

all_df = []

for i in paths[1:]:
    df = pd.read_csv(i) #pd.read_csv(csv_path) 
    df = df.drop_duplicates(subset=["Comments"], keep='last', inplace=False).dropna()
   
    df = df[df["Comments"].apply(detect).eq('en')]
  
    labels_noushin = df['Noushin'].values.squeeze()
    labels_khiem = df["Khiem"].values.squeeze()
    labels_banerjee = df['Banerjee'].values.squeeze()
    
    all_labels = np.vstack((labels_banerjee, labels_noushin, labels_khiem))
    all_labels_mode = stats.mode(all_labels)
    labels = all_labels_mode[0].squeeze().astype(int)
    df['Label'] = labels
    all_df.append(df)

labeled_df = pd.read_csv(paths[0])
labeled_df = labeled_df.drop_duplicates(subset=["Comments"], keep='last', inplace=False).dropna()
all_df.append(labeled_df)

new_df = pd.concat(all_df, ignore_index=True, axis=0)


# Calculate Fliess Kappa

import scipy

anno_1_labels = np.array(new_df["Khiem"].values, dtype=int).reshape(-1,1)
anno_2_labels = np.array(new_df["Noushin"].values, dtype=int).reshape(-1,1)
anno_3_labels = np.array(new_df["Banerjee"].dropna().values, dtype=int).reshape(-1,1)
anno_4_labels = np.array(new_df["Brett"].dropna().values, dtype=int).reshape(-1,1) # stack banerjee on brett


anno_3_labels = np.vstack((anno_3_labels, anno_4_labels))
data = np.hstack((anno_1_labels, anno_2_labels, anno_3_labels))

location, val = np.where(anno_1_labels != scipy.stats.mode(data, axis=1).mode)
for i in location:
    data[i][0] = int(not data[i][0])
new_df["Label"] = scipy.stats.mode(data, axis=1).mode

rater_mat = np.zeros((data.shape[0], 2), dtype=int)

for i in range(rater_mat.shape[0]): 
    comment_labels = data[i]
    zero_count = len(np.where(comment_labels==0)[0])
    one_count = len(np.where(comment_labels==1)[0])
    rater_mat[i][0] = zero_count
    rater_mat[i][1] = one_count

def fleiss_kappa(M):
  """
  See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
  :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
  :type M: numpy matrix
  """
  N, k = M.shape  # N is # of items, k is # of categories
  n_annotators = float(np.sum(M[0, :]))  # # of annotators
  print(n_annotators)

  p = np.sum(M, axis=0) / (N * n_annotators)
  P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
  Pbar = np.sum(P) / N
  PbarE = np.sum(p * p)

  kappa = (Pbar - PbarE) / (1 - PbarE)

  return kappa


#Total = 3+3+1 = 10
#Total = 2
import statsmodels.stats.inter_rater as rater

format_data, _ = rater.aggregate_raters(data)
print(format_data)
print(rater.fleiss_kappa(format_data))

new_df = new_df.drop('Noushin',1)
new_df = new_df.drop('Banerjee',1)
new_df = new_df.drop('Khiem',1)
new_df = new_df.drop('Brett',1)
new_df = new_df.drop('Likes',1)
new_df = new_df.dropna()

new_df.to_csv('annotations_1645.csv', index=False)
