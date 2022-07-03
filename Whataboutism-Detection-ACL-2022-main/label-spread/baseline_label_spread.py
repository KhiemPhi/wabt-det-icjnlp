import string
import nltk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import copy
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import csv
import json
nltk.download('punkt')


def load_comments(csv_path):
    df = pd.read_csv(csv_path)
    all_comments = df[['Comments']].values.squeeze()
    all_labels = df[['Label']].values.squeeze()
    positive_comments = df.loc[df['Label'] == 1][['Comments']].values.squeeze()
    negative_comments = df.loc[df['Label'] == 0][['Comments']].values.squeeze()
    return all_comments, all_labels, positive_comments, negative_comments


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


def scale_scores(scores):
    scores = np.asarray(scores)
    standardized_scores = (scores - scores.mean()) / scores.std()  # Scale to standard Gaussian
    sigmoid_scores = 1 / (1 + np.exp(-standardized_scores))  # Apply sigmoid to convert to probabilities
    norm_scores = (sigmoid_scores - min(sigmoid_scores)) / (
                max(sigmoid_scores) - min(sigmoid_scores))  # Minimum prob appears to be 0.5 with sigmoid --> [0, 1]
    return norm_scores


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


def train_label_prop(similarity_matrix, labels, comments, num_labels=9, folds=1):
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

    for i in range(folds):
        supervised_positive_labels = np.random.choice(all_positive_labels[0], num_labels, replace=False)
        supervised_negative_labels = np.random.choice(all_negative_labels[0], num_labels, replace=False)
        label_index = np.hstack((supervised_positive_labels, supervised_negative_labels))

        unlabeled_indices = np.setdiff1d(all_index, label_index)
        new_labels = np.copy(labels)
        new_labels[unlabeled_indices] = -1

        lp_model = LabelSpreading(kernel='rbf', gamma=20, max_iter=100, alpha=0.9)

        lp_model.fit(similarity_matrix, new_labels)
        y_pred = lp_model.predict(similarity_matrix)
        y_proba = lp_model.predict_proba(similarity_matrix)[:, 1]

        test_labels = np.delete(default_labels, label_index)
        y_pred_unlabeled = np.delete(y_pred, label_index)

        acc = accuracy_score(y_pred_unlabeled, test_labels) * 100
        f1 = f1_score(y_pred_unlabeled, test_labels)
        ap = average_precision_score(y_pred_unlabeled, test_labels)

        if acc > best_acc:
            best_acc = acc
            best_labels = new_labels
            best_fold = i + 1
            worst_acc = acc
            worst_comments = comments[label_index]
            worst_labels = default_labels[label_index]
            best_comments = comments[label_index]
        if worst_acc > acc:
            worst_acc = acc
            worst_comments = comments[label_index]
            worst_labels = default_labels[label_index]

    return y_pred, y_proba, best_comments, best_acc, f1, worst_comments, worst_acc, best_labels, label_index


def compute_baseline(similarity_file, all_comments, all_labels, initial_labels=0.1):
    # 1. Loading feature matrix + Labels
    similarity_matrix = pd.read_csv(similarity_file, header=None)
    total_labels = int(initial_labels * len(all_comments) / 2)
    # num_comments = len(positive_comments) + len(negative_comments)
    pred_labels, pred_proba, best_comments, best_acc, f1, worst_comments, worst_acc, _, label_index = train_label_prop(
        similarity_matrix, all_labels,
        all_comments, num_labels=total_labels, folds=1)

    return best_acc, f1, pred_proba, label_index, pred_labels


def main(args):
    data = "annotations_200.csv"
    all_comments, all_labels, positive_comments, negative_comments = load_comments(data)


    avg_acc = 0
    avg_f1 = 0
    # avg_labels = np.zeros(len(all_labels))

    K = 150  # consider experiments with 500, 1000, 2000, etc.

    for i in range(K):
        acc, f1, _, _, pred_labels = compute_baseline("annotations_200_sim.csv")
        avg_acc += acc
        avg_f1 += f1
        # avg_labels += pred_labels
    avg_acc /= K
    avg_f1 /= K
    # avg_labels /= K

    print("Average accuracy for LS is " + str(avg_acc))
    print("Average f1 score for LS is " + str(avg_f1))
    # print(avg_labels)

    tfidf_mapping = get_tfidf_mapping('reddit_impeachment_scores.csv',
                                    ignore_words=[])
    tfidf_scores = get_tfidf_scores(all_comments, tfidf_mapping, norm=True)
    tfidf_probs = scale_scores(tfidf_scores)

    avg_acc = 0
    avg_f1 = 0
    # avg_labels = np.zeros(len(all_labels))

    for i in range(K):
        _, _, pred_proba, label_index, pred_labels = compute_baseline("annotations_200_sim.csv")

        pred_proba_test = np.delete(pred_proba, label_index)
        tfidf_test_probs = np.delete(tfidf_probs, label_index)
        test_labels = np.delete(all_labels, label_index)

        _, acc, f1 = combine_metrics(pred_proba_test, tfidf_test_probs, test_labels, alpha=0.6)
        avg_acc += acc
        avg_f1 += f1
        # avg_labels += pred_labels

    avg_acc /= K
    avg_f1 /= K
    # avg_labels /= K

    # Why is accuracy better but F1 worse?
    # More comments without target words -> weighted less -> fewer whataboutisms
    print("Average accuracy  for LS+TI is " + str(avg_acc))
    print("Average f1 score for LS+TI is " + str(avg_f1))


