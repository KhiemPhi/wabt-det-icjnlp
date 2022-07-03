import argparse
import os
import warnings
import numpy as np
from utils import preprocess_clean, load_comments, train_split_balance
from data import WhataboutismDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def train_test_split(comments, titles, labels, topics, ids):
    train_idx, test_idx  = train_split_balance(comments, titles, labels)
    
    train_comments = comments[train_idx]
    test_comments = comments[test_idx]
    
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    train_topics = topics[train_idx]
    test_topics = topics[test_idx]

    train_titles = titles[train_idx]
    test_titles = titles[test_idx]

    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    return train_comments, train_labels, train_topics, train_titles, train_ids, test_comments, test_labels, test_topics, test_titles, test_ids


def main(args):

    comments, labels, topics, titles, ids, _, _ = load_comments("dataset/annotations_1342.csv")    
    comments = np.array([ preprocess_clean(x) for x in comments])    
    train_comments, train_labels, train_topics, train_titles, train_ids, test_comments, test_labels, test_topics, test_titles, test_ids = train_test_split(comments, titles, labels, topics, ids)

    train_set = WhataboutismDataset(train_comments, train_labels, train_topics, train_titles, train_ids, args.context)
    test_set = WhataboutismDataset(test_comments, test_labels, test_topics, test_titles, test_ids, args.context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extract_feats", action='store_true',
                        default=False, help="Whether to extract features")
    parser.add_argument("-f", "--feats_path", type=str,
                        default='feats/comment_feats.npy', help="Path to the feat matrix")
    parser.add_argument("-t", "--titles_path", type=str,
                        default='feats/title_feats.npy', help="Path to the feat matrix")
    parser.add_argument("-g", "--gpu", type=str,
                        default='0', help="GPU to use")
    parser.add_argument("-b", "--batch", type=int,
                        default=5, help="batch size to use")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=1e-3, help="batch size to use")
    parser.add_argument("-ep", "--epochs", type=int,
                        default=20, help="batch size to use")
    parser.add_argument("-ts", "--get_transcripts", action='store_true',
                        default=False, help="Whether to get transcripts") 
    parser.add_argument("-tp", "--topics_num", type=int,
                        default=50, help="Whether to get transcripts") 
    parser.add_argument("-w", "--words_num", type=int,
                        default=5, help="Whether to get transcripts") 
    parser.add_argument("-mtl", "--multi_task", action='store_true',
                        default=False, help="Whether to do MTL w/ NextSeq Prediction")
    
    parser.add_argument("-ft", "--fine_tune", action='store_true',
                        default=False, help="Whether to fine-tune pre-trained models")
    
    parser.add_argument("-c", "--context", action='store_true',
                        default=False, help="Whether to fine-tune pre-trained models")
 
 
    args = parser.parse_args()
    main(args)