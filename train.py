import argparse
from builtins import breakpoint
import os
import warnings

import numpy as np
from modeling.model import ContextSentenceTransformerMultiTask
import optuna
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data import WhataboutismDataset, WhataboutismDatasetUnlabeled
from modeling import ContextSentenceTransformer, SentenceTransformer, SelfSupervisedContextSentenceTransformer, ProtoTransformer
from utils import load_comments, add_augmentation, train_test_split_helper
from utils.utils import train_split_balance



os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")



def load_data(args, file_path="./dataset/annotations_986.csv", aug_path="./dataset/augment.csv", unlabel_testing=False):
    comments, labels, topics, titles, ids, _, sent_to_related, all_transcript_sents, df  = load_comments(file_path)  # load dataset w/ transcript
   
    queries = np.unique(df.index)
    non_wabt_count = []
    wabt_count = []
    for query in queries:        
        non_wabt, wabt = np.bincount(df.loc[query]["Label"])
        non_wabt_count.append(non_wabt)
        wabt_count.append(wabt)
    
    df_plot = pd.DataFrame({'non-wabt': non_wabt_count,

                   'wabt': wabt_count}, index=queries)
    df_plot.to_csv("dataset_summary.csv", index=True)

    queries = np.unique(df.index)
    title_counts = []
    for query in queries:        
        title_count = len(np.unique(df.loc[query]["Title"]))
        title_counts.append(title_count)
    
    df_plot = pd.DataFrame({'no. of videos': title_counts
                   }, index=queries)
    df_plot.to_csv("dataset_summary_by_title.csv", index=True)
    
   
    if unlabel_testing:
        extra_comments, extra_labels, extra_topics, extra_titles, extra_ids, _, _, _, new_df = load_comments("./dataset/annotations_1500.csv") # load dataset w/o transcripts
        diff_comments = np.setdiff1d(extra_comments, comments)
        idx_of_diff_comments = [np.where(extra_comments==i)[0][0] for i in diff_comments]    
        unlabeled_topics = extra_topics[idx_of_diff_comments]
        unlabeled_titles = extra_titles[idx_of_diff_comments]
        unlabeled_ids = extra_ids[idx_of_diff_comments]
        unlabeled_test_comments = extra_comments[idx_of_diff_comments]
        unlabeled_labels = extra_labels[idx_of_diff_comments]
        unlabel_test_set = WhataboutismDataset(unlabeled_test_comments, unlabeled_labels, unlabeled_topics, unlabeled_titles, unlabeled_ids, False,  new_df, False)

              
    train_comments, train_labels, train_topics, train_titles, train_ids, test_comments, test_labels, test_topics, test_titles, test_ids, train_idx_all, test_idx_all = train_test_split_helper(comments, titles, labels, topics, ids)
    aug_to_idx_train = {}
    aug_to_idx_test = {}

    # Add Augmentation
    if args.train_augment:
        train_comments, train_labels, train_topics, train_titles, train_ids, aug_to_idx_train = add_augmentation(train_comments, train_labels, train_topics, train_titles, train_ids, aug_path="./dataset/augment.csv", dataframe=df)
       
    if args.test_augment:
        test_comments, test_labels, test_topics, test_titles, test_ids, aug_to_idx_test = add_augmentation(test_comments, test_labels, test_topics, test_titles, test_ids, aug_path="./dataset/augment.csv",  dataframe=df)

    # Divide by 2 to get train_val
    test_idx, val_idx = train_split_balance(test_comments, test_topics, test_labels, percentage=0.5)
    train_set = WhataboutismDataset(train_comments, train_labels, train_topics, train_titles, train_ids, args.context,  df, False, train_idx_all, test_comments, aug_to_idx_train, args.random, args.agnostic, args.title)
    val_set =  WhataboutismDataset(test_comments[val_idx], test_labels[val_idx], test_topics[val_idx], test_titles[val_idx], test_ids[val_idx], args.context,  df, True, val_idx, test_comments,aug_to_idx_test, args.random, args.agnostic, args.title)
    test_set = WhataboutismDataset(test_comments[test_idx], test_labels[test_idx], test_topics[test_idx], test_titles[test_idx], test_ids[test_idx], args.context,  df, True, test_idx, test_comments,aug_to_idx_test, args.random, args.agnostic, args.title)   
    
    unlabel_set = WhataboutismDatasetUnlabeled(comments=all_transcript_sents, comments_to_related=sent_to_related)

    # Let's generate a dataset summary 
    

    if unlabel_testing:
        return train_set, test_set, unlabel_set, unlabel_test_set
    else: 
        return train_set, test_set, unlabel_set, val_set
    
def objective(trial: optuna.trial.Trial):

    gamma = trial.suggest_float("gamma", 0.5, 3.5) #best gamma is 3.38, best beta is 0.999``      
    beta = trial.suggest_float("beta", 0.9, 0.999) #best gamma is 3.38, best beta is 0.999``     
    
    train_set, test_set, unlabel_set, val_set = load_data(args)

      
    checkpoint_callback = ModelCheckpoint(
        monitor="validation-f1",
        dirpath="best_ckpts",
        filename="wabt-det-{epoch:02d}-{validation-f1:.2f}" +  "-" + args.study_name,
        save_top_k=1,
        mode="max",
    )
    if args.context:
        if args.mtl:
            model = ContextSentenceTransformerMultiTask(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.99, gamma=1.85,class_num=2, context=args.context, loss=args.loss, cross=False, unlabel_set=unlabel_set)
        elif args.pro: 
            model = ProtoTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.99, gamma=1.5,class_num=2, context=args.context, loss=args.loss, cross=False, unlabel_set=unlabel_set)
        else: 
            model = ContextSentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.9    , gamma=3.38,class_num=2, context=args.context, loss=args.loss, cross=False, unlabel_set=unlabel_set)
    
    
    else: 
        model = SentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.9, gamma=1.85, class_num=2, context=args.context, loss=args.loss)
    
    trainer = Trainer(devices=[int(args.gpu)] if torch.cuda.is_available() else 0,  accelerator="gpu",
                      max_epochs=args.epochs, auto_select_gpus=True, benchmark=True,        
                      auto_lr_find=True, check_val_every_n_epoch=1, num_sanity_val_steps=0, callbacks=[checkpoint_callback], logger=True)
   
    #hyperparameters = dict(gamma=gamma, beta=beta)
    #trainer.logger.log_hyperparams(hyperparameters)
    #trainer.tune(model)
    trainer.fit(model)   
    return trainer.callback_metrics["best-f1"].item()


def main(args):
    if not args.testing:

        source = open('{}.txt'.format(args.study_name), 'w')
      
        if args.loss == "focal" or args.loss == "softmax":   
            pruner: optuna.pruners.BasePruner = (
                optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
            )
            study = optuna.create_study(direction="maximize", pruner=pruner, study_name=args.study_name)
            study.optimize( objective, n_trials=1, gc_after_trial=True)
            
            print("Number of finished trials: {}".format(len(study.trials)), file=source)

            print("Best trial:", file=source)
            trial = study.best_trial

            print("  Validation-F1: {}".format(trial.value), file=source)

            print("  Params Optimized: ", file=source)
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value), file=source)
            
            f1_results = []
            for i in study.trials:
                f1_results.append(i.value)
            f1_results = np.array(f1_results)

            print(np.mean(f1_results))
            print(np.std(f1_results)) 

            # Read vis/validation_acc_tab
            with open('vis/validation_acc_tab.txt') as meta_acc_file, open('{}.txt'.format(args.study_name)) as res_file:
                for line in meta_acc_file:
                    print(line, file=source)
                   
                    
             
        else:
            train_set, test_set, unlabel_set, val_set = load_data(args)
            if args.context:
                model = ContextSentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss, cross=False)
            else: 
                model = SentenceTransformer(train_set, test_set, val_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss)
            
            checkpoint_callback = ModelCheckpoint(
                monitor="validation-f1",
                dirpath="best_ckpts",
                filename="wabt-det-{epoch:02d}-{validation-f1:.2f}" +  "-" + args.study_name,
                save_top_k=3,
                mode="max",
            )
            trainer = Trainer(devices=[int(args.gpu)] if torch.cuda.is_available() else 0,  accelerator="gpu",
                            max_epochs=args.epochs, auto_select_gpus=True, benchmark=True,        
                            auto_lr_find=True, check_val_every_n_epoch=1, num_sanity_val_steps=0, callbacks=[checkpoint_callback])
            trainer.fit(model) 

            print("  Validation-F1: {}".format(model.best_f1), file=source)
            with open('vis/validation_acc_tab.txt') as meta_acc_file, open('{}.txt'.format(args.study_name)) as res_file:
                for line in meta_acc_file:
                    print(line, file=source)

    else: 
        train_set, test_set, unlabel_set, unlabel_test_set = load_data(args, unlabel_testing=True)        
        model = ContextSentenceTransformer(train_set, test_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.99, gamma=3.38, class_num=2, context=args.context, loss=args.loss, cross=False, unlabel_set=unlabel_set)

        #1. Complete Initial Training 
        trained_model = model.load_from_checkpoint(train_set=train_set, test_set=test_set, checkpoint_path="best_ckpts/wabt-det-epoch=39-validation-f1=93.55-context no augments.ckpt")           
        trainer = Trainer(devices=[int(args.gpu)], accelerator="gpu", logger=False, auto_select_gpus=False, num_sanity_val_steps=0, max_epochs=args.epochs)

        #3. Self-Supervised Training: We take the cosine similarity of the comment w/ a set of random comments in the train-labeled set, find the top-5. Avg. Scores of Their Prediction and then 
        # regress on MSE Loss if the results improved then we are good, asssume the set-diff is a held-out unlabeled set
        self_supervised_model = SelfSupervisedContextSentenceTransformer(trained_model.sent_transformer, trained_model.tokenizer, train_set, test_set, unlabel_test_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0.99, gamma=3.38, class_num=2, context=args.context, loss=args.loss, cross=False, unlabel_set=unlabel_set)
        
        #4. Now in the self-supervised-model, we have 2 goals: improve performance on test_set + good performance on unlabel test set + good visual performance on unlabel set
        trainer.fit(self_supervised_model)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpu", type=str,
                        default='0', help="GPU to use")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=35, help="batch size to use")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=1e-4, help="batch size to use")
    parser.add_argument("-ep", "--epochs", type=int,
                        default=20, help="batch size to use")
    parser.add_argument("-c", "--context", action='store_true',
                        default=False, help="Whether to fine-tune pre-trained models")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )

    parser.add_argument("-ta", "--test_augment",  action="store_true", help="whether to use augmentation in testing")
    parser.add_argument("-tra", "--train_augment", action="store_true", help="whether to use augmentation in training")
    parser.add_argument("-l", "--loss", type=str,  default="focal", help="type of loss to use")
    parser.add_argument("-sn", "--study_name", type=str, default="context-aug", help="name of test ran")
    parser.add_argument("-t", "--testing", action='store_true',
                        default=False, help="Whether to test using pre-trained models")
    parser.add_argument("-r", "--random", action='store_true',
                        default=False, help="Whether to test using pre-trained models")
    parser.add_argument("-a", "--agnostic", action='store_true',
                        default=False, help="Whether to test using pre-trained models")
    parser.add_argument("-ti", "--title", action='store_true',
                        default=False, help="Use title as context")
    parser.add_argument("-mtl", "--mtl", action='store_true',
                        default=False, help="multi-task learning")
    parser.add_argument("-pr", "--pro", action='store_true',
                        default=False, help="prototype learning")
 
 
    args = parser.parse_args()
    main(args)
