import argparse
import os
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data import WhataboutismDataset
from modeling import ContextSentenceTransformer, SentenceTransformer
from utils import load_comments, add_augmentation, train_test_split_helper

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def load_data(args, file_path="../dataset/annotations_1342.csv", aug_path="../dataset/augment.csv"):
    comments, labels, topics, titles, ids, _, _, df = load_comments(file_path)  
    
    train_comments, train_labels, train_topics, train_titles, train_ids, test_comments, test_labels, test_topics, test_titles, test_ids = train_test_split_helper(comments, titles, labels, topics, ids)
    
    # Add Augmentation
    if args.train_augment:
        train_comments, train_labels, train_topics, train_titles, train_ids = add_augmentation(train_comments, train_labels, train_topics, train_titles, train_ids, aug_path="../dataset/augment.csv")

    if args.test_augment:
        test_comments, test_labels, test_topics, test_titles, test_ids = add_augmentation(test_comments, test_labels, test_topics, test_titles, test_ids, aug_path="../dataset/augment.csv")

    train_set = WhataboutismDataset(train_comments, train_labels, train_topics, train_titles, train_ids, False,  df, False)
    test_set = WhataboutismDataset(test_comments, test_labels, test_topics, test_titles, test_ids, False,  df, True)

    return train_set, test_set
    
def objective(trial: optuna.trial.Trial):

    gamma = trial.suggest_float("gamma", 2.5, 3.5) #best gamma is 3.38, best beta is 0.999``      
    beta = trial.suggest_float("beta", 0.99, 0.999) #best gamma is 3.38, best beta is 0.999``     
    
    train_set, test_set = load_data(args)
   
    checkpoint_callback = ModelCheckpoint(
        monitor="validation-f1",
        dirpath="best_ckpts",
        filename="wabt-det-{epoch:02d}-{validation-f1:.2f}" +  "-" + args.study_name,
        save_top_k=3,
        mode="max",
    )
    if args.context:
        model = ContextSentenceTransformer(train_set, test_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=beta, gamma=gamma, class_num=2, context=args.context, loss=args.loss, cross=False)
    else: 
        model = SentenceTransformer(train_set, test_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=beta, gamma=gamma, class_num=2, context=args.context, loss=args.loss)
    
    trainer = Trainer(gpus=args.gpu if torch.cuda.is_available() else 0, 
                      max_epochs=20, auto_select_gpus=True, benchmark=True,        
                      auto_lr_find=True, check_val_every_n_epoch=1, num_sanity_val_steps=0, callbacks=[checkpoint_callback])
   
    hyperparameters = dict(gamma=gamma, beta=beta)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model)          

    return trainer.callback_metrics["best-f1"].item()


def main(args):
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
    else:
        train_set, test_set = load_data(args)
        if args.context:
            model = ContextSentenceTransformer(train_set, test_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss, cross=False)
        else: 
            model = SentenceTransformer(train_set, test_set, learning_rate=args.learning_rate, batch_size=args.batch_size, beta=0, gamma=0, class_num=2, context=args.context, loss=args.loss)
        
        checkpoint_callback = ModelCheckpoint(
            monitor="validation-f1",
            dirpath="best_ckpts",
            filename="wabt-det-{epoch:02d}-{validation-f1:.2f}" +  "-" + args.study_name,
            save_top_k=3,
            mode="max",
        )
        trainer = Trainer(gpus=args.gpu if torch.cuda.is_available() else 0, 
                        max_epochs=20, auto_select_gpus=True, benchmark=True,        
                        auto_lr_find=True, check_val_every_n_epoch=1, num_sanity_val_steps=0, callbacks=[checkpoint_callback])
        trainer.fit(model)  


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

 
    args = parser.parse_args()
    main(args)
