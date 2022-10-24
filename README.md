[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)

# WABT-DET-CONTEXT
This is the official source code for **Whataboutism Detection Using Context From Topical Discourse**.  Our methodology are built upon two observations:

(1) Whataboutism are alike each other pragmatically across topics. 
(2) To model these pragmatics, if we create a combine embedding of whataboutism + non-whataboutism embedding, we will be able to classifiy each classes with a simple MLP head
(3) However, this does not generalize because we do not have access of labels during test time. 
(4) We can try to use Cosine-Similarity or Euclidean Distance to simulate embeddings having a different label but these metrics break down in high dimiesions such as BERT-embeddings
(5) Therefore, we train a seperate transformer encoder that can learn a seperate distance-attention function which is more robust than Cosine or Euclidean distance.


## Instalation:

To run the codebase, you can install a Conda environment with the following command:

```shell
conda env create -f environment.yml
```

## Code Structure
```bash
├── add_pair_column.py
├── build_data.py
├── data
│   ├── dataset.py
│   └── __init__.py
├── dataset
│   ├── annotations_1342.csv
│   ├── annotations_1500.csv
│   ├── annotations_1500_sim_idx.csv
│   ├── annotations_1615.csv
│   ├── annotations_1645.csv
│   ├── annotations_1645_sim_idx.csv
│   ├── annotations_986.csv
│   ├── augment.csv
│   └── collect.py
├── dataset_split.png
├── dataset_summary_by_title.csv
├── dataset_summary.csv
├── dataset_summary.jpg
├── environment.yml
├── experiment.sh
├── LICENSE
├── loss_fn
│   ├── __init__.py
│   └── loss.py
├── modeling
│   ├── __init__.py
│   ├── model.py
│   ├── proto.py
│   └── self_sup.py
├── README.md
├── results
│   ├── dataset_summary_by_title.csv
│   ├── dataset_summary.csv
│   └── test.jpg
├── test.jpg
├── test.py
├── train.py
├── utils
│   ├── __init__.py
│   └── utils.py
├── vis
│   ├── label_dataset_summary.jpg
│   ├── test_label_dataset_summary.jpg
│   ├── topic_dataset_summary.jpg
│   ├── train_label_dataset_summary.jpg
│   ├── tSNE
│   │   ├── test-tSNE-epoch-0.jpg
│   │   ├── test-tSNE-epoch-10.jpg
│   │   ├── test-tSNE-epoch-11.jpg
│   │   ├── test-tSNE-epoch-12.jpg
│   │   ├── test-tSNE-epoch-13.jpg
│   │   ├── test-tSNE-epoch-14.jpg
│   │   ├── test-tSNE-epoch-15.jpg
│   │   ├── test-tSNE-epoch-1.jpg
│   │   ├── test-tSNE-epoch-20.jpg
│   │   ├── test-tSNE-epoch-26.jpg
│   │   ├── test-tSNE-epoch-2.jpg
│   │   ├── test-tSNE-epoch-3.jpg
│   │   ├── test-tSNE-epoch-4.jpg
│   │   ├── test-tSNE-epoch-5.jpg
│   │   ├── test-tSNE-epoch-6.jpg
│   │   ├── test-tSNE-epoch-7.jpg
│   │   ├── test-tSNE-epoch-8.jpg
│   │   └── test-tSNE-epoch-9.jpg
│   ├── validation_acc_tab.txt
│   ├── validation_results_1342.csv
│   ├── validation_results_1615.csv
│   └── validation_results_unlabeled.csv
└── Whataboutism-Detection-ACL-2022-main
    ├── dataset
    │   ├── annotations_1342.csv
    │   ├── CNN_Impeachment trial of President Donald Trump.json
    │   ├── collect.py
    │   ├── Fox News_Trump Impeachment Inquiry.json
    │   ├── george_floyd.json
    │   ├── transcript_whataboutism_985.csv
    │   ├── trump_biden.json
    │   ├── trump_impeachment.json
    │   ├── unite_the_right_rally_charlottesville.json
    │   └── us_soldier_russian_bounty.json
    ├── label-spread
    │   ├── annotations_1500.csv
    │   ├── annotations_200.csv
    │   ├── annotations_200_sim.csv
    │   ├── baseline_label_spread.py
    │   ├── best_seed_list.csv
    │   ├── best_seed_list_tf_idf.csv
    │   ├── correct_and_smooth.py
    │   ├── dataset.py
    │   ├── failure_cases.csv
    │   ├── failure_cases_tf_idf.csv
    │   ├── features_raw.npy
    │   └── reddit_impeachment_scores.csv
    ├── README.md
    └── sent-trans-model
        ├── data
        │   ├── dataset.py
        │   └── __init__.py
        ├── modeling
        │   └── model.py
        ├── test.py
        ├── train.py
        └── utils
            ├── __init__.py

```

## Datasets

Located in the dataset folder are the .csv files containg the YouTube comments and their Whataboutism Annotations. We have various versions of the dataset including 
versions of 986, 1500 and 1642 comments, all hand-annotated and collected. 

You can collect more data for further testing by using the following command:

```shell
python -u dataset/collect.py --api_key [your Youtube API key] --topic [the topic of videos you want collected] --save_as [file path to save your .csv results]
```

More instructions on how to set up a YouTube API-V3 API key can be found in this following tutorial https://www.youtube.com/watch?v=th5_9woFJmk


## Training and Evaluation

For quick evaluation, you can run the following command: 

```shell
bash experiment.sh
```

This will allow you to view all test results in different text-files for the various F1-scores. The train.py script both trains the model and evaulate the model on the test set each epoch. It automaticall registers the epoch with the best F1 results with the help of the PyTorch-Lightning wrappers.  