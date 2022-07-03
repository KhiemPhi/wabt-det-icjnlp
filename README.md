[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)

# WABT-DET-CONTEXT
This is the official source code for **Whataboutism Detection Using Context From Topical Discourse**. 


## Instalation:

To run the codebase, you can install a Conda environment with the following command:

```shell
conda env create -f environment.yml
```

## Code Structure
├── data
│   ├── dataset.py
│   ├── __init__.py
├── dataset
│   ├── annotations_1342.csv
│   ├── augment.csv
│   ├── collect.py
│   └── transcript_whataboutism_985.csv
├── environment.yml
├── experiment.sh
├── LICENSE
├── loss_fn
│   ├── __init__.py
│   ├── loss.py
├── modeling
│   ├── __init__.py
│   ├── model.py
├── README.md
├── train.py
├── utils
│   ├── __init__.py
│   └── utils.py


## Datasets

Located in the dataset folder are the .csv files containg the YouTube comments and their Whataboutism Annotations.

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