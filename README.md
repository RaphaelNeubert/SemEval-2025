# Emotion Detection in Text using Transformer-Based Models

## Overview

This repository contains code for multi-label emotion classification using transformer-based models, specifically focusing on the SemEval 2025 Task 11-A benchmark. 
The project explores different pretraining and fine-tuning strategies to enhance emotion detection from short text snippets.

## Methods

We investigate three transformer-based strategies for emotion classification:
* **Method 1:** Custom Pretraining on Sentiment Data: Pretraining a transformer model on emotionally charged Twitter data before fine-tuning on the SemEval dataset for emotion detection.
* **Method 2:** Direct Fine-Tuning of RoBERTa: Using a standard RoBERTa model and fine-tuning it directly on the SemEval dataset.
* **Method 3:** Intermediate Pretraining on GoEmotions: Pretraining the model on the GoEmotions dataset before fine-tuning on the SemEval dataset.


## Repository Structure

Each method has its own implementation branch:

* **Method 1:**  Branch: `sentiment140-v2`
* **Method 2:** Branch: `pretrained_bert`
* **Method 3:** 
  - Pretraining on GoEmotions -> Branch: `bert_goemotion`
  - Final Fine-Tuning on SemEval ->  Branch: `pretrained_bert`

## Datasets

The datasets used in this project are:

- [Sentiment140](https://huggingface.co/datasets/stanfordnlp/sentiment140) - Twitter-based sentiment dataset used for custom pretraining.
- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) - A dataset containing fine-grained emotion labels.
- [SemEval 2025 Task 11-A](https://www.codabench.org/competitions/3863/#/pages-tab) - The offical SemEval dataset used for fine-tuning (requires login).

## Requirements

To set up the environment, install the required Python packages using:
* `torch`, `numpy`, `h5py`
* `transformers` -> For the RoBERTa tokenizer and pretrained model (Method 2)
* `tqdm` -> Only required when using the -t flag to have a statusbar 
* `pandas` -> Only required for the datset preprocessing jupyter notebook and the submission / submission evaluation script
* `scikit-learn` -> Only required for the submission evaluation script
```sh
pip install torch numpy pandas scikit-learn transformers tqdm h5py
