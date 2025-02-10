# Emotion Detection in Text using Transformer-Based Models

## Overview

This repository contains code for multi-label emotion classification using transformer-based models, specifically focusing on the SemEval 2025 Task 11-A benchmark. 
The project explores different pretraining and fine-tuning strategies to enhance emotion detection from short text snippets.

## Methods Evaluated

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
