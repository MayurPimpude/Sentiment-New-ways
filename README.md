# Sentiment Analysis Model Training using Twitter Sentiment Dataset

Welcome to the Sentiment Analysis Model Training project! This repository contains code and instructions for training a sentiment analysis model using the Twitter Sentiment dataset. We explore two approaches: a custom deep learning model achieving an accuracy of 0.7926 and a pre-trained model from Hugging Face, specifically `distilbert-base-uncased-finetuned-sst-2-english`.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Models](#models)
  - [Deep Learning Model](#deep-learning-model)
  - [Hugging Face DistilBERT Model](#hugging-face-distilbert-model)
- [Requirements](#requirements)
- [Installation](#installation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to train and evaluate sentiment analysis models on the Twitter Sentiment dataset. Sentiment analysis involves classifying text into positive, negative, or neutral sentiments. We implement and compare two approaches: a custom deep learning model and a pre-trained Hugging Face DistilBERT model.

## Datasets

The Twitter Sentiment dataset contains tweets labeled with their sentiment. This dataset is used to train and evaluate our sentiment analysis models.

## Models

### Deep Learning Model

Our custom deep learning model is a neural network trained from scratch on the Twitter Sentiment dataset. This model achieves an accuracy of 0.7926.

### Hugging Face DistilBERT Model

We also use `distilbert-base-uncased-finetuned-sst-2-english`, a pre-trained model from Hugging Face's Transformers library. This model is fine-tuned on the SST-2 dataset and provides a strong baseline for sentiment analysis.

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers (Hugging Face)
- Datasets
- NumPy
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
   ```sh
   git clone [https://github.com/your-username/sentiment-analysis.git](https://github.com/MayurPimpude/Sentiment-New-ways)
   cd sentiment-analysis

## Results
The results of the training process, including model performance metrics and example outputs, in resepective ipynb files.

## Contributing
We welcome contributions to improve this project! Please submit pull requests or open issues to discuss potential enhancements or bug fixes.
