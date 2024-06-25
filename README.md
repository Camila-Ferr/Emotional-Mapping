# Music Emotion Recognition (MER) Project

This repository contains the implementation of a Music Emotion Recognition (MER) system using three different machine learning models: Random Forest, Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN). The goal of this project is to predict emotions in instrumental music tracks.

## Repository Structure

- **notebooks/**: This folder contains Jupyter notebooks with the code used to train and evaluate the models.
  - `RF_MLP.ipynb`: Notebook for training and evaluating Random Forest and Multi-Layer Perceptron models.
  - `CNN.ipynb`: Notebook for training and evaluating the Convolutional Neural Network model.
- **csv/**: This folder contains the csv's with the labels of each music used on each dataset.

## Project Overview

### 1. Random Forest and Multi-Layer Perceptron (RF_MLP.ipynb)

This notebook includes:
- **Libraries used**: Importing necessary libraries.
- **Datasets**: Loading and preprocessing the datasets.
- **Extracting Data to DataFrame**: Converting the raw data into a structured DataFrame.
- **Showing the Dataset**: Visualizing the dataset.
- **Checking the Input Values**: Analyzing the input features.
- **Training Random Forest Model**: Implementing the Random Forest algorithm.
- **Training Multi-Layer Perceptron Model**: Implementing the MLP algorithm.
- **Model Evaluation**: Evaluating the performance of the trained models.

### 2. Convolutional Neural Network (CNN.ipynb)

This notebook includes:
- **Libraries**: Importing necessary libraries.
- **Drive**: Setting up the environment for accessing datasets.
- **Extracting Dataframe**: Converting the raw data into a structured DataFrame.
- **Model**: Building and training the CNN model.
- **Accuracy History**: Tracking and visualizing the model's accuracy over training epochs.
- **Evaluation**: Assessing the performance of the CNN model.

## Conclusion
This project demonstrates the application of different machine learning models to the task of emotion recognition in music. By comparing the performance of Random Forest, Multi-Layer Perceptron, and Convolutional Neural Network models, we aim to identify the most effective approach for this task.
