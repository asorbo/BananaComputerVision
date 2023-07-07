# Banana state estimation using Convolutional Neural Networks for waste reduction
Bachelor Thesis of AI @Vrije Universiteit Amsterdam. <br>Written by Agostino Sorbo with the supervision of Bob Borsboom.


## Description - Abstract

This paper explores the application of Convolutional Neural Networks to the task of perishable food state classification.
Fresh fruit and vegetables contribute to almost half of household food waste. Furthermore, perishable food spoilage and quality directly impact the processed perishable food industry.
Currently, the state of fruits and vegetables is mostly done by visual assessment, both domestically and in industrial settings. Introducing an automated system able to classify perishable foods according to their state may help reduce food waste. This can be done domestically by integrating such solutions in smart home appliances such as fridges and pantries and in industrial settings by enabling automated, highly accurate quality control.
Current research mainly focuses on the binary classification of perishable foods as either fresh or rotten. However, to allow perishable foods to be consumed or processed before they reach the rotting stage and avoid food waste, it may be helpful to move beyond this binary classification. 
This thesis aims to show how Convolutional Neural Networks can be used to perform an external analysis of perishable foods to estimate their state.
In particular, this paper illustrates how to train multi-class, single-label classifiers for perishable food state classification.
To do so a dataset is adapted for the task. Following, two approaches are experimented with: developing a custom, shallow CNN architecture, randomly initialized, and training a Deep Residual CNN leveraging transfer learning.
For both architectures, several models were trained to perform a grid search and select the best hyperparameters. As suggested by previous research, models trained with the ResNet50 architecture were successful and especially well-performing in the task of classifying bananas as either: unripe, ripe, overripe, or rotten.
The procedure shown in this paper may be replicated and applied to other perishable foods to further limit perishable food waste both for domestic and industrial applications.


## Table of contents

* BananaStateClassification.py: The main file including the code to train banana state classification models.
* predicter.py: A file that can be used to load trained models and make predictions.
* toONNX.py: Script that can be used to convert trained PyTorch models into ONNX models that can be easily integrated into other projects.
* Thesis presentation.pdf: a brief presentation of the study.
* Thesis presentation.pdf: the thesis paper extensively describing the research process and the development of this project.


## How to train models

In the BananaStateClassification.py set the constants to the desired values and run the file to perform a grid search.

To be able to run this unzip the dataset, place it in the project directory, and make sure that IMAGE_DATASET_LABELS_PATH and IMAGE_DATASET_LABELS are set accordingly
It is possible to swap the dataset maintaining the structure of the labels file (classesTotal.csv)
The parameters under "#Gridsearch and training parameters" can be set to perform a grid search, alternatively, one single value per parameter can be set to train a single model (e.g. ARCHITECTURES = ["Res50"] BATCH_SIZES = [10] LEARNING_RATES = [0.0005].

The model training will now begin. The trained models, the training plots, and relative data will be saved in the directory specified in the "BASE_PATH" constant (make sure the directory is valid). By default, this is the "outputs" folder in the project directory. 
