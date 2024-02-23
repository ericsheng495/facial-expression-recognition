---
layout: default
---

## Introduction / Motivation

Smart Home Appliances currently lack the ability to recognize and respond to users' emotional states, leading to a one-size-fits-all approach that can be unresponsive to individual needs. By integrating facial emotion recognition into smart homes elements, smart systems can adjust lighting, music, and temperature in real-time to support the recognized emotion state of the occupants, creating a more satisfying home experience [1].

In this proposal, we discuss how facial expression image datasets can undergo various machine learning methods to classify human emotions for Smart Homes Appliances.

## Dataset

We will utilize the FER2013 dataset for Facial Expression Recognition training. The FER2013 dataset is a collection of 48x48 pixel grayscale images, containing 28,000 labeled images in the training set. Each image is labeled as one of seven emotions (happy, sad, angry, afraid, surprise, disgust, neutral).

![FER2013 Dataset Sample](https://miro.medium.com/v2/resize:fit:720/format:webp/1*BVp2NO-EYaiF1GDIpBs37Q.png)

[Dataset Link: FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)

## Data Preprocessing

We plan to use **image normalization**, Python's **Scikit-image** & Scikit-learn packages, histogram of oriented gradients (HOG), and **linear averaging filter** for image preprocessing.

- Image normalization standardizes pixel values, improving the model’s stability and performance.
- Scikit-image uses image processing capabilities; Scikit-learn has data manipulation and implementation of algorithms.
- HOG can capture edges and textures and distinguish facial expressions.
- Linear averaging filters can reduce image noise and improve the accuracy of feature extraction in human expression recognition.

## Methods

We plan to use **CNN (feature extraction)**, **Neural network (classification)**, and **Support Vector Machine (SVM)**, for facial expression machine learning.

From _Deep Learning Approaches for Facial Emotion Recognition: A Case Study on FER-2013_, CNN can be used for facial expression recognition as a supervised machine learning method [2]. We will perform prediction by forward propagation, calculate the loss function using the labels given in the dataset and perform weight update by backward propagation.

We use neural networks to forecast and classification as _Artificial Neural Network Models for Forecasting and Decision Making_ suggests that ANN have been widely touted as solving many forecasting and decision modeling problems [3].

SVMs classify facial expressions by analyzing images as pixel arrays, extracting key features such as facial contours and textures. They identify the optimal boundary (hyperplane) that separates different expressions, focusing on maximizing the distance between the closest points (support vectors) of each expression category for precise classification.

## (Potential) Results and Discussion

Key metrics to assess the performance of our machine learning model are **Precision**, **Recall** and **Accuracy**.

1. **Precision**: the proportion of samples predicted as positive by the model that are truly positive, aims to minimize false positives.
2. **Recall**: the proportion of truly positive samples correctly predicted by the model, focuses on minimizing false negatives.
3. **Overall accuracy**: the ratio of all correctly predicted samples to the total number of samples, is a fundamental metric for assessing the model's effectiveness.

Project goals:

1. Achieve a high overall accuracy (exceeding 85% on a held-out test dataset) in sentiment prediction
2. Strike a balance between precision and recall, ensuring that predictions are not biased towards positive or negative sentiments.

## Contribution

| Member Name   | Proposal Contributions                                                         |
| ------------- | ------------------------------------------------------------------------------ |
| Shize Sheng   | - Introduction / Background <br> - Potential Data Set <br> - GitHub Page       |
| Yuanhong Zhou | - Methods <br> - Reference <br> - Data preprocessing <br> - Video presentation |
| Chunzhen Hu   | - Video slides <br> - Data preprocessing <br> - Video presentation             |
| Jiasheng Cao  | - Results and Discussion <br> - Reference <br> - Methods                       |
| Xingyu Hu     | - Problem Definition / Motivation <br> - Video presentation                    |

## Gantt Chart

[https://docs.google.com/spreadsheets/d/1UAXvvqgorT2BbgdRRo9CnRNFgh6w91AE8r79ES6mEFg/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1UAXvvqgorT2BbgdRRo9CnRNFgh6w91AE8r79ES6mEFg/edit?usp=sharing)

## References

[1] Fodor, Kristián & Balogh, Zoltán & Molnár, György. (2023). Real-time Emotion Recognition in Smart Homes. 10.1109/SACI52869.2023.10158664.

[2] P. Giannopoulos, I. Perikos, and I. Hatzilygeroudis, "Deep learning approaches for facial emotion recognition: A case study on fer2013," _Advances in Hybridization of Intelligent Methods_, pp. 1–16, Oct. 2017. DOI: 10.1007/978-3-319-66790-4_1

[3] T. Hill, L. Marquez, M. O'Connor, and W. Remus, "Artificial Neural Network Models for Forecasting and Decision Making," _International Journal of Forecasting_, vol. 10, no. 1, pp. 5-15, 1994. DOI: 10.1016/0169-2070(94)90045-0

[4] B. Juba and H. S. Le, "Precision-Recall versus Accuracy and the Role of Large Data Sets," _AAAI_, vol. 33, no. 01, pp. 4039-
