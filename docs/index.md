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

## Data Preprocessing Method Implemented

### Gray Scale Conversion

For the data preprocessing pipeline, we start by converting every image of the training and testing dataset from its original format to grayscale, which simplifies the data representation and reduces the computational complexity. The grayscale images are represented as matrices.

### Emotion Labeling

The corresponding emotion label is assigned to each image based on the folder, and this relationship is stored as a list of dictionaries.

### Data augmentation(training data)

We also incorporated several data augmentation techniques to a portion of the training data such as random movement, rotation, flipping, and zooming. After data augmentation.

### Data shuffling

We shuffled the processed training dataset randomly to ensure our model is not biased on the specific order of the training data.

### Data saving

Finally, we saved the preprocessed training and testing datasets to the Numpy file to allow for convenient access in later pipelines.

## Visualizations

### Analyzing the Training & Testing Dataset

From the preprocessed training and testing datasets (stored as numpy arrays), we first converted them as pandas dataframes with columns for 'emotion' labels and 'pixels’, the latter being flattened arrays representing the image data.

Returning the dataset shapes

- The training set has 41,463 individual data samples (testing set with 7178 samples) and 2 features (emotion, flattened pixel array of the image)

  - Training Data Shape: (41463, 2)
  - Test Data Shape: (7178, 2)

- Previewing the first 5 rows of the training & testing dataframe
- Emotion Distribution

### Visualizing the Training Dataset

- For visualizing the emotion distribution within the training dataset, we used the seaborn visualization library in conjunction matplotlib to output a barplot displaying each of the 7 emotions and their corresponding occurrence within the set.

- Emotions were sorted in ascending order based on their occurrence, with the resulting plot showcasing a spectrum of emotions from 'disgust' (least common) to 'happy' (most common).

![CNN Training Set Distribution](/docs/images/cnn_distribution.png)

### Visualizing the Testing Dataset

- The same visualization process was used to visualize the testing dataset. The emotion frequencies reflect a similar distribution pattern as the training dataset, where 'disgust having the least occurrence and happy having the most.

![CNN Testing Set Distribution](/docs/images/cnn_test_distribution.png)

## Method Implemented (CNN)

### Why we used CNN

CNNs are preferred for facial expression recognition due to their efficient extraction of local features through convolutional operations with shared parameters and their ability to improve model generalization by training filters. They possess parameter sharing and local connectivity properties, resulting in fewer parameters and computations, making them suitable for large-scale data processing. Additionally, CNNs utilize secondary sampling and parameter reduction via pooling layers to preserve salient features, decrease computational load, and prevent overfitting. Their translation invariance, achieved through locally connected patterns, is particularly beneficial for facial expression recognition tasks where facial features may vary in location. Furthermore, CNNs employ fully connected layers to alter feature map dimensions and generate probabilities for classification categories, facilitating image classification.

### Quantitative Metrics

We use accuracy and loss to show how well an ML model learns over time (or epochs). Accuracy shows us how well the model makes correct predictions during training. Loss represents the deviation of the model’s predicted and true values. To evaluate an ML model, a low loss value and a high accuracy indicate that model learning is effective.

- For training and validation loss, the x-axis represents the epochs number, and the y-axis represents the loss values. Picture 1 shows how the loss value changes with the increasing of epochs.

- For training and validation accuracy, the x-axis represents the epochs number, and the y-axis represents the accuracy. Picture 2 shows how accuracy changes with the increasing of epochs.

Our model's training loss decreases as epochs increase and eventually approaches 0, while our validation loss decreases and then increases as epochs increase. The training accuracy of our model has been increasing with epochs to almost 100% accuracy, and the validation accuracy increases with epochs and then stabilizes to around 52%.

Our model can achieve high accuracy as well as low loss for learning on the training data with the performance of the CNN algorithm, with high prediction loss and stabilized accuracy on the test data.

### Analysis of the CNN Model

We used a CNN model for this multi-classification task. In our model, we used 3 convolutional layers to extract features from the image and pooling layer to reduce the amount of data and speed up the training.

After this, we flatten all the features. This step is to bridge the fully connected layers that come later. We used two fully connected layers for subsequent training. The fully connected layer reduces the value of the loss function by continuously adjusting the weights during training. In all of these processes, we use the Relu function as the activation function, which prevents the gradient from disappearing and improves the speed of training.

Finally, we used a fully connected layer and a sigmoid function for the final classification, the number of points in this layer is equal to the number of classifications so that we can get a probabilistic output for each classification.

### Next Steps

We have tried various transformations such as translation rotation of images in data preprocessing, but our results did not improve well. After that we may try to increase the filter of the convolutional layer to make the image with more feature values to improve the accuracy. Or a hybrid approach to data preprocessing.

After that we will implement the SVM model, a supervised learning model, which can also efficiently perform classification tasks well suited for our facial expression recognition task.

We are going to try more different parameters and see which ones lead to the best training results. This includes, but is not limited to, changing the number of layers in the fully connected layer, the number of layers in the convolutional layer, the batch size, and the number of convolutional kernels. Since we perform data enhancement during data preprocessing, including translation, rotation, mirroring, and zooming, we are also going to try to adjust their parameters to get relatively better results.

## Proposed Data Methods

We plan to use **CNN (feature extraction)**, **Neural network (classification)**, and **Support Vector Machine (SVM)**, for facial expression machine learning.
From _Deep Learning Approaches for Facial Emotion Recognition: A Case Study on FER-2013_, CNN can be used for facial expression recognition as a supervised machine learning method [2]. We will perform prediction by forward propagation, calculate the loss function using the labels given in the dataset and perform weight update by backward propagation.

We use neural networks to forecast and classification as _Artificial Neural Network Models for Forecasting and Decision Making_ suggests that ANN have been widely touted as solving many forecasting and decision modeling problems [3].

SVMs classify facial expressions by analyzing images as pixel arrays, extracting key features such as facial contours and textures. They identify the optimal boundary (hyperplane) that separates different expressions, focusing on maximizing the distance between the closest points (support vectors) of each expression category for precise classification.

## Proposed (Potential) Results and Discussion

Key metrics to assess the performance of our machine learning model are **Precision**, **Recall** and **Accuracy**.

1. **Precision**: the proportion of samples predicted as positive by the model that are truly positive, aims to minimize false positives.
2. **Recall**: the proportion of truly positive samples correctly predicted by the model, focuses on minimizing false negatives.
3. **Overall accuracy**: the ratio of all correctly predicted samples to the total number of samples, is a fundamental metric for assessing the model's effectiveness.

Project goals:

1. Achieve a high overall accuracy (exceeding 85% on a held-out test dataset) in sentiment prediction
2. Strike a balance between precision and recall, ensuring that predictions are not biased towards positive or negative sentiments.

## Contribution

| Member Name   | Proposal Contributions                                                                                                                                                                                    |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shize Sheng   | - Implement Training & Dataset Visualizations <br> - Implement CNN Model Evaulations <br> - Midterm Report: Visualization <br> - GitHub Repository (README) Modification <br> - Configure to Github Pages |
| Yuanhong Zhou | - Midterm Report: Quantitative Metrics                                                                                                                                                                    |
| Chunzhen Hu   | - Midterm Report: CNN Model Explainations <br> - Midterm Report: Next Steps                                                                                                                               |
| Jiasheng Cao  | - Implement Data Preprocessing <br> - Implement CNN Training <br> - Midterm Report: Analysis of CNN <br> - Midterm Report: Next Steps                                                                     |
| Xingyu Hu     | - Midterm Report: Data Preprocessing                                                                                                                                                                      |

## Gantt Chart

[https://docs.google.com/spreadsheets/d/1UAXvvqgorT2BbgdRRo9CnRNFgh6w91AE8r79ES6mEFg/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1UAXvvqgorT2BbgdRRo9CnRNFgh6w91AE8r79ES6mEFg/edit?usp=sharing)

## References

[1] Fodor, Kristián & Balogh, Zoltán & Molnár, György. (2023). Real-time Emotion Recognition in Smart Homes. 10.1109/SACI52869.2023.10158664.

[2] P. Giannopoulos, I. Perikos, and I. Hatzilygeroudis, "Deep learning approaches for facial emotion recognition: A case study on fer2013," _Advances in Hybridization of Intelligent Methods_, pp. 1–16, Oct. 2017. DOI: 10.1007/978-3-319-66790-4_1

[3] T. Hill, L. Marquez, M. O'Connor, and W. Remus, "Artificial Neural Network Models for Forecasting and Decision Making," _International Journal of Forecasting_, vol. 10, no. 1, pp. 5-15, 1994. DOI: 10.1016/0169-2070(94)90045-0

[4] B. Juba and H. S. Le, "Precision-Recall versus Accuracy and the Role of Large Data Sets," _AAAI_, vol. 33, no. 01, pp. 4039-
