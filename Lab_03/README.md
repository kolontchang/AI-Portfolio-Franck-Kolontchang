# Lab 03: CNNs - Puppy vs Bagel

## Problem Statement
Image classification tasks often struggle with visually similar objects. This lab tackles the classic "Puppy vs Bagel" classification problem to demonstrate the feature extraction capabilities of Convolutional Neural Networks (CNNs).

## Approach and Methodology
Developed a Convolutional Neural Network utilizing PyTorch. The architecture included convolutional layers for extracting spatial hierarchies, max-pooling layers for downsampling, and dense layers for final binary classification. Data augmentation was used to improve generalization.

## Results and Evaluation
The CNN achieved high accuracy in distinguishing between complex, visually similar features of puppies and bagels, significantly outperforming a standard feed-forward network baseline.

## Your Learning Outcomes
I learned how convolutional filters learn spatial patterns like edges, textures, and shapes. I also gained practical experience in mitigating overfitting in image data using dropout and image augmentation techniques.

## Requirements or Dependencies
* Ensure `requirements.txt` from the root directory is installed.
* Standard Python 3.8+ environment with PyTorch.

## Sample Data
* Instructions for data access or the necessary subsets are detailed within the respective Jupyter Notebook cells or provided through standard torch/ torchvision datasets.
