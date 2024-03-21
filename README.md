# COMP425-HW3
## Overview
This project is dedicated to the implementation of a visual recognition system using the Bag-of-Words (BoW) model, particularly focused on texture recognition. It involves creating a filter bank, extracting features from images, clustering these features to create "visual words," and finally, classifying images based on their histogram representations over these visual words.

## Key Components
- Filter Bank Creation: Utilize LMFilters.py to generate a bank of 48 filters for feature extraction.
- Feature Extraction and Clustering: Apply filters to training images to extract features, followed by K-means clustering to form visual words.
- BoW Representation and Classification: Represent images as histograms over the visual words and classify them using a nearest-neighbor classifier.

## Running the code
To run the components of this assignment, navigate to the project directory in your terminal and execute the specified Python scripts. Make sure you have placed the training and test images in the appropriate directory as outlined in the assignment instructions.
- Generating BoW Model: Run `python run_train.py` to generate the textons and BoW representations for the training images. This will create a model file **model.pkl**.
- Classifying Test Images: Execute python `run_test.py` to classify the test images based on the previously generated model.

Each script will carry out its task according to the specifications provided in the assignment document. The results should closely align with the expected outcomes if the implementation is correct
