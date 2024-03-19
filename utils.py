from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist


def createTextons(F, file_list, K):
    """
    Compute the textons (cluster centers) from training images.

    :param F: Filter bank
    :param file_list: List of filenames corresponding to training images
    :param K: Number of clusters
    :return: It returns a K*48 matrix where each row corresponds to a cluster center
    """

    responses = []  # To accumulate all filter responses from all images

    for file in file_list:
        # Load the image as float to ensure compatibility with the filters
        img = img_as_float(io.imread(file))

        # Convert to grayscale if needed
        if img.ndim == 3:
            img = rgb2gray(img)

        # Apply the 48 filters in the filter bank and collect all 48-dimensional vectors of all pixels from all training
        # images
        for i in range(F.shape[2]):
            filtered_img = correlate(img, F[:, :, i], mode="reflect")
            sampled_pixels = np.random.choice(filtered_img.flatten(), 100, replace=False)
            responses.append(sampled_pixels)

    # Reshape responses to a 2D array where each row is a response vector
    responses = np.array(responses).reshape(-1, F.shape[2])

    # Use KMeans for clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=K)
    kmeans.fit(responses)
    textons = kmeans.cluster_centers_
    return textons


def computeHistogram(img_file, F, textons):
    """
    Compute the Bag of Words (BoW) histogram of an image.

    :param img_file: Filename of an image
    :param F: Filter bank
    :param textons: The matrix of texton vectors
    :return: It returns a K dimension vector for the BoW representation of an image
    """
