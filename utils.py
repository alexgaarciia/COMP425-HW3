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

    # Initialize an empty list to store the response of applying filters to images. This will be used to collect and
    # accumulate feature vectors from all images
    responses = []

    for file in file_list:
        # Load the image as float to ensure compatibility with the filters
        img = img_as_float(io.imread(file))

        # Convert to grayscale if needed
        if img.ndim == 3:
            img = rgb2gray(img)

        # Array to store the filter responses for this image
        img_responses = np.zeros((img.shape[0], img.shape[1], F.shape[2]))

        # Apply the 48 filters in the filter bank and collect all 48-dimensional vectors of all pixels from
        # all training images
        for i in range(F.shape[2]):
            # Apply the i-th filter to the image. The "correlate" function convolves the filter with the image.
            # In addition, "mode="reflect"" specifies the behavior at the image borders (reflecting the image)
            img_responses[:, :, i] = correlate(img, F[:, :, i], mode="reflect")

        # Reshape the responses so that each row is a 48-dimensional feature vector for a pixel
        all_responses = img_responses.reshape(-1, F.shape[2])

        # As indicated in the instructions, randomly sample 100 pixel values from the filtered image to create
        # a more manageable dataset
        sampled_responses = all_responses[np.random.choice(all_responses.shape[0], 100, replace=False), :]

        # Append the sampled pixel values to the "responses" list
        responses.extend(sampled_responses)

    # Convert the list of responses into a numpy array for clustering
    responses = np.array(responses)

    # Use KMeans for clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_init=10, max_iter=300, tol=1e-4)
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

    # Load the image as float to ensure compatibility with the filters
    img = img_as_float(io.imread(img_file))

    # Convert to grayscale if needed
    if img.ndim == 3:
        img = rgb2gray(img)

    # Create a list to keep each filter's response
    img_responses = np.zeros((img.shape[0], img.shape[1], F.shape[2]))

    # Apply the 48 filters in the filter bank
    for i in range(F.shape[2]):
        img_responses[:, :, i] = correlate(img, F[:, :, i], mode='reflect')

    # Reshape responses to a 2D array where each row is a response vector
    pixels = img_responses.reshape((-1, F.shape[2]))

    # Find the nearest texton for each pixel (done using "cdist", which computes the pairwise distances between two sets
    # of observations)
    distances = cdist(pixels, textons)
    nearest_texton_indices = np.argmin(distances, axis=1)

    # Compute the histogram of texton occurrences.
    histogram, _ = np.histogram(nearest_texton_indices, bins=np.arange(len(textons) + 1), density=True)
    return histogram
