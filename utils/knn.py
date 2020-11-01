import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        dists = self.compute_distances(X)

        return self.predict_labels(dists)

    def compute_distances(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        
        diff = abs(X[: , None] - self.train_X[None, :])
        diff = diff.reshape(num_train*num_test, -1)
        dists = np.sum(diff, axis = 1).reshape(-1, num_train)

        return dists

    def predict_labels(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
            '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            counter = 0
            k_min_ind = np.argpartition(dists[i], self.k)[:self.k]
            for j in k_min_ind:
                cur_freq = list(self.train_y).count(self.train_y[j])
                if cur_freq > counter:
                    counter = cur_freq
                    res_ind = j
            pred[i] = self.train_y[res_ind]

        return pred
