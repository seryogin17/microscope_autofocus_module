import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''   
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    
    if predictions.ndim == 2:
        expons = np.exp(predictions.T - np.max(predictions, 1)).T
        probs = expons / np.sum(expons, 1).reshape(1, -1).T
    elif predictions.ndim == 1:
        expons = np.exp(predictions - np.max(predictions))
        probs = expons / np.sum(expons)
    else:
        raise Exception("Wrong array shape!")

    return probs
    
    
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value for the entire batch
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    
    return -np.log(np.choose(target_index, probs.T))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    if predictions.ndim == 1:
        probs[target_index] -= 1 # -1 is the result of algebraic calculations left behind (dCE/dZ, where Z=XW+B)
    else:
        probs[np.arange(probs.shape[0]), target_index] -= 1
    dprediction = probs
    
    return loss, dprediction
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss
    '''
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    
    predictions = np.dot(X, W)
    loss, grad_dZ = softmax_with_cross_entropy(predictions, target_index)
    grad_dW = np.dot(X.T,grad_dZ) / X.shape[0] # norming is needed to find the mean of grad for all samples in batch

    return loss.mean(), grad_dW

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    
    return loss, grad


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, loader, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_features = 3*224*224
        num_classes = 5
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
             for i_step, (x, y) in enumerate(loader):
                loss, grad = linear_softmax(x.reshape(x.shape[0], -1).numpy(), self.W, y.numpy())
                loss_reg, grad_reg = l2_regularization(self.W, reg)
                self.W -= learning_rate * (grad + grad_reg)
                loss_history.append(loss + loss_reg)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''

        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        return y_pred



                
                                                          

            

                
