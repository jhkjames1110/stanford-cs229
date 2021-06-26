import numpy as np
import matplotlib.pyplot as plt
import argparse

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    ### YOUR CODE HERE
    maxVal = np.max(x, axis=1, keepdims=True)

    numerator = np.exp(x - maxVal)
    denominator = np.sum(numerator, axis=1, keepdims=True)

    return numerator/denominator
    ### END YOUR CODE


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1 / (1 + np.exp(-x))
    # *** END CODE HERE ***


def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    paramsDict = {}
    weightOne = np.random.normal(0, 1, (input_size, num_hidden))
    weightTwo = np.random.normal(0, 1, (num_hidden, num_output))
    biasOne = np.zeros(num_hidden)
    biasTwo = np.zeros(num_output)

    paramsDict['weightOne'] = weightOne
    paramsDict['weightTwo'] = weightTwo
    paramsDict['biasOne'] = biasOne
    paramsDict['biasTwo'] = biasTwo

    return paramsDict
    # *** END CODE HERE ***

def z_function(input, weight, bias):
    return input @ weight + bias
    #return np.transpose(weight) @ input + bias


def crossEntropy(labels, yHypothesis):
    lossArray = -np.multiply(np.log(yHypothesis + 10**(-16)), labels)
    return lossArray.sum() / yHypothesis.shape[0]

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    weightOne = params['weightOne']
    weightTwo = params['weightTwo']
    biasOne = params['biasOne']
    biasTwo = params['biasTwo']

    batchSize = data.shape[0]
    hiddenLayer = biasOne.shape[0]
    outputLayer = biasTwo.shape[0]
    
    aOnes = np.zeros((batchSize, hiddenLayer))
    zTwo = np.zeros((batchSize, outputLayer))

    '''
    for i in range(batchSize):
        zOne = z_function(data[i, :], weightOne, biasOne)
        aOnes[i] = np.transpose(sigmoid(zOne).reshape(hiddenLayer, 1))
        zTwo[i] = z_function(aOnes[i, :], weightTwo, biasTwo)
    '''
    zOne = z_function(data, weightOne, biasOne)
    aOnes = sigmoid(zOne)
    zTwo = z_function(aOnes, weightTwo, biasTwo)
    yHypothesis = softmax(zTwo)

    averageLoss = crossEntropy(labels, yHypothesis)

    return (aOnes, yHypothesis, averageLoss)
    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***   
    
    weightTwo = params['weightTwo']
    a1, h, loss = forward_prop_func(data, labels, params)
    
    dW2 = np.transpose(a1) @ (h - labels)
    dB2 = np.sum(h - labels, axis = 0, keepdims = True) 
    dW1 = np.transpose(data) @ (((h - labels) @ np.transpose(weightTwo)) * (a1 * (1 - a1)))
    dB1 = np.sum(((h - labels) @ np.transpose(weightTwo)) * (a1 * (1 - a1)), axis = 0, keepdims = True)

    backpropDict = {
        'dW2': dW2 / data.shape[0],
        'dW1': dW1 / data.shape[0],
        'dB2': dB2 / data.shape[0],
        'dB1': dB1 / data.shape[0]
    }

    return backpropDict
    
    # *** END CODE HERE ***


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    weightOne = params['weightOne']
    weightTwo = params['weightTwo']
    a1, h, loss = forward_prop_func(data, labels, params)
    
    dW2 = np.transpose(a1) @ (h - labels)
    dB2 = np.sum(h - labels, axis = 0, keepdims = True) 
    dW1 = np.transpose(data) @ (((h - labels) @ np.transpose(weightTwo)) * (a1 * (1 - a1)))
    dB1 = np.sum(((h - labels) @ np.transpose(weightTwo)) * (a1 * (1 - a1)), axis = 0, keepdims = True)


    backpropDict = {
        'dW2': (dW2 / data.shape[0]) + 2 * reg * weightTwo,
        'dW1': (dW1 / data.shape[0]) + 2 * reg * weightOne,
        'dB2': dB2 / data.shape[0],
        'dB1': dB1 / data.shape[0]
    }

    return backpropDict

    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    totIteration = train_data.shape[0] / batch_size

    for i in range(np.int(totIteration)):
        trainData = train_data[i*1000:(i*1000)+1000, :]
        trainLabels = train_labels[i*1000:(i*1000)+1000, :]
        backpropDict = backward_prop_func(trainData, trainLabels, params, forward_prop_func)

        params['weightOne'] = params['weightOne'] - learning_rate * backpropDict['dW1']
        params['weightTwo'] = params['weightTwo'] - learning_rate * backpropDict['dW2']
        params['biasOne'] = params['biasOne'] - learning_rate * backpropDict['dB1']
        params['biasTwo'] = params['biasTwo'] - learning_rate * backpropDict['dB2']
    # *** END CODE HERE ***

    # This function does not return anything    
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)
        
    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
