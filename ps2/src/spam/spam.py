import collections

import numpy as np
from numpy.ma.core import maximum_fill_value

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split(" ")
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    wordDict = {}

    for message in messages:
        wordCollection = get_words(message)
        for word in wordCollection:
            if word in wordDict:
                wordDict[word] += 1
            else:
                wordDict[word] = 1
    index = 0
    for count in list(wordDict.keys()):
        if wordDict[count] < 5:
            del wordDict[count]
        else:
            wordDict[count] = index
            index += 1
    return wordDict
    # *** END CODE HERE ***



def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    wordOccurances = np.zeros((len(messages), len(word_dictionary)))
    wordList = list(word_dictionary.keys())
    for message in range(len(messages)):
        normMessage = get_words(messages[message])
        for word in range(len(wordList)):
            occurance = normMessage.count(wordList[word])
            wordOccurances[message][word] = occurance
    return wordOccurances
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    n, V = matrix.shape
    #matrix1, matrix2 is of size (, d)
    matrix1 = matrix[labels == 1, : ].sum(axis = 0)
    matrix0 = matrix[labels == 0, : ].sum(axis = 0)
    #phi_y1, phi_y0 is of size (, d)
    phi_y1 = (1 + matrix1) / (V + matrix1.sum()) #matrix of the probability of choosing a word (given it being in a spam) from the spam word collection
    phi_y0 = (1 + matrix0) / (V + matrix0.sum()) #matrix of the probability of choosing a word (given it being ham) from the ham word collection
    phi_y = np.mean(labels)
    return phi_y0, phi_y1, phi_y
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    
    phi_y0, phi_y1, phi_y = model
    n_messages, n_vocab = matrix.shape
    h = np.zeros(n_messages)
    
    pY1 = (np.log(phi_y) + (np.log(phi_y1) * matrix).sum(axis = 1))
    pY0 = (np.log(1 - phi_y) + (np.log(phi_y0) * matrix).sum(axis = 1))
    
    return(pY1 > pY0).astype(int)
    '''
    for i in range(n_messages):
        log_y1 = np.log(phi_y)
        log_y0 = np.log(1 - phi_y)
        for j in range(n_vocab):
            count = matrix[i][j]
            log_y1 += count * np.log(phi_y1)[j]
            log_y0 += count * np.log(phi_y0)[j]
            print(log_y1)
            print(log_y0)
        h[i] = log_y1 > log_y0
    print(h)
    return h
    '''
    
    
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_y0, phi_y1, phi_y = model
    words = list(dictionary.keys())
    sortedWords = []

    token = np.log((phi_y1)/(phi_y0))
    for word in range(len(words)):
        dictionary[words[word]] = token[word]
    sorted_dict = dict(sorted(dictionary.items(), key = lambda item: item[1], reverse = True))

    for i in range(10):
        sortedWords.extend([list(sorted_dict.keys())[i]])
    return sortedWords
    
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accuracyDic = {}
    for radius in range(len(radius_to_consider)):
        svmPredict = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius_to_consider[radius])
        accMatrix = (svmPredict == val_labels).astype(np.int)
        accuracyDic[accMatrix.sum()] = radius_to_consider[radius]
    return accuracyDic[max(accuracyDic.keys())] 
    # *** END CODE HERE ***

def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
