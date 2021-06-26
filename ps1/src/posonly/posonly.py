import numpy as np
from numpy.lib.npyio import save
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'

def filterPosY(x, y):
        n = x.shape[0]
        filtered = list()
        for i in range( n ):
            if int( y[i] ) == 1:
                filtered.append( x[i] )
        result = np.zeros( (len( filtered ), x.shape[1]) )
        for i in range( len( result ) ):
            result[i] = filtered[i]
        return result

def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path, label_col = 't', add_intercept = True)
    x_train, y_train = util.load_dataset(train_path,label_col = 'y', add_intercept = True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col = 't', add_intercept = True)
    x_valid, y_valid = util.load_dataset(valid_path, label_col = 'y', add_intercept = True)
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    clf = LogisticRegression()
    clf.fit(x_train, t_train)
    print(clf.theta)
    x_test, t_test = util.load_dataset(test_path, add_intercept = True)
    util.plot(x_test, t_test, clf.theta, "test_t.png", correction = 1.0)
    np.savetxt(save_path, clf.predict(x_test))
    # Part (b): Train on y-labels and test on true labels
    clf.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept = True)
    util.plot(x_test, y_test, clf.theta, "test_y.png", correction = 1.0)
    np.savetxt(save_path, clf.predict(x_test))
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (f): Apply correction factor using validation set and test on true labels
    xPos = filterPosY(x_valid, y_valid)
    alpha = np.sum( clf.predict( xPos ) ) / xPos.shape[0]
    util.plot(x_valid, t_valid, clf.theta, '2f_validation.jpg', correction=alpha)
    prediction = clf.predict(x_valid)
    print( clf.theta )
    np.savetxt(output_path_true, prediction)
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
