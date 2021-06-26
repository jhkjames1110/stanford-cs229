from operator import xor
import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    
    #Train Data
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    #Validate
    if(train_path == "ds1_train.csv"):
        util.plot(x_valid, y_valid, clf.theta, "logreg_valid1.png", correction = 1.0)
    else:
        util.plot(x_valid, y_valid, clf.theta, "logreg_valid2.png", correction = 1.0)

    #Test Case
    np.savetxt(save_path, clf.predict(x_valid))

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def xTheta(self, x, theta):
        return x@theta
    
    def gFunction(self, input):
        return 1 / (1 + np.exp(-input))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        tol = 1e-2
        theta = np.zeros(x.shape[1])
        while(tol>self.eps):
            theta_prev = theta.copy()
            gs = self.gFunction(self.xTheta(x, theta))
            grad_J = np.mean(np.multiply((gs-y).reshape((x.shape[0], 1)), x), axis = 0)
            hessian = np.zeros((x.shape[1], x.shape[1]))
            for i in range(hessian.shape[0]):
                for j in range(hessian.shape[0]):
                    hessian[i][j] = np.mean(np.prod([gs, (1-gs), x[:,i], x[:,j]], axis = 0))
            theta = theta - np.linalg.inv(hessian)@grad_J
            tol = np.linalg.norm((theta-theta_prev), ord = 1)
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self.gFunction(self.xTheta(x, self.theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
