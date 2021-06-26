import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset('ds1_test.csv', add_intercept = True)


    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    clf = GDA()
    clf.fit(x_train, y_train)
    theta = np.insert(clf.theta, 0, clf.theta0) #insert clf.theta0 as first index of theta to plot
    util.plot(x_valid, y_valid, theta, "outgda_train2.png", correction = 1.0)
    np.savetxt(save_path, clf.predict(x_test))
    # *** END CODE HERE ***



class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        self.theta0 = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        def count_ones(y): 
            # 1{yi = 1} 
            # y: Training example labels. Shape (n_examples, ) => d x undefined
            counter = 0
            y = y.reshape((y.shape[0], 1)) # Change y shape to d x 1
            for i in y:
                if (i == 1):
                    counter += 1
            
            return counter


        def phi(x,y):
            return((1/(x.shape[0]))*count_ones(y))

        def mu(x,y):
            sum_vec0 = np.zeros((1, x.shape[1]))
            sum_vec1 = np.zeros((1, x.shape[1]))

            for i in range(y.shape[0]):
                if y[i] == 0:
                    sum_vec0 = sum_vec0 + x[i]
                else:
                    sum_vec1 = sum_vec1 + x[i]
            mu_1 = np.transpose(sum_vec1) / count_ones(y)
            mu_0 = np.transpose(sum_vec0) / (x.shape[0] - count_ones(y))
            return(mu_0, mu_1)

        def sigma(x, y):
            n = x.shape[0]
            sum_vec = np.zeros((x.shape[1], x.shape[1]))
            for i in range(n):
                holder = (x[i].reshape((x[i].shape[0],1)))
                update = np.dot((holder - mu(x,y)[int(y[i])]), np.transpose((holder - mu(x,y)[int(y[i])])))
                sum_vec = sum_vec + update

            return (sum_vec/n)

        phi = phi(x,y)
        mu_0 = mu(x, y)[0]
        mu_1 = mu(x, y)[1]
        sigma = sigma(x,y)

        self.theta0 = 0.5 * ((np.transpose(mu_0) @ np.linalg.inv(sigma) @ mu_0) - (np.transpose(mu_1) @ np.linalg.inv(sigma) @ mu_1)) + np.log(phi/(1-phi))
        theta = -(np.transpose(mu_0 - mu_1) @ np.linalg.inv(sigma))
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        theta = np.insert(self.theta, 0, self.theta0)
        tx = x.dot(theta)
        gs = 1/ (1 + np.exp(-tx))
        return gs
        # *** END CODE HERE

if __name__ == '__main__':
    """
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')
    """

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')

