import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg


class FastGradRidgeLogit(object):
    """Fast Gradient Descent Ridge Logistic Regression
        
        Generate step size by backtracking rule.
    """

    opt_betas = None
    beta_vals = None
    lambduh = None
    x_train = None
    y_train = None

    def __init__(self):
        pass


    def objective(self, beta, lambduh, x, y):
        """objective(ndarray, float, ndarray, ndarray)

        'Compute objective value of l2 logistic regression'
        
        Args:
            beta (ndarray): Coefficient of l2 logistic regression (1 x n)
            lambduh (float): Regularization parameter.
            x (ndarray): Features to input. (m x n)
            y (ndarray): Response variables. (m x 1)

        Returns:
            float: Objective value of l2 logistic regression.
        """
        obj = 1/len(y) * np.sum(np.log(1 + np.exp(-y*x.dot(beta)))) + lambduh * np.linalg.norm(beta)**2

        return obj


    def fit(self,lambduh, x, y, maxiter = 300):
        """fit(float, ndarray, ndarray, int)

        'Perform fast gradient descent l2 logistic regression'
        
        Args:
            lambduh (float): Regularization parameter.
            x (ndarray): Features to input. (m x n)
            y (ndarray): Response variables. (m x 1)
            maxiter (int): Maximum number of iterations to run the algorithm

        Returns:
            ndarray: Array contains beta values of each iteration. (n x maxiter)
        """

        self.x_train = x
        self.y_train = y
        self.lambduh = lambduh

        d = np.size(x, 1)
        beta_init = np.zeros(d)
        theta_init = np.zeros(d)
        eta_init = 1/(scipy.linalg.eigh(1/len(y)*x.T.dot(x), eigvals=(d-1, d-1), eigvals_only=True)[0]+lambduh)

        beta = beta_init
        theta = theta_init
        grad_theta = self.__computegrad(theta, lambduh, x=x, y=y)
        beta_vals = beta
        theta_vals = theta
        
        print("Start fast gradient descent:")
        iter = 0
        while iter < maxiter:
            eta = self.__bt_line_search(theta, lambduh, eta=eta_init, x=x, y=y)
            beta_new = theta - eta*grad_theta
            theta = beta_new + iter/(iter+3)*(beta_new-beta)
            # Store all of the places we step to
            beta_vals = np.vstack((beta_vals, beta_new))
            theta_vals = np.vstack((theta_vals, theta))
            grad_theta = self.__computegrad(theta, lambduh, x=x, y=y)
            beta = beta_new
            iter += 1
            if iter % 100 == 0:
                print('Fast gradient iteration', iter)

        self.opt_betas = beta_vals[-1, :]
        self.beta_vals = beta_vals

        return beta_vals

    
    def plot_objective(self):
        """plot_objective()

        'Plot objective value changes through all iterations'

        """

        if self.beta_vals is None:
            print("You should fit the model first.")
        num_points = np.size(self.beta_vals, 0)
        objs_fg = np.zeros(num_points)
        for i in range(0, num_points):
            objs_fg[i] = self.objective(self.beta_vals[i, :], self.lambduh, x=self.x_train, y=self.y_train)
        fig, ax = plt.subplots()
        ax.plot(range(1, num_points + 1), objs_fg, c='red', label='fast gradient l2 logit')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('Objective value vs. iteration when lambda='+str(self.lambduh))
        ax.legend(loc='upper right')
        plt.show()

    def plot_misclassification_error(self):
         """plot_misclassification_error()

        'Plot misclassification error value changes through all iterations'

        """

        if self.beta_vals is None:
            print("You should fit the model first.")
        niter = np.size(self.beta_vals, 0)
        error_fastgrad = np.zeros(niter)
        for i in range(niter):
            error_fastgrad[i] = self.__compute_misclassification_error(self.beta_vals[i, :], self.x_train, self.y_train)
        fig, ax = plt.subplots()
        ax.plot(range(1, niter + 1), error_fastgrad, c='red', label='fast gradient l2 logit')
        plt.xlabel('Iteration')
        plt.ylabel('Misclassification error')
        plt.title('Misclassification errore vs. iteration when lambda='+str(self.lambduh))
        ax.legend(loc='upper right')
        plt.show()


    #################### help functions ###################################################    
    

    def __computegrad(self,beta, lambduh, x, y):
        """__computegrad(ndarray, float, ndarray, ndarray)

        'Compute gradient of l2 logistic regression'

        Args:
        beta (ndarray): Coefficient of l2 logistic regression (1 x n)
        lambduh (float): Regularization parameter.
        x (ndarray): Features to input. (m x n)
        y (ndarray): Response variables. (m x 1)

        Returns:
        ndarray: Gradient of l2 logistic regression. (1 x n)
        """
        yx = y[:, np.newaxis]*x
        denom = 1+np.exp(-yx.dot(beta))
        grad = 1/len(y)*np.sum(-yx*np.exp(-yx.dot(beta[:, np.newaxis]))/denom[:, np.newaxis], axis=0) + 2*lambduh*beta
        return grad

    def __bt_line_search(self,beta, lambduh, eta, x, y, alpha=0.5, betaparam=0.8,maxiter=100):
        """__bt_line_search(ndarray, float, float, ndarray, ndarray)

        'Perform backtracking line search'
        
        Args:
            beta (ndarray): Coefficient of l2 logistic regression (1 x n)
            lambduh (float): Regularization parameter.
            eta(float): Starting (maximum) step size
            x (ndarray): Features to input. (m x n)
            y (ndarray): Response variables. (m x 1)
            alpha(float): Constant used to define sufficient decrease condition
            betaparam(float): Fraction by which we decrease t if the previous t doesn't work
            maxiter(int): Maximum number of iterations to run the algorithm

        Returns:
            float: Step size to use
        """

        grad_beta = self.__computegrad(beta, lambduh, x=x, y=y)
        norm_grad_beta = np.linalg.norm(grad_beta)
        found_eta = 0
        iter = 0
        while found_eta == 0 and iter < maxiter:
            if self.objective(beta - eta * grad_beta, lambduh, x=x, y=y) < self.objective(beta, lambduh, x=x, y=y)- alpha * eta * norm_grad_beta ** 2:
                found_eta = 1
            elif iter == maxiter - 1:
                print('Warning: Max number of iterations of backtracking line search reached')
            else:
                eta *= betaparam
                iter += 1
        return eta


    def __compute_misclassification_error(self, beta_opt, x, y):
        """__compute_misclassification_error(ndarray, ndarray, ndarray)

        'Compute misclassification error'

        Args:
        beta (ndarray): Coefficient of l2 logistic regression (1 x n)
        x (ndarray): Features to input. (m x n)
        y (ndarray): Response variables. (m x 1)

        Returns:
        float: Misclassification error value of l2 logistic regression.
        """

        y_pred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
        y_pred = y_pred*2 - 1  # Convert to +/- 1
        return np.mean(y_pred != y)

    



    