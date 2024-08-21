import models
import numpy as np

class KernelRidgeRegression:
    def __init__(self, **kwargs):
        self.kernel_type = kwargs.get('kernel_type', None)
        self.p = kwargs.get('p', 2)   # anova
        self.d = kwargs.get('d', 2)   # spline
        self.lambd = kwargs.get('lambd', 1.0)
        self.sigma = kwargs.get('sigma', 1.0)
        self.kernel = models.kernel()
        self.K_train = None
        self.K_test = None
    
    def fit(self, X, y):
        '''
        (K + a*I)*alpha = y
        => alpha = (pow((K + a*I), -1))*y
        Instead of doing the INVERSE computation, I have used 
        np.linalg.solve() : to solve the system of linear equations(more stable method), 
        '''
        self.X_train = X

        # calculating the Kernel Matrix on Training data
        if self.kernel_type == 'anova':
            self.K_train = self.kernel.anova_kernel(X, X, self.p, True)
        elif self.kernel_type == 'spline':
            self.K_train = self.kernel.spline_kernel(X, X, self.d)
        else:
            self.K_train = self.kernel.gaussian_kernel(X, X, self.sigma)

        # calculating alpha
        self.alpha = np.linalg.solve(self.K_train + self.lambd * np.eye(len(X)), y)
    
    def predict(self, X):
        if self.kernel_type == 'anova':
            self.K_test = self.kernel.anova_kernel(X, self.X_train, self.p, True)
        elif self.kernel_type == 'spline':
            self.K_test = self.kernel.spline_kernel(X, self.X_train, self.d)
        else:
            self.K_test = self.kernel.gaussian_kernel(X, self.X_train, self.sigma)
            
        return self.K_test.dot(self.alpha)
