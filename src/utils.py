import numpy as np
import model_dispatcher
import config
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt



class model_selection:
    def __init__(self, shuffle = True):
        self.shuffle = shuffle
        
    def train_test_split(self, X, Y, test_ratio = 0.2):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.test_ratio = test_ratio
        self.n_samples = self.X.shape[0]
        self.n_test = int(self.n_samples * self.test_ratio)
        
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
    
        test_indices = indices[:self.n_test]
        train_indices = indices[self.n_test:]
    
        X_train, X_test = self.X[train_indices, :], self.X[test_indices, :]
        Y_train, Y_test = self.Y[train_indices], self.Y[test_indices]
        
        return X_train, X_test, Y_train, Y_test

    def cross_val_score(self, kernel_type, X, Y, cv = 5):
        self.x = np.array(X)
        self.y = np.array(Y)
        self.model = model_dispatcher.KernelRidgeRegression(kernel_type)
        self.kernel_type = kernel_type
        self.samples = self.x.shape[0]
        fold_size = self.samples // cv
        indices = np.arange(self.samples)
        np.random.shuffle(indices)
        bs = '\033[1m'
        be = '\033[0m'
        mse_score = []
        mae_score = []
        r_sq_score = []

        print(f"\n{bs}{self.kernel_type}{be}\n")
        for fold in range(cv):
            if fold == cv-1:
                validation_indices = indices[fold*fold_size :]
                train_indices = np.concatenate([indices[:fold*fold_size], indices[(fold+1)*fold_size:]])
            else:
                validation_indices = indices[fold*fold_size : (fold+1)*fold_size]
                train_indices = np.concatenate([indices[:fold*fold_size], indices[(fold+1)*fold_size:]])
            
            x_train, x_val = self.x[train_indices], self.x[validation_indices]
            y_train, y_val = self.y[train_indices], self.y[validation_indices]
            
            model = self.model
            model.fit(x_train, y_train)
            prediction = model.predict(x_val)

            mse = mean_squared_error(prediction, y_val)
            mae = mean_absolute_error(prediction, y_val)
            r2 = r2_score(prediction, y_val)
            
            mse_score.append(mse)
            mae_score.append(mae)
            r_sq_score.append(r2)
            print(f"MSE : {round(mse, 3)}\t MAE : {round(mae, 3)}\t R2 : {round(r2, 3)}")
        return mse_score, mae_score, r2_score, 

class StandardScaler:
    def __init__(self):
        self.mean_ = None     #this '_' indicates that these variables are derived from the 
        self.scale_ = None    # data during the fitting process.

    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis = 0)
        self.scale_ = np.std(X, axis = 0)

    def transform(self, X):
        X = np.array(X)
        return (X-self.mean_)/(self.scale_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MinMaxScaler:
    def __init__(self, feature_range = (0, 1)):
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.feature_range = feature_range

    def fit(self, X):
        X = np.array(X)
        self.data_min_ = np.min(X, axis = 0)
        self.data_max_ = np.max(X, axis = 0)
        self.min_, self.max_ = self.feature_range

    def transform(self, X):
        X = np.array(X)
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        X_scaled = X_std * (self.max_ - self.min_) + self.min_
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class Ploting:
    def __init__(self, params, kernel_type, x_train, x_test, y_train, y_test):
        '''
        params,
        kernel_type,
        x_train,
        y_train
        '''
        self.params = params
        self.kernel_type = kernel_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def plot(self, mse, mae, rsq, title, save):
        plt.figure(figsize = (8, 6))
        plt.plot(np.arange(len(mse)), mse, marker = 'o', linestyle = '-', color = 'b', label = 'MSE')
        plt.plot(np.arange(len(mae)), mae, marker = 'x', linestyle = '-', color = 'y', label = 'MAE')
        plt.plot(np.arange(len(rsq)), rsq, marker = '*', linestyle = '-', color = 'r', label = 'R2_score')
        plt.xlabel('Iterations')
        plt.ylabel('scores')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save)

    def tuning_plot(self):
        bs = '\033[1m'
        be = '\033[0m'
        mae = dict()
        mse = dict()
        rsq = dict()
        save = None
        print(f"{bs}{self.kernel_type}{be}")
        for i in range(1, self.params):
            print(f"Iteration {i}")
            if self.kernel_type in ['anova', 'spline']:
                model = model_dispatcher.KernelRidgeRegression(kernel_type = self.kernel_type, p = i, d = i)
            else:
                model = model_dispatcher.KernelRidgeRegression(kernel_type = self.kernel_type, sigma = 1 + (i/10))
            model.fit(self.x_train, self.y_train)
            prediction = model.predict(self.x_test)
            mae[i] = mean_absolute_error(self.y_test, prediction)
            mse[i] = mean_squared_error(self.y_test, prediction)
            rsq[i] = r2_score(self.y_test, prediction)
        if self.kernel_type == 'gaussian':
            save = config.GAUSSSIAN
        if self.kernel_type == 'spline':
            save = config.SPLINES
        if self.kernel_type == 'anova':
            save = config.ANOVA

        self.plot([i[1] for i in mse.items()], [i[1] for i in mae.items()], [i[1] for i in rsq.items()], self.kernel_type, save)
        return mae, mse, rsq