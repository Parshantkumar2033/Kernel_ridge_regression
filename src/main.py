import numpy as np
import pandas as pd
import config
import model_dispatcher
import utils
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
bs = '\033[1m'
be = '\033[0m'

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return (mse, mae, r2)

def data_preprocessing(dataset_path):
    data = pd.read_csv(dataset_path)
    df = data.copy(deep = True)

    # dropping useless features
    df.drop(columns = ['ID', 'zn', 'chas', 'nox'], inplace = True)

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    split = utils.model_selection()
    x_train, x_test, y_train, y_test = split.train_test_split(X, Y, test_ratio=0.2)

    scale = utils.StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.fit_transform(x_test)

    return x_train, x_test, y_train, y_test


# PARAMETERS TUNING AND PLOTTING

def tuning_plots(x_train, x_test, y_train, y_test):

    g_plot = utils.Ploting(31, kernel_type='gaussian', x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)
    mae_gaussian, mse_gaussian, rsq_gaussian = g_plot.tuning_plot()

    a_plot = utils.Ploting(11, kernel_type='anova', x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)
    mae_anova, mse_anova, rsq_anova = a_plot.tuning_plot()

    s_plot = utils.Ploting(11, kernel_type='spline', x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)
    mae_spline, mse_spline, rsq_spline = s_plot.tuning_plot()
    return mae_gaussian, mse_gaussian, rsq_gaussian, mae_anova, mse_anova, rsq_anova, mae_spline, mse_spline, rsq_spline


# INDIVIDUAL MODELS WITH TUNED PARAMETERS

def gaussian(sigma, x_train, x_test, y_train, y_test):
    model_gaussian = model_dispatcher.KernelRidgeRegression(kernel_type = 'gaussian', sigma = sigma)
    model_gaussian.fit(x_train, y_train)
    gaussian_pred = model_gaussian.predict(x_test)

    print(f"{bs}Gaussian Kernel{be}\n")
    mse, mae, r2 = evaluate_model(gaussian_pred, y_test)
    print(f"MSE : {round(mse, 3)}\t MAE : {round(mae, 3)}\t R2 : {round(r2, 3)}")

def anova(p, x_train, x_test, y_train, y_test):
    model_anova = model_dispatcher.KernelRidgeRegression(kernel_type = 'anova', p = 2)
    model_anova.fit(x_train, y_train)
    anova_pred = model_anova.predict(x_test) 

    print(f"{bs}ANOVA{be}\n")
    mse, mae, r2 = evaluate_model(anova_pred, y_test)
    print(f"MSE : {round(mse, 3)}\t MAE : {round(mae, 3)}\t R2 : {round(r2, 3)}")

def splines(d, x_train, x_test, y_train, y_test):
    model_spline = model_dispatcher.KernelRidgeRegression(kernel_type = 'spline', d = 5)
    model_spline.fit(x_train, y_train)
    spline_pred = model_spline.predict(x_test)

    print(f"{bs}Splines{be}\n")
    mse, mae, r2 = evaluate_model(spline_pred, y_test)
    print(f"MSE : {round(mse, 3)}\t MAE : {round(mae, 3)}\t R2 : {round(r2, 3)}")


# PICKLE THE SCORES AND TUNED PARAMETERS

def pickle_file(x_train, x_test, y_train, y_test):
    mae_gaussian, mse_gaussian, rsq_gaussian, mae_anova, mse_anova, rsq_anova, mae_spline, mse_spline, rsq_spline = tuning_plots(x_train, x_test, y_train, y_test)

    min_sigma = min([i[1] for i in mae_gaussian.items()])
    min_anova = min([i[1] for i in mae_anova.items()])
    min_spline = min([i[1] for i in mae_spline.items()])

    sigma = None
    p = None
    d = None

    for i in mae_gaussian.items():
        if i[1] == min_sigma:
            sigma = (i[0]/10) - 1

    for i in mae_anova.items():
        if i[1] == min_anova:
            p = i[0]

    for i in mae_spline.items():
        if i[1] == min_spline:
            d = i[0]

    with open(config.GAUSSSIAN_PKL, 'wb') as f1:
        pickle.dump(mae_gaussian, f1)
        pickle.dump(mse_gaussian, f1)
        pickle.dump(rsq_gaussian, f1)
        pickle.dump(sigma, f1)

    with open(config.ANOVA_PKL, 'wb') as f2:
        pickle.dump(mae_anova, f2)
        pickle.dump(mse_anova, f2)
        pickle.dump(rsq_anova, f2)
        pickle.dump(p, f2)

    with open(config.SPLINES_PKL, 'wb') as f3:
        pickle.dump(mae_spline, f3)
        pickle.dump(mse_spline, f3)
        pickle.dump(rsq_spline, f3)
        pickle.dump(d, f3)


# LOAD SAVED PARAMETERS

def load_parameters():
    sigma = None
    p = None
    d = None
    with open(config.GAUSSSIAN_PKL, 'rb') as f1:
        pickle.load(f1)
        pickle.load(f1)
        pickle.load(f1)
        sigma = pickle.load(f1)

    with open(config.ANOVA_PKL, 'rb') as f2:
        pickle.load(f2)
        pickle.load(f2)
        pickle.load(f2)
        p = pickle.load(f2)

    with open(config.SPLINES_PKL, 'rb') as f3:
        pickle.load(f3)
        pickle.load(f3)
        pickle.load(f3)
        d = pickle.load(f3)

    return sigma, p, d


if __name__ == "__main__":
    dataset_path = config.TRAINING_DATASET
    x_train, x_test, y_train, y_test = data_preprocessing(dataset_path)

    pickle_file(x_train, x_test, y_train, y_test)       # saving the parameters

    sigma, p, d = load_parameters()
    print(sigma)
    print(p)
    print(d)

    gaussian(sigma, x_train, x_test, y_train, y_test)
    anova(p, x_train, x_test, y_train, y_test)
    splines(d, x_train, x_test, y_train, y_test)