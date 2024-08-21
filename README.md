# **KERNEL-RIDGE-REGRESSION**

## **Description**
This project is an implementation of **Kernel Ridge Regression** based on the research paper [Ridge Regression Learning Algorithm
in Dual Variables](https://www.researchgate.net/publication/221345362_Ridge_Regression_Learning_Algorithm_in_Dual_Variables). The project includes implementations of **Gaussian**, **Splines**, and **ANOVA kernel** algorithms.


This project implements the dual version of **Ridge Regression**, enabling non-linear regression by constructing a linear regression function in a high-dimensional feature space using kernel functions. The approach addresses the "curse of dimensionality" by applying kernel methods similar to those used in Support Vector Machines. The project also explores a powerful family of kernel functions constructed using the **ANOVA** **decomposition** method applied to **spline kernels** with an infinite number of nodes. The implemented algorithm combines these elements, applying the dual Ridge Regression to the ANOVA-enhanced infinite-node splines. Experimental results, based on the [**Boston Housing dataset**](https://www.kaggle.com/c/boston-housing), demonstrate the algorithm's performance relative to other regression methods.



## **Table of Content**

- [Installation](#Installation)
- [Usage](#Usage)
- [Project Structure](#Project-Structure)
- [Results](#Results)
- [Contributions](#Contributions)

## **Installation**
### 1. Clone the repository:
    git clone https://github.com/Parshantkumar2033/Kernel_ridge_regression.git

### 2. Navigate to the project directory:
    cd kernel_ridge_regression

### 3. Install the required dependencies:
    pip install -r requirements.txt


## **Usage**
To run the kernel ridge regression with the different kernels, use the following code:

![kernels](Kernel_ridge_regression/demo_pics/kernels.png)

- Update **src/config.py** accordingly.

To run the models,
```bash
python src/main.py
```

- **Gaussian Kernel**: A radial basis function kernel with parameter **`sigma`**.
        
    ![Gaussian](Kernel_ridge_regression/demo_pics/gaussian.png)
- **Splines Kernel**: A kernel based on spline functions, useful for non-linear regression tasks, parameter **`d`** defining the non-linearity order.

    ![splines](Kernel_ridge_regression/demo_pics/splines.png)
- **ANOVA Kernel**: A kernel that captures interactions between features, useful for models where feature interaction is important.

    ![anova](Kernel_ridge_regression/demo_pics/anova.png)


## **Project-Structure**

- **inputs/**
    
    - **train.csv**`Boaston Housing Dataset`

- **models/**`Contains the pickle files.`

- **notebooks/**

    - **main.ipynb** `Raw model`

- **outputs/**
- **src/**

    - **config.py**
    - **main.py** `executes the model`
    - **model_dispatcher.py** `Contains the actual Model : KernelRidgeRegression`
    - **models.py** `Kernels(gaussian, splines and anova) are defined here.`
    - **utils.py** `Contains uitlities like : train_test_split, StandardScaler, MinMaxScaler, cross-validation, ploting, etc`

- **requirements.txt**




## **Results**

The following table summarizes the performance of different kernels on the provided dataset:

| Kernel   | MSE     |    MAE  | R2-score|       
|----------|---------|---------|---------|
| Gaussian | 21.027  | 3.031   |  0.636  |
| Splines  | 223.821 | 11.608  |  0.447  |
| ANOVA    | 63.94   | 5.782   | 0.082   |

## **Contributions**

- **Parshant kumar**
- **Akash singh**



## **References**
[**Ridge Regression Learning Algorithm
in Dual Variables**](https://www.researchgate.net/publication/221345362_Ridge_Regression_Learning_Algorithm_in_Dual_Variables)
