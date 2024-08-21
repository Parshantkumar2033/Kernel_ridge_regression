from itertools import combinations
from math import comb, pow
import numpy as np



class kernel:
    def __init__(self):
        pass
        
    def anova_kernel(self, X, Y, p, upto_pth = True):
        '''
        One-dimensional kernel k, the ANOVA kernels are defined as follows
        '''
        def cal_k(x, y, ps):
            if ps == 0:
                return 1
            sum_k = 0
            for combination in combinations(range(len(x)), ps):
                product = 1
                for i in combination:
                    product *= x[i] * y[i]
                sum_k += product
            return sum_k
    
        '''
        n : length of the vector x
        Using, K(s) (x, y) = summ(i = 1 to n) (k(x(i), y(i)))**s
        '''
        def cal_ks(x, y, s):
            sum_i_to_n = 0
            for i in range(len(x)):
                product = (x[i] * y[i])**s
                sum_i_to_n += product
            return sum_i_to_n
    
        '''
        calculating the pth-kernel using Kp(x, y)
        only for pth order
        '''
        def kernel_p(x, y, p):
            sum_at_p = 0
            for s in range(1, p+1):
                sum_at_p += ((-1)**(s+1))*(cal_k(x, y, p-s))*(cal_ks(x, y, s))
            return sum_at_p/p  
            
        '''
        summation upto p order, including all the lower orders
        Using, K(x, y) = summ(i = 1 to p) Ki(x, y)
        Function, "kernel_p(x, y, p)" will be called for (i = 1 to p)
        '''
        def kernel_upto_p(x, y, P):
            sum_till_p = 0
            for p in range(1, P+1):
                sum_till_p += kernel_p(x, y, p)
            return sum_till_p
    
        #computing Kernel Matrix
        if upto_pth:
            K = np.zeros((X.shape[0], Y.shape[0]))     
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    K[i, j] = kernel_upto_p(X[i], Y[j], p)
        else:
            K = np.zeros((X.shape[0], Y.shape[0]))     
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    K[i, j] = kernel_p(X[i], Y[j], p)
            
        return K
    
    def spline_kernel(self, X, Y, d):
        '''
        kernel(x, y, d) : 
        x and y are vectors
        d : the parameter which defines the type of splines 
            if d = 1, linear splines
        calculates the summation from r = 0 to r = d
        Using, kd(x(i), y(i)) = first + second
        '''
        def kernels(x, y, d):
            sum = 0
            for r in range(d+1):
                seq = 2*d - r + 1
                first = (comb(d, r) / seq) * ((min(x, y)) ** seq) * ((abs(x - y)) ** r)
                second = ((x) ** r) * ((y) ** r)
                sum += first + second
            return sum
    
        '''
        The function : kernel(x, y, d) is for two elements of vectors (x(i), y(i)).
        So, in order to make is possible for N-dimensional case, 
        we multiply all the kd's 
        '''
        def splines(x, y, d):
            count = 0
            # product = 1
            sum = 0
            for i in range(len(x)):
                for j in range(len(y)):
                    count += 1
                    # product *= kernels(x[i], y[j], d)
                    sum += kernels(x[i], y[j], d)
            # if product < 0:
            #     product = abs(product) ** (1/count)
            #     return -product
            # return product ** (1/count*2)
            return sum/count
                    
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                K[i, j] = splines(X[i], Y[j], d)
        return K
    
    def gaussian_kernel(self, X, Y, sigma = 1.0):
        X = np.array(X)
        Y = np.array(Y)
    
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
    
        K = np.zeros((X.shape[0], Y.shape[0]))
    
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                squared_distance = np.sum((X[i] - Y[j]) ** 2)
                K[i, j] = np.exp(-squared_distance / (2 * sigma ** 2))
                # print(f"K[{i}, {j}] = {K[i, j]}")
        
        return K