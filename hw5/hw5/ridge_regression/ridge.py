#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
import ridge_regression.provided_functions as pf

sns.set_style('whitegrid')

#%%
def LeastSquares(X, Y):
    ''' solves linear regression with
    L2 Loss

    X: deisgn matrix n x d
    Y: true values n x 1

    output: weight of linear regression d x 1
    '''
    return np.linalg.inv(X.T @ X) @ X.T @ Y

#%%
def RidgeRegression(D, Y, lmbd_reg=0.):
    ''' solves linear regression with
    L2 Loss + L2 regularization

    D: design matrix n x d
    Y: true values n x 1
    lmbd_reg: weight regularization

    output: weight of linear regression d x 1
    '''
    return np.linalg.inv(D.T @ D + lmbd_reg * np.eye(D.shape[1])) @ D.T @ Y

#%%
def Basis(X, k):
    ''' 
    Compute the fourier basis of X.
    '''
    print("Shape input matrix:",X.shape) 
    # X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_{1000} \end{bmatrix}
    
    phi = np.zeros((X.shape[0], 2 * k + 1)) # (1000, 21) #k = 10
    print("Shape Design Matrix:", phi.shape)
    # phi = \begin{bmatrix} 
    #           1 & \cos(2\pi x_1) & \sin(2\pi x_1) & \cos(4\pi x_1) & \sin(4\pi x_1) & \cdots & \cos(20\pi x_1) & \sin(20\pi x_1) 
    #           1 & \cos(2\pi x_2) & \sin(2\pi x_2) & \cos(4\pi x_2) & \sin(4\pi x_2) & \cdots & \cos(20\pi x_2) & \sin(20\pi x_2)
    #           \vdots
    #           1 & \cos(2\pi x_{1000}) & \sin(2\pi x_{1000}) & \cos(4\pi x_{1000}) & \sin(4\pi x_{1000}) & \cdots & \cos(20\pi x_{1000}) & \sin(20\pi x_{1000})
    # \end{bmatrix} \in \mathbb{R}^{1000 \times 21}
    for l in range(1, k + 1):
        phi[:, 2 * l - 1] = np.cos(2 * np.pi * l * X).flatten()
        phi[:, 2 * l] = np.sin(2 * np.pi * l * X).flatten() 
    phi[:, 0] = 1 
    # print("Phi:",phi)
    return phi

#%%
def FourierBasisNormalized(X, k):
    pass
#%%
def plot_data(X, Y):
    ''' plot data points
    '''
    plt.figure(figsize=(15, 10))
    plt.scatter(X, Y)
    plt.show()
    return

#%%
def run_plot(data, train=False, test=False):
    ''' run the plot
    '''
    if train:
        plot_data(data['Xtrain'], data['Ytrain'])
    elif test:
        plot_data(data['Xtest'], data['Ytest'])
        
#%%
def learned_weights(X, Y, k, lmbd_reg=0., type='ridge'):
   ''' learned_weights: learn the weights of the model, from training data with 
   k basis functions, and regularization parameter lmbd_reg.'''
   D = Basis(X, k)
   if type == 'ridge':
    return  RidgeRegression(D, Y, lmbd_reg)
   elif type == 'L1':
        return pf.L1LossRegression(D, Y, lmbd_reg)
   raise ValueError('Unknown type: %s'%type)

#%%
if __name__ == '__main__':
    data = np.load('../data/onedim_data.npy', allow_pickle=True).item()
    run_plot(data,train=True)
    run_plot(data,test=True)
    for type in ['ridge', 'L1']:
        plt.figure(figsize=(7, 7))
        plt.scatter(data['Xtrain'], data['Ytrain'],s=1,cmap='viridis',label='Training Data')
        for k in [1,2, 3,5, 10, 15,20]:
            w = learned_weights(data['Xtrain'], data['Ytrain'],k=k, lmbd_reg=30, type=type)
            x = np.linspace(0, 1, 1000)
            phi = Basis(x, k)
            plt.plot(x, phi @ w, label='%s Loss Regression k=%d'%(type, k))
            plt.legend()

# %%
