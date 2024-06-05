#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
import ridge_regression.provided_functions as pf

sns.set_style('whitegrid')
FONT_SIZE= 16
font ={ 'size'   : FONT_SIZE}

matplotlib.rc('font', **font)
#%%
def LeastSquares(X, Y):
    ''' solves linear regression with
    L2 Loss

    X: design matrix n x d
    Y: true values n x 1

    output: weight of linear regression d x 1
    '''
    # TODO: To use inverse or use solve ?
    return np.linalg.inv(X.T @ X) @ X.T @ Y

#%%
def RidgeRegression(Phi, Y, lmbd_reg=0.):
    ''' solves linear regression with
    L2 Loss + L2 regularization

    D: design matrix n x d
    Y: true values n x 1
    lmbd_reg: weight regularization

    output: weight of linear regression d x 1
    '''
    d = Phi.shape[1]
    I_d = np.eye(d)
    # TODO: To use inverse or use solve?
    # Returns weight vector of size d x 1
    return np.linalg.inv(Phi.T @ Phi + lmbd_reg * I_d) @ Phi.T @ Y

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
   Phi = Basis(X, k)
   if type == 'ridge':
    return  RidgeRegression(Phi, Y, lmbd_reg)
   elif type == 'l1':
        return pf.L1LossRegression(Phi, Y, lmbd_reg)
   raise ValueError('Unknown type: %s'%type)

def plot_comparison_l1_vs_ridge(data):
    for k in [1,2, 3,5, 10, 15,20]:
        plt.figure(figsize=(7, 7))
        plt.scatter(data['Xtrain'], data['Ytrain'],s=1,cmap='viridis',label='Training Data')
        for type in ['ridge', 'l1']:
            w = learned_weights(data['Xtrain'], data['Ytrain'],k=k, lmbd_reg=30, type=type)
            x = np.linspace(0, 1, 1000)
            phi = Basis(x, k)
            plt.plot(x, phi @ w, label='%s Loss Regression k=%d'%(type.upper(), k))
            plt.legend()
        plt.savefig('../figures/l1_vs_ridge_regression_k_%d.png'%( k))

def plot_l1_for_k(data, lmdb_reg=30):
    type = 'l1'
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(data['Xtrain'], data['Ytrain'],s=1,cmap='viridis',label='Training Data')
    for k in [1,2, 3,5, 10, 15,20]:
        w = learned_weights(data['Xtrain'], data['Ytrain'],k=k, lmbd_reg=lmdb_reg, type=type)
        x = np.linspace(0, 1, 1000)
        phi = Basis(x, k)
        plt.plot(x, phi @ w, label='%s Loss Regression k=%d'%(type.upper(), k))
        plt.legend()
    plt.savefig('../figures/l1_regression_all_k.png')
    return fig

def plot_l1_loss_for_k(data):
    type = 'l1'
    fig = plt.figure(figsize=(10, 10))
    K_VALS = [1,2, 3,5, 10, 15,20]
    for lmdb_reg in [0, 30]:
        for partition in ['train', 'test']:
            l1_losses = np.zeros((len(K_VALS), 1))
            for k_idx, k in enumerate(K_VALS):
                w = learned_weights(data['Xtrain'], data['Ytrain'],k=k, lmbd_reg=lmdb_reg, type=type)
                phi = Basis(data['X%s'%partition], k)
                f_X = phi @ w
                Y = data['Y%s' % partition] 
                l1_loss_l2_reg = np.abs(f_X - Y).sum() + lmdb_reg * (w ** 2).sum()
                l1_losses[k_idx] = l1_loss_l2_reg
            plt.xlabel('k', fontsize=FONT_SIZE)    
            plt.ylabel('Loss', fontsize=FONT_SIZE)
            plt.plot(K_VALS,l1_losses, label='Losses : %s and Lambda: %d' % (partition.capitalize(), lmdb_reg))
    plt.legend()
    plt.savefig('../figures/l1_losses_for_k.png' )
    return fig
#%%
if __name__ == '__main__':
    data = np.load('../data/onedim_data.npy', allow_pickle=True).item()
    plot_comparison_l1_vs_ridge(data)
    plot_l1_for_k (data)
    plot_l1_loss_for_k(data)
#%%
