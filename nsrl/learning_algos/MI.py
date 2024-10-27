# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:03:51 2020

@author: Shujian Yu, Xi Yu

Calculate Mutual inforamtion based on Matrix-based Entropy Functional
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_gram_mat(x, sigma):
    """calculate gram matrix for variables x
        Args:
        x: random variable with two dimensional (N,d).
        sigma: kernel size of x (Gaussain kernel)
    Returns:
        Gram matrix (N,N)
    """
    x = x.view(x.shape[0],-1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    dist= -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()
    return torch.exp(-dist /sigma)

def renyi_entropy(x, sigma, alpha):
    
    """calculate entropy for single variables x (Eq.(9) in paper)
        Args:
        x: random variable with two dimensional (N,d).
        sigma: kernel size of x (Gaussain kernel)
        alpha:  alpha value of renyi entropy
    Returns:
        renyi alpha entropy of x. 
    """

    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.linalg.eigvalsh(k)) #new version eigen value
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y,alpha):
    
    """calculate joint entropy for random variable x and y (Eq.(10) in paper)
        Args:
        x: random variable with two dimensional (N,d).
        y: random variable with two dimensional (N,d).
        s_x: kernel size of x
        s_y: kernel size of y
        alpha:  alpha value of renyi entropy
    Returns:
        joint entropy of x and y. 
    """
    
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigvalsh(k)) #new version eigen value
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy

def calculate_MI(x,y,s_x,s_y,alpha,normalize):
    
    """calculate Mutual information between random variables x and y

    Args:
        x: random variable with two dimensional (N,d).
        y: random variable with two dimensional (N,d).
        s_x: kernel size of x
        s_y: kernel size of y
        normalize: bool True or False, noramlize value between (0,1)
    Returns:
        Mutual information between x and y (scale)

    """
    Hx = renyi_entropy(x,sigma=s_x,alpha=alpha)
    Hy = renyi_entropy(y,sigma=s_y,alpha=alpha)
    Hxy = joint_entropy(x,y,s_x,s_y,alpha)
    if normalize:
        Ixy = Hx+Hy-Hxy
        Ixy = Ixy/(torch.max(Hx,Hy))
    else:
        Ixy = Hx+Hy-Hxy
    return Ixy

if __name__ == '__main__':
    pass