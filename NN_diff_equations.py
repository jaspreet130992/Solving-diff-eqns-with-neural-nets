# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:42:11 2018

@author: Jaspreet Singh
"""
# This is an elementary demonstration of how a Neural Net can be used to solve the differential equations.
# This is in priliminary stages right now and I am currently working on it.
# This neural net solve equations of the type dy/dx=f(x)
# Next step is to include equations such as (d^n/d x^n)y=f(x), (d^n/d x^n)y=f(y,x). The code needs to be modified for these. 
# For any questions, shoot me an email @ sinjas@seas.upenn.edu

import numpy as np
import matplotlib.pyplot as plt

def clumsy_plot(xv,yv,xlab,ylab,fignum,arg1,legend_label,legend_loc):
    plt.figure(fignum)
    plt.plot(xv,yv,arg1,linewidth=3.0,markersize=5,label=legend_label)
    plt.xlabel(xlab,fontsize=30)
    plt.ylabel(ylab,fontsize=30)
    plt.legend(loc=legend_loc,fontsize=15)

np.random.seed(3) 

class NN:
    
    def __init__(self, no_of_inputs,str):
        str.astype(int)
        no_of_layers=len(str)
        no_of_weights=no_of_inputs*str[0]+str[0]
        for i in range(1,no_of_layers):
            no_of_weights+=str[i]*str[i-1]+str[i]
        self.weights=np.random.rand(no_of_weights,1)
        self.str=str
        self.number_of_weights=no_of_weights
        self.number_of_inputs=no_of_inputs
        self.number_of_layers=no_of_layers
        
    def extract_weights_bias(self,layer):
        if layer>len(self.str):
            sys.exit("No_of_layer>length of str")
        if layer==1:
            ind1=self.str[0]*self.number_of_inputs # index of the weights
            ind2=ind1+self.str[0]   # index of the bias
            return (self.weights[0:ind1].reshape(self.str[0],self.number_of_inputs)), (self.weights[ind1:ind2])
        if layer>1:
            ind=self.number_of_inputs*self.str[0]+self.str[0]
            for i in range(1,layer-1):
                ind+=self.str[i]*self.str[i-1]+self.str[i]
            ind1= ind # beginning index of weights of layer 
            ind2= ind+self.str[layer-1]*self.str[layer-2]  # ending index of the weights and beginning of the bias
            ind3= ind+self.str[layer-1]*self.str[layer-2]+self.str[layer-1]   # ending indec of the bias
            return self.weights[ind:ind2].reshape(self.str[layer-1],self.str[layer-2]), self.weights[ind2:ind3 ]
    
    def layer_mult(self,weight_matrix,inp_column,bias_column):
        return np.sin(np.matmul(weight_matrix,inp_column)+bias_column)
    
    def evaluate_x(self,x): 
       weights1,bias1=self.extract_weights_bias(1)
       out=self.layer_mult(weights1,x,bias1)
       for i in range(2,self.number_of_layers+1):
           weights,bias=self.extract_weights_bias(i)
           out=self.layer_mult(weights,out,bias)
       return out

def dfdx(x,nn):         # Only for one dimension, you need to write other routines for 2-D differentiation 
    h=1E-4
    return (1/h)*(nn.evaluate_x(np.add(x,[[h/2]]))-nn.evaluate_x(np.add(x,[[-h/2]])))

def df2dx2(x,nn):         # Only for one dimension, you need to write other routines for 2-D differentiation 
    h=1E-4
    return (1/h**2)*(nn.evaluate_x(np.add(x,[[h]]))+nn.evaluate_x(np.add(x,[[-h]]))-2*nn.evaluate_x(x))


def solve_diff(weight_array):
    #print('nn in func',nn1.evaluate_x([[1.0]]))
    if len(weight_array)!=nn1.number_of_weights:
        print('Dimensions do not match')
        return 0            
    nn1.weights=weight_array
    #print(np.size(nn1.weights))
    xv=np.linspace(0,3.14,20)
    loss_sum=0
    for i in range(len(xv)):
        loss_sum+=(np.abs( df2dx2([[xv[i]]],nn1) + np.sin(xv[i])))
    return loss_sum[0,0]

def grad_loss_weights(loss_func,weights):
    gradient=np.zeros_like(weights)
    delta=1E-4
    for i in range(len(weights)):
        aux=np.zeros_like(weights)
        aux[i]=delta
        gradient[i]=(1/delta)*(loss_func(np.add(weights,aux))-loss_func(weights))
    return gradient    

    #return loss_sum 
    #print('dfdx(x=1)=',dfdx([[1]],nn1))   



nn1="global"
nn1 = NN(1,np.array([5,2,1])) # final layer has to be 1
nn2 = NN(1,np.array([5,2,1]))
iter=200
loss_vec=np.zeros_like(range(iter))
for i in range(iter):
    print('iter',i)
    nn1.weights -= grad_loss_weights(solve_diff,nn1.weights)*0.01 
    curr_loss=solve_diff(nn1.weights)
    if curr_loss<=1.0:
        print('Iteration broken at', i)
        break
    loss_vec[i]=curr_loss

xv=np.linspace(0,1.0,10)
ytrue=np.zeros_like(xv)
ypredict=np.zeros((len(xv),1))
ypredict_init=np.zeros((len(xv),1))

f_true=lambda x: np.sin(x)
init_cond=f_true(xv[0])-nn1.evaluate_x([[xv[0]]])
for i in range(len(xv)):
    ytrue[i]=f_true(xv[i])
    ypredict[i,0]=nn1.evaluate_x([[xv[i]]])+init_cond
    ypredict_init[i,0]=nn2.evaluate_x([[xv[i]]])+init_cond   
#    
clumsy_plot(xv,ypredict,'x','y',1,'-bs','Predicted','upper right')
clumsy_plot(xv,ypredict_init,'x','y',1,'-gs','Initial prediction','upper right')
clumsy_plot(xv,ytrue,'x','y',1,'-r^','true','upper right')
clumsy_plot(range(iter),loss_vec,'iter','loss',2,'-r^','loss','upper right')



