# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:47:47 2026

@author: 16847
"""

import tensorflow.compat.v1 as tf
import numpy as np
from scipy.special import expit
import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from scipy.stats import truncnorm
import random
import time
import pandas as pd
tf.disable_v2_behavior()

num_rep = 100
sample_size = 500
train_size = math.ceil(0.8 * sample_size)
validation_size = math.ceil(0.2 * sample_size)
test_size = 500
total_size = test_size + sample_size
MonteCarlo_size = 10000
cov_dimension = 20
K = 4
LearningR = 0.0001
reference_dimension = 3
parameters = {'h_dim_g':32, 'h_dim_d':32, 'batch_size': 256, 'iteration':10000, 'alpha':1} 
transform = transforms.ToTensor()

def generate_data(rep_index): 
    print(f"Replication {rep_index}")
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)
    
    W_1 = truncnorm.rvs(-3, 3, loc = 0, scale = 1, size = (total_size,2))  # generate x1, x2 from normal distribution
    W_2 = np.random.uniform(-1, 1, size=(total_size, 2 ))  # generate x3, x4 from unif distribution
    WW = np.hstack((W_1, W_2))
    
    XX = np.zeros((total_size,cov_dimension))
    XX[:, [0, 5, 10, 15]] = WW
    XX[:, [1, 6, 11, 16]] = 2 * WW
    XX[:, [2, 7, 12, 17]] = -1 * WW
    XX[:, [3, 8, 13, 18]] = WW**3
    XX[:, [4, 9, 14, 19]] = np.sin(WW)

    PA = np.hstack(( np.abs(WW[:,0]).reshape(-1, 1), np.abs(2*WW[:,1]).reshape(-1, 1), 
                     np.abs(3*WW[:,2]).reshape(-1, 1), np.abs(4*WW[:,3]).reshape(-1, 1) ))
    PA = PA / PA.sum(axis=1).reshape(-1, 1) 
    choices = [1, 2, 3, 4]
    A = np.zeros(total_size, dtype=int).reshape(-1, 1) 
    for i in range(total_size):
        A[i] = np.random.choice(choices, p = PA[i])
    indicator_A = np.zeros((total_size, K-1))
    indicator_A[ A[:, 0] > 1, (A[A[:, 0] > 1] - 2).squeeze()] = 1
    
    mu = np.exp(WW).mean(axis=1).reshape(-1, 1)
    gamma_1 = 2 * np.sum(WW, axis=1).reshape(-1, 1)
    gamma_2 = np.sum( np.sqrt(np.abs(WW)) , axis=1) .reshape(-1, 1)/2
    gamma_3 = np.sum(WW**2, axis=1).reshape(-1, 1)/2
    gamma_4 = np.sum(np.cos(WW), axis=1).reshape(-1, 1)/2
    epsilon_sd = (np.sqrt( np.sum(WW**2,axis=1) )/4).reshape(-1, 1)
    epsilon = np.zeros(total_size).reshape(-1, 1)
    for i in range(total_size):
        epsilon[i] = truncnorm(-3, 3, 0, epsilon_sd[i]).rvs(1)
        
    R_1 = mu + gamma_1 + epsilon
    R_2 = mu + gamma_2 + epsilon
    R_3 = mu + gamma_3 + epsilon
    R_4 = mu + gamma_4 + epsilon
    R = np.zeros_like(A,dtype="float64") 
    R[A == 1] = R_1[A == 1]
    R[A == 2] = R_2[A == 2]
    R[A == 3] = R_3[A == 3]
    R[A == 4] = R_4[A == 4]
    
    train_x = XX[:train_size] 
    train_t = indicator_A[:train_size]
    train_y = R[:train_size]
    test_x = XX[-test_size:, :]
    test_w = WW[-test_size:, :]
    train_z = np.random.uniform(-1, 1, size=(train_x.shape[0],reference_dimension) ) 
    test_z = np.random.uniform(-1, 1, size=(test_size*MonteCarlo_size,reference_dimension) )
    return  {'train_x':train_x, 'train_t':train_t, 'train_y':train_y, 'train_z':train_z, 'test_x':test_x, 'test_w':test_w, 'test_z':test_z} 

def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)

def batch_generator(x, t, y, z, size):
  batch_idx = np.random.randint(0, x.shape[0], size)
  X_mb = x[batch_idx, :]
  Z_mb = z[batch_idx, :]
  T_mb = t[batch_idx]
  Y_mb = np.reshape(y[batch_idx], [size,1])   
  return X_mb, T_mb, Y_mb, Z_mb

def ganite (train_x, train_t, train_y, train_z, test_x, test_z, parameters):
  # Parameters 
  h_dim_g = parameters['h_dim_g']
  h_dim_d = parameters['h_dim_d']
  batch_size = parameters['batch_size']
  iterations = parameters['iteration']
  alpha = parameters['alpha']
  no, dim = train_x.shape
  # Reset graph
  tf.reset_default_graph()
  ## 1. Placeholder
  X = tf.placeholder(tf.float32, shape = [None, dim])
  T = tf.placeholder(tf.float32, shape = [None, K-1])
  Y = tf.placeholder(tf.float32, shape = [None, 1])
  Z = tf.placeholder(tf.float32, shape = [None, reference_dimension])
  ## 2. Variables
  # 2.1 Generator
  G_W1 = tf.Variable(xavier_init([(dim + K + reference_dimension), h_dim_g])) # Inputs: X + Treatment + Factual outcome + Z
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim_g]))
  # Multi-task outputs for increasing the flexibility of the generator 
  G_W31 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  G_b31 = tf.Variable(tf.zeros(shape = [h_dim_g]))
  G_W32 = tf.Variable(xavier_init([h_dim_g, 1]))
  G_b32 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (0,0,0)
  G_W41 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  G_b41 = tf.Variable(tf.zeros(shape = [h_dim_g])) 
  G_W42 = tf.Variable(xavier_init([h_dim_g, 1]))
  G_b42 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (1,0,0)
  G_W51 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  G_b51 = tf.Variable(tf.zeros(shape = [h_dim_g])) 
  G_W52 = tf.Variable(xavier_init([h_dim_g, 1]))
  G_b52 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (0,1,0)
  G_W61 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  G_b61 = tf.Variable(tf.zeros(shape = [h_dim_g])) 
  G_W62 = tf.Variable(xavier_init([h_dim_g, 1]))
  G_b62 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (0,0,1)
  # Generator variables
  theta_G = [G_W1, G_W31, G_W32, G_W41, G_W42, G_W51, G_W52, G_W61, G_W62, G_b1, G_b31, G_b32, G_b41, G_b42, G_b51, G_b52, G_b61, G_b62]
  # 2.2 Discriminator
  D_W1 = tf.Variable(xavier_init([(dim+K), h_dim_d])) # Inputs: X + Factual outcomes + Estimated counterfactual outcomes
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim_d]))
  D_W21 = tf.Variable(xavier_init([h_dim_d, h_dim_d]))
  D_b21 = tf.Variable(tf.zeros(shape = [h_dim_d]))
  D_W22 = tf.Variable(xavier_init([h_dim_d, 1]))
  D_b22 = tf.Variable(tf.zeros(shape = [1]))
  D_W31 = tf.Variable(xavier_init([h_dim_d, h_dim_d]))
  D_b31 = tf.Variable(tf.zeros(shape = [h_dim_d]))
  D_W32 = tf.Variable(xavier_init([h_dim_d, 1]))
  D_b32 = tf.Variable(tf.zeros(shape = [1]))
  D_W41 = tf.Variable(xavier_init([h_dim_d, h_dim_d]))
  D_b41 = tf.Variable(tf.zeros(shape = [h_dim_d]))
  D_W42 = tf.Variable(xavier_init([h_dim_d, 1]))
  D_b42 = tf.Variable(tf.zeros(shape = [1]))
  # Discriminator variables
  theta_D = [D_W1, D_W21, D_W22, D_W31, D_W32, D_W41, D_W42, D_b1, D_b21, D_b22, D_b31, D_b32, D_b41, D_b42]
  # 2.3 Inference network
  I_W1 = tf.Variable(xavier_init([(dim+reference_dimension), h_dim_g])) # Inputs: X + Z
  I_b1 = tf.Variable(tf.zeros(shape = [h_dim_g]))
  # Multi-task outputs for increasing the flexibility of the inference network
  I_W31 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  I_b31 = tf.Variable(tf.zeros(shape = [h_dim_g]))
  I_W32 = tf.Variable(xavier_init([h_dim_g, 1]))
  I_b32 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (0,0,0)
  I_W41 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  I_b41 = tf.Variable(tf.zeros(shape = [h_dim_g]))
  I_W42 = tf.Variable(xavier_init([h_dim_g, 1]))
  I_b42 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (1,0,0)
  I_W51 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  I_b51 = tf.Variable(tf.zeros(shape = [h_dim_g]))
  I_W52 = tf.Variable(xavier_init([h_dim_g, 1]))
  I_b52 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (0,1,0)
  I_W61 = tf.Variable(xavier_init([h_dim_g, h_dim_g]))
  I_b61 = tf.Variable(tf.zeros(shape = [h_dim_g]))
  I_W62 = tf.Variable(xavier_init([h_dim_g, 1]))
  I_b62 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = (0,0,1)
  # Inference network variables
  theta_I = [I_W1, I_W31, I_W32, I_W41, I_W42, I_W51, I_W52, I_W61, I_W62, I_b1, I_b31, I_b32, I_b41, I_b42, I_b51, I_b52, I_b61, I_b62]
  ## 3. Definitions of generator, discriminator and inference networks
  # 3.1 Generator
  def generator(x, t, y, z):
    # Concatenate feature, treatments, and observed labels as input
    inputs = tf.concat(axis = 1, values = [x,t,y,z])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    # Estimated outcome if t = (0,0,0)
    G_h31 = tf.nn.relu(tf.matmul(G_h1, G_W31) + G_b31)
    G_logit1 = tf.matmul(G_h31, G_W32) + G_b32
    # Estimated outcome if t = (1,0,0)
    G_h41 = tf.nn.relu(tf.matmul(G_h1, G_W41) + G_b41)
    G_logit2 = tf.matmul(G_h41, G_W42) + G_b42
    # Estimated outcome if t = (0,1,0)
    G_h51 = tf.nn.relu(tf.matmul(G_h1, G_W51) + G_b51)
    G_logit3 = tf.matmul(G_h51, G_W52) + G_b52
    # Estimated outcome if t = (0,0,1)
    G_h61 = tf.nn.relu(tf.matmul(G_h1, G_W61) + G_b61)
    G_logit4 = tf.matmul(G_h61, G_W62) + G_b62
    G_logit = tf.concat(axis = 1, values = [G_logit1, G_logit2, G_logit3, G_logit4])
    return G_logit
  # 3.2. Discriminator
  def discriminator(x, t, y, hat_y):
    t1 = tf.reshape(t[:, 0], [-1, 1])   # (1,0,0)
    t2 = tf.reshape(t[:, 1], [-1, 1])   # (0,1,0)
    t3 = tf.reshape(t[:, 2], [-1, 1])   # (0,0,1)
    t0 = 1.0 - t1 - t2 - t3             # (0,0,0) baseline
    # if factual is (0,0,0): input0=y else input0=hat_y[:,0]
    input0 = t0 * y + (1.0 - t0) * tf.reshape(hat_y[:, 0], [-1, 1])
    # if factual is (1,0,0): input1=y else input1=hat_y[:,1]
    input1 = t1 * y + (1.0 - t1) * tf.reshape(hat_y[:, 1], [-1, 1])
    # if factual is (0,1,0): input2=y else input2=hat_y[:,2]
    input2 = t2 * y + (1.0 - t2) * tf.reshape(hat_y[:, 2], [-1, 1])
    # if factual is (0,0,1): input3=y else input3=hat_y[:,3]
    input3 = t3 * y + (1.0 - t3) * tf.reshape(hat_y[:, 3], [-1, 1])
    # (n, dim+3)
    inputs = tf.concat(axis=1, values=[x,input0,input1,input2,input3]) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    # Estimated probability for t = (1,0,0)
    D_h21 = tf.nn.relu(tf.matmul(D_h1, D_W21) + D_b21)
    D_logit1 = tf.matmul(D_h21, D_W22) + D_b22
    # Estimated probability for t = (0,1,0)
    D_h31 = tf.nn.relu(tf.matmul(D_h1, D_W31) + D_b31)
    D_logit2 = tf.matmul(D_h31, D_W32) + D_b32
    # Estimated probability for t = (0,0,1)
    D_h41 = tf.nn.relu(tf.matmul(D_h1, D_W41) + D_b41)
    D_logit3 = tf.matmul(D_h41, D_W42) + D_b42
    D_logit = tf.concat(axis = 1, values = [D_logit1, D_logit2, D_logit3])
    return D_logit
  # 3.3. Inference Nets
  def inference(x, z):
    inputs = tf.concat(axis = 1, values = [x,z])
    I_h1 = tf.nn.relu(tf.matmul(inputs, I_W1) + I_b1)
    # Estimated outcome if t = (0,0,0)
    I_h31 = tf.nn.relu(tf.matmul(I_h1, I_W31) + I_b31)
    I_logit1 = tf.matmul(I_h31, I_W32) + I_b32
    # Estimated outcome if t = (1,0,0)
    I_h41 = tf.nn.relu(tf.matmul(I_h1, I_W41) + I_b41)
    I_logit2 = tf.matmul(I_h41, I_W42) + I_b42
    # Estimated outcome if t = (0,1,0)
    I_h51 = tf.nn.relu(tf.matmul(I_h1, I_W51) + I_b51)
    I_logit3 = tf.matmul(I_h51, I_W52) + I_b52
    # Estimated outcome if t = (0,0,1)
    I_h61 = tf.nn.relu(tf.matmul(I_h1, I_W61) + I_b61)
    I_logit4 = tf.matmul(I_h61, I_W62) + I_b62
    I_logit = tf.concat(axis = 1, values = [I_logit1, I_logit2, I_logit3, I_logit4])
    return I_logit
  ## Structure
  # 1. Generator
  Y_tilde_logit = generator(X, T, Y, Z)
  # 2. Discriminator
  D_logit = discriminator(X,T,Y,Y_tilde_logit)
  # 3. Inference network
  Y_hat_logit = inference(X, Z)
  ## Loss functions
  # 1. Discriminator loss
  T_full = tf.concat([1 - tf.reshape(T[:,0],[-1,1]) - tf.reshape(T[:,1],[-1,1]) - tf.reshape(T[:,2],[-1,1]), 
                      tf.reshape(T[:,0],[-1,1]), tf.reshape(T[:,1],[-1,1]), tf.reshape(T[:,2],[-1,1])], axis=1) 
  D_logit_full = tf.concat([tf.zeros_like(tf.reshape(D_logit[:,0],[-1,1])), tf.reshape(D_logit[:,0],[-1,1]), 
                            tf.reshape(D_logit[:,1],[-1,1]), tf.reshape(D_logit[:,2],[-1,1])], axis=1) 
  D_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=T_full, logits=D_logit_full))
  # 2. Generator loss
  G_loss_GAN = - D_loss 
  Y_tilde_fac = tf.reduce_sum(T_full * Y_tilde_logit, axis=1, keepdims=True)
  G_loss_Factual = tf.reduce_mean(tf.square(Y - Y_tilde_fac))
  # G_loss_Factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( labels = Y, logits = Y_tilde_fac)) 
  G_loss = G_loss_Factual + alpha * G_loss_GAN
  # 3. Inference loss
  label0 = tf.reshape(T_full[:,0],[-1,1]) * Y + (1 - tf.reshape(T_full[:,0],[-1,1])) * tf.reshape(Y_tilde_logit[:,0],[-1,1])
  label1 = tf.reshape(T_full[:,1],[-1,1]) * Y + (1 - tf.reshape(T_full[:,1],[-1,1])) * tf.reshape(Y_tilde_logit[:,1],[-1,1])
  label2 = tf.reshape(T_full[:,2],[-1,1]) * Y + (1 - tf.reshape(T_full[:,2],[-1,1])) * tf.reshape(Y_tilde_logit[:,2],[-1,1])
  label3 = tf.reshape(T_full[:,3],[-1,1]) * Y + (1 - tf.reshape(T_full[:,3],[-1,1])) * tf.reshape(Y_tilde_logit[:,3],[-1,1])
  I_loss  = tf.reduce_mean(tf.square(label0 - tf.reshape(Y_hat_logit[:,0],[-1,1]))) \
            + tf.reduce_mean(tf.square(label1 - tf.reshape(Y_hat_logit[:,1],[-1,1]))) \
            + tf.reduce_mean(tf.square(label2 - tf.reshape(Y_hat_logit[:,2],[-1,1]))) \
            + tf.reduce_mean(tf.square(label3 - tf.reshape(Y_hat_logit[:,3],[-1,1])))
  ## Solver
  G_solver = tf.train.RMSPropOptimizer(LearningR).minimize(G_loss, var_list=theta_G)
  D_solver = tf.train.RMSPropOptimizer(LearningR).minimize(D_loss, var_list=theta_D)
  I_solver = tf.train.RMSPropOptimizer(LearningR).minimize(I_loss, var_list=theta_I)
  ## GANITE training
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  print('Start training Generator and Discriminator')
  time1 = time.time()
  # 1. Train Generator and Discriminator
  for it in range(iterations):
    for _ in range(2):
      # Discriminator training
      X_mb, T_mb, Y_mb, Z_mb = batch_generator(train_x, train_t, train_y, train_z, batch_size)      
      _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb, Z: Z_mb})
    # Generator traininig
    X_mb, T_mb, Y_mb, Z_mb = batch_generator(train_x, train_t, train_y, train_z, batch_size)           
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb, Z: Z_mb})
    # Check point
    if it % 1000 == 0:
      print('Iteration: ' + str(it) + '/' + str(iterations) + ', D loss: ' + \
            str(np.round(D_loss_curr, 4)) + ', G loss: ' + str(np.round(G_loss_curr, 4)))
  print('Start training Inference network')
  # 2. Train Inference network
  for it in range(iterations):
    X_mb, T_mb, Y_mb, Z_mb = batch_generator(train_x, train_t, train_y, train_z, batch_size) 
    _, I_loss_curr = sess.run([I_solver, I_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb, Z: Z_mb})    
    # Check point
    if it % 1000 == 0:      
      print('Iteration: ' + str(it) + '/' + str(iterations) + 
            ', I loss: ' + str(np.round(I_loss_curr, 4)))   
  ## Generate the potential outcomes
  time2 = time.time()
  test_y_hat = np.zeros( (MonteCarlo_size*test_size, K) )
  for i in range(test_size): 
     start = i * MonteCarlo_size
     end = (i + 1) * MonteCarlo_size 
     X_i = np.repeat(test_x[i, :].reshape(1, -1), MonteCarlo_size, axis=0)
     test_y_hat[start:end,:] = sess.run(Y_hat_logit, feed_dict = {X: X_i, Z: test_z[start:end,:]})
  time3 = time.time()
  Ttime = np.array([time2 - time1, time3 - time2])
  return  {'test_y_hat':test_y_hat, 'Ttime':Ttime} 

def true_value(Mon_size):
    W_1 = truncnorm.rvs(-3, 3, loc = 0, scale = 1, size = (Mon_size,2))  # generate x1, x2 from normal distribution
    W_2 = np.random.uniform(-1, 1, size=(Mon_size, 2 ))  # generate x3, x4 from unif distribution
    WW = np.hstack((W_1, W_2))
    E_sigma = np.mean( np.sqrt(np.sum(WW**2, axis=1)) /4)
    mu = np.exp(WW).mean(axis=1).reshape(-1, 1)
    gamma_1 = 2*np.sum(WW, axis=1).reshape(-1, 1) 
    gamma_2 = np.sum( np.sqrt(np.abs(WW)) , axis=1) .reshape(-1, 1)/2
    gamma_3 = np.sum(WW**2, axis=1).reshape(-1, 1)/2
    gamma_4 = np.sum(np.cos(WW), axis=1).reshape(-1, 1)/2
    Mean = np.hstack((gamma_1 + mu, gamma_2 + mu, gamma_3 + mu, gamma_4 + mu))
    T_value = np.max(Mean, axis=1).mean()
    T_Cvar75 = T_value - E_sigma * 0.4195317
    T_Cvar50 = T_value - E_sigma * 0.7911568
    T_Cvar25 = T_value - E_sigma * 1.258595
    T_Cvar10 = T_value - E_sigma * 1.72914
    return {'value':T_value, 'Cvar75':T_Cvar75, 'Cvar50':T_Cvar50, 'Cvar25':T_Cvar25, 'Cvar10':T_Cvar10}

def test_ITR(test_w,y_all_hat):
    mu = np.exp(test_w).mean(axis=1).reshape(-1, 1)
    ER_1 = mu + 2*np.sum(test_w, axis=1).reshape(-1, 1) 
    ER_2 = mu + np.sum( np.sqrt(np.abs(test_w)) , axis=1) .reshape(-1, 1)/2
    ER_3 = mu + np.sum(test_w**2, axis=1).reshape(-1, 1)/2
    ER_4 = mu + np.sum(np.cos(test_w), axis=1).reshape(-1, 1)/2
    ER = np.hstack((ER_1, ER_2, ER_3, ER_4))
    T_ITR = np.argmax(ER, axis=1)
    T_ITR = (T_ITR + 1).reshape(-1, 1)

    R1_test = np.zeros((test_size,1))
    R2_test = np.zeros((test_size,1))
    R3_test = np.zeros((test_size,1))
    R4_test = np.zeros((test_size,1))
    Cvar75_1_test = np.zeros((test_size,1))
    Cvar75_2_test = np.zeros((test_size,1))
    Cvar75_3_test = np.zeros((test_size,1))
    Cvar75_4_test = np.zeros((test_size,1))
    Cvar50_1_test = np.zeros((test_size,1))
    Cvar50_2_test = np.zeros((test_size,1))
    Cvar50_3_test = np.zeros((test_size,1))
    Cvar50_4_test = np.zeros((test_size,1))
    Cvar25_1_test = np.zeros((test_size,1))
    Cvar25_2_test = np.zeros((test_size,1))
    Cvar25_3_test = np.zeros((test_size,1))
    Cvar25_4_test = np.zeros((test_size,1))
    Cvar10_1_test = np.zeros((test_size,1))
    Cvar10_2_test = np.zeros((test_size,1))
    Cvar10_3_test = np.zeros((test_size,1))
    Cvar10_4_test = np.zeros((test_size,1))
    
    for rep_test in range(test_size):
        R1_Generator = np.sort( y_all_hat[rep_test,:,0] )
        R2_Generator = np.sort( y_all_hat[rep_test,:,1] )
        R3_Generator = np.sort( y_all_hat[rep_test,:,2] )
        R4_Generator = np.sort( y_all_hat[rep_test,:,3] )
        R1_test[rep_test, :] = R1_Generator.mean()
        R2_test[rep_test, :] = R2_Generator.mean()
        R3_test[rep_test, :] = R3_Generator.mean()
        R4_test[rep_test, :] = R4_Generator.mean()
        Cvar75_1_test[rep_test, :] = R1_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_2_test[rep_test, :] = R2_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_3_test[rep_test, :] = R3_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_4_test[rep_test, :] = R4_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar50_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar50_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar50_3_test[rep_test, :] = R3_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar50_4_test[rep_test, :] = R4_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar25_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar25_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar25_3_test[rep_test, :] = R3_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar25_4_test[rep_test, :] = R4_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar10_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/10))].mean()
        Cvar10_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/10))].mean()
        Cvar10_3_test[rep_test, :] = R3_Generator[0:round((MonteCarlo_size/10))].mean()
        Cvar10_4_test[rep_test, :] = R4_Generator[0:round((MonteCarlo_size/10))].mean()

    R_test = np.hstack((R1_test, R2_test, R3_test, R4_test))
    Cvar75_test = np.hstack((Cvar75_1_test, Cvar75_2_test, Cvar75_3_test, Cvar75_4_test))
    Cvar50_test = np.hstack((Cvar50_1_test, Cvar50_2_test, Cvar50_3_test, Cvar50_4_test))
    Cvar25_test = np.hstack((Cvar25_1_test, Cvar25_2_test, Cvar25_3_test, Cvar25_4_test))
    Cvar10_test = np.hstack((Cvar10_1_test, Cvar10_2_test, Cvar10_3_test, Cvar10_4_test))
    Mean_Est_ITR = (np.argmax(R_test, axis=1) + 1).reshape(-1, 1)
    Cvar75_Est_ITR = (np.argmax(Cvar75_test, axis=1) + 1).reshape(-1, 1)
    Cvar50_Est_ITR = (np.argmax(Cvar50_test, axis=1) + 1).reshape(-1, 1)
    Cvar25_Est_ITR = (np.argmax(Cvar25_test, axis=1) + 1).reshape(-1, 1)
    Cvar10_Est_ITR = (np.argmax(Cvar10_test, axis=1) + 1).reshape(-1, 1)
    Mean_rate = (T_ITR == Mean_Est_ITR).astype(int).mean()
    Cvar75_rate = (T_ITR == Cvar75_Est_ITR).astype(int).mean()
    Cvar50_rate = (T_ITR == Cvar50_Est_ITR).astype(int).mean()
    Cvar25_rate = (T_ITR == Cvar25_Est_ITR).astype(int).mean()
    Cvar10_rate = (T_ITR == Cvar10_Est_ITR).astype(int).mean()
    Est_value = np.max(R_test, axis=1).mean()
    Est_Cvar75 = np.max(Cvar75_test, axis=1).mean()
    Est_Cvar50 = np.max(Cvar50_test, axis=1).mean()
    Est_Cvar25 = np.max(Cvar25_test, axis=1).mean()
    Est_Cvar10 = np.max(Cvar10_test, axis=1).mean()
    Mean_bias  = np.abs(T_value - Est_value)/T_value 
    Cvar75_bias  = np.abs(T_Cvar75 - Est_Cvar75)/T_Cvar75
    Cvar50_bias  = np.abs(T_Cvar50 - Est_Cvar50)/T_Cvar50 
    Cvar25_bias  = np.abs(T_Cvar25 - Est_Cvar25)/T_Cvar25
    Cvar10_bias  = np.abs(T_Cvar10 - Est_Cvar10)/T_Cvar10 
    T_rate = np.hstack((Mean_rate, Cvar75_rate, Cvar50_rate, Cvar25_rate, Cvar10_rate))
    value_bias = np.hstack((Mean_bias, Cvar75_bias, Cvar50_bias, Cvar25_bias, Cvar10_bias))
    return {'T_rate':T_rate, 'value_bias':value_bias}

value = true_value(500000)
T_value = value['value']
T_Cvar75 = value['Cvar75']
T_Cvar50 = value['Cvar50']
T_Cvar25 = value['Cvar25']
T_Cvar10 = value['Cvar10']
classrate = np.zeros( (num_rep, 5) )
bias = np.zeros( (num_rep, 5) )
Ttime = np.zeros( (num_rep, 2) )

for rep_index in range(num_rep):
    Data = generate_data(rep_index)
    train_x = Data['train_x']
    train_t = Data['train_t']
    train_y = Data['train_y']
    train_z = Data['train_z']
    test_x = Data['test_x']
    test_w = Data['test_w']
    test_z = Data['test_z']
    ganite_res = ganite (train_x, train_t, train_y, train_z, test_x, test_z, parameters)
    y_all_hat = ganite_res['test_y_hat'].reshape(test_size, MonteCarlo_size, K)
    time1 = time.time()
    test_result = test_ITR(test_w,y_all_hat)
    time2 = time.time()
    classrate[rep_index,:] = test_result['T_rate']
    bias[rep_index,:] = test_result['value_bias']
    Ttime[rep_index,0] = (ganite_res['Ttime'])[0]
    Ttime[rep_index,1] = (ganite_res['Ttime'])[1] + time2 - time1
    
pd_classrate = pd.DataFrame(classrate)
pd_bias = pd.DataFrame(bias)
pd_Ttime = pd.DataFrame(Ttime)
pd_classrate.to_excel('classrate.xlsx',index=False,header=False)
pd_bias.to_excel('bias.xlsx',index=False,header=False)
pd_Ttime.to_excel('Ttime.xlsx',index=False,header=False)


