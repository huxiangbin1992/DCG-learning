# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:05:12 2024

@author: Hu Xiangbin
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from scipy.integrate import quad
from scipy.stats import truncnorm
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde
import os
# import ot
import matplotlib.pyplot as plt
import time
# from lifelines import KaplanMeierFitter
import pandas as pd

# Change the path to the current file
# script_path = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_path)
# os.getcwd()
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_rep = 100
sample_size = 2000
train_size = math.ceil(0.8 * sample_size)
validation_size = math.ceil(0.2 * sample_size)
test_size = 500
total_size = sample_size + test_size
MonteCarlo_size = 10000
J = math.ceil(20000/train_size)
cov_dimension = 2
K = 2
turning = 10
LearningR = 0.0001
reference_dimension = 4
G_width = 32
D_width = 32
epoch_size = 20000
patience = 2000
batch_size = math.ceil(0.05*train_size)
lambda_0 = 1

################## Define the deep neural network #################
class DCG(nn.Module):
    def __init__(self):
        super(DCG, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(cov_dimension + (K-1) + reference_dimension, G_width),
            nn.ReLU(),
            nn.Linear(G_width, G_width),
            nn.ReLU(),
            nn.Linear(G_width, 1),
        )
    def forward(self, x, a):
        xa = torch.cat((x, a), dim=-1)
        return self.main(xa)

class DCD(nn.Module):
    def __init__(self):
        super(DCD, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(cov_dimension + K, D_width),
            nn.ReLU(),
            nn.Linear(D_width, D_width),
            nn.ReLU(),
            nn.Linear(D_width, 1),
        )
    def forward(self, x, y):
        xy = torch.cat((x, y), dim=-1)
        return self.main(xy)
    
def dis_gradient(discriminator, X_train, Y_train):

    def relu_derivative(x): return (x > 0).float()

    dis_input = torch.cat((X_train, Y_train), dim=-1)
    
    # forward check
    a1 = discriminator.main[0](dis_input)
    h1 = discriminator.main[1](a1)
    a2 = discriminator.main[2](h1)
    # h2 = discriminator.main[3](a2)
    # output = discriminator.main[4](h2)
    
    # backward calculate
    grad_to_h2 = discriminator.main[4].weight
    grad_to_a2 = relu_derivative(a2) * grad_to_h2
    grad_to_h1 = torch.matmul(grad_to_a2, discriminator.main[2].weight)
    grad_to_a1 = relu_derivative(a1) * grad_to_h1
    grad = torch.matmul(grad_to_a1, discriminator.main[0].weight)

    return grad.squeeze() 

##################### Train the conditional GAN #########################

def train_generator(rep_index, Data):
    X_train = Data['X_train'].to(device)
    R_train = Data['R_train'].to(device)
    reference = Data['reference'].to(device)
    X_validation = Data['X_validation'].to(device)
    R_validation = Data['R_validation'].to(device)
    reference_validation = Data['reference_validation'].to(device)

    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    def Distribution_distence(X_validation, reference_validation, R_validation, generator):
        generated = generator(X_validation, reference_validation).detach().cpu().numpy()
        ws_distance = wasserstein_distance(R_validation.cpu().numpy().flatten(), generated.flatten())
        return ws_distance
    
    generator = DCG().to(device)
    discriminator = DCD().to(device)
    generator_optimizer = optim.RMSprop(generator.parameters(), lr=LearningR)
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=LearningR)
    
    one = torch.tensor(1, dtype=torch.float, device=device)
    minus_one = one * -1
    best_WD = float('inf')
    
    X_train_repeated = X_train.repeat(1, J).view(-1, cov_dimension + (K-1))
    offsets = torch.arange(0, J, device=device)
    patience_counter = 0

    ### start iteration
    for epoch in range(epoch_size):
        # Discriminator update
        # time1 = time.time()
        
        for l in discriminator.parameters():
            l.requires_grad = True
        discriminator.zero_grad()
        # Select indices
        perm = torch.randperm(train_size, device=device)  
        indices = perm[:batch_size]
        repeated_indices = (J * indices[:, None] + offsets[None, :]).view(-1)
        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real data
        d_loss_real = discriminator(X_train[indices,:], R_train[indices,:])
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(one)
        
        fake_images = generator(X_train_repeated[repeated_indices,:], reference[repeated_indices,:]).to(device)   
        d_loss_fake = discriminator(X_train_repeated[repeated_indices,:], fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(minus_one)
        
        d_loss_penalty = dis_gradient(discriminator,X_train[indices,:], R_train[indices,:])
        d_loss_penalty = torch.norm (d_loss_penalty , p=2, dim=1)
        d_loss_penalty = turning * ( d_loss_penalty - 1 ) ** 2
        d_loss_penalty = d_loss_penalty.mean()
        d_loss_penalty.backward(one)
        
        discriminator_optimizer.step()
        # time2 = time.time()
        
        # Generator update
        for l in discriminator.parameters():
            l.requires_grad = False  # to avoid computation
        generator.zero_grad()
        # Select indices
        perm = torch.randperm(X_train.size(0), device=device)  
        indices = perm[: 2 * batch_size]
        repeated_indices = (J * indices[:, None] + offsets[None, :]).view(-1)
        # Train generator
        fake_images = generator(X_train_repeated[repeated_indices,:], reference[repeated_indices,:]).to(device)
        g_loss = discriminator(X_train_repeated[repeated_indices,:], fake_images)
        g_loss = g_loss.mean()
        g_loss.backward()
        generator_optimizer.step()
        # time3 = time.time()
        
        WS_D = Distribution_distence(X_validation, reference_validation, R_validation, generator)
        # time4 = time.time()
        
        if WS_D < best_WD:
            best_WD= WS_D
            best_Wgenerator = generator.to(device)
            patience_counter = 0
            # Best_WDEpic = epoch
        else: 
            patience_counter += 1
        
        # if epoch % 1000 == 0:
            # print(f"epoch {epoch} WS_D {WS_D} best_WD {best_WD}")
        
        if epoch % 2000 == 0 and epoch > 9999:
            # Rate = Rate + 1
            for param_group in generator_optimizer.param_groups:
                param_group['lr'] *= 1/2
            for param_group in discriminator_optimizer.param_groups:
                param_group['lr'] *= 1/2
            # print(f"Epoch {epoch}, gloss {g_loss}, dloss {d_loss_real - d_loss_fake}")
        # time5 = time.time()    
        
        if patience_counter >= patience and epoch >= 9999:
            print("Early stopping triggered.")
            break

    # torch.save(best_Wgenerator.state_dict(), f"n{sample_size}_rep{rep_index}.pth")
    # singleP_residual_cdf(best_WDgenerator, e1, e2, e3, rep_index, 50000, Best_WDEpic, np.min(Y_train),np.max(Y_train))
    # print(f"best_WD {best_WD}")
    return best_Wgenerator

###################### Observation sample #####################
def generate_data(rep_index): 
    print(f"Replication {rep_index}")
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    A = np.random.choice([1, 2], size=(sample_size, 1), replace=True) # generate A from unif 0,1
    indicator_A = np.zeros((sample_size, K-1))
    indicator_A[ A[:, 0] > 1, (A[A[:, 0] > 1] - 2).squeeze()] = 1
    XX = np.random.uniform(-1, 1, size=(sample_size, 2 ))  # generate x3, x4 from unif distribution
    mu_1 = 0.5 * (XX[:,0] - XX[:,1]).reshape(-1, 1)
    mu_2 = 0.5 * (XX[:,1] - XX[:,0]).reshape(-1, 1)
    epsilon_1 = truncnorm(-3, 3, 0, 1).rvs(sample_size).reshape(-1, 1)
    epsilon_2 = truncnorm(-3, 3, 0, 0.1).rvs(sample_size).reshape(-1, 1)
    R_1 = mu_1 + 2 + epsilon_1
    R_2 = mu_2 + 2 + epsilon_2
    R = np.zeros_like(A,dtype="float64") 
    R[A == 1] = R_1[A == 1]
    R[A == 2] = R_2[A == 2]

    # turn to PyTorch tensor
    X = np.hstack((indicator_A, XX))
    X_train = torch.tensor(X[:train_size], dtype=torch.float32)
    R_train = torch.tensor(R[:train_size], dtype=torch.float32)
    X_validation = torch.tensor(X[train_size:train_size+validation_size], dtype=torch.float32)
    R_validation = torch.tensor(R[train_size:train_size+validation_size], dtype=torch.float32)
    X_test = torch.tensor(X[-test_size:, :], dtype=torch.float32)
    R_test = torch.tensor(R[-test_size:, :], dtype=torch.float32)
    
    reference =  torch.tensor( np.random.uniform(-1, 1, size=(X_train.shape[0]*J,reference_dimension) ) , dtype=torch.float32)
    reference_validation = torch.tensor( np.random.uniform(-1, 1, size=(validation_size,reference_dimension) ) , dtype=torch.float32)
    return {'X_train':X_train, 'R_train':R_train, 'reference':reference, 'X_validation':X_validation, 'R_validation':R_validation, 
            'reference_validation':reference_validation, 'X_test':X_test, 'R_test':R_test}

###################### Calculate the true value #####################
############# The CVaR of Normal distribution is given belwo ########
##### > library(cvar)
##### > library(truncnorm)
##### > ES(ptruncnorm , a=-3, b=3, dist.type = "cdf",p_loss = 0.75)
##### [1] 0.4195317
##### > ES(ptruncnorm , a=-3, b=3, dist.type = "cdf",p_loss = 0.5)
##### [1] 0.7911568
##### > ES(ptruncnorm , a=-3, b=3, dist.type = "cdf",p_loss = 0.25)
##### [1] 1.258595
##### > ES(ptruncnorm , a=-3, b=3, dist.type = "cdf",p_loss = 0.1)
##### [1] 1.72914
##### > ES(ptruncnorm , a=-3, b=3, dist.type = "cdf",p_loss = 0.05)
##### [1] 2.019352

##### > ES(ptruncnorm , a=-0.3, b=0.3, mean = 0, sd = 0.1, dist.type = "cdf",p_loss = 0.75)
##### [1] 0.04195317
##### > ES(ptruncnorm , a=-0.3, b=0.3, mean = 0, sd = 0.1, dist.type = "cdf",p_loss = 0.5)
##### [1] 0.07911568
##### > ES(ptruncnorm , a=-0.3, b=0.3, mean = 0, sd = 0.1, dist.type = "cdf",p_loss = 0.25)
##### [1] 0.1258595
##### > ES(ptruncnorm , a=-0.3, b=0.3, mean = 0, sd = 0.1, dist.type = "cdf",p_loss = 0.1)
##### [1] 0.172914
##### > ES(ptruncnorm , a=-0.3, b=0.3, mean = 0, sd = 0.1, dist.type = "cdf",p_loss = 0.05)
##### [1] 0.2019352

##################### Test Procedure #########################
def mesh_ITR(W_Gan):
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)
    points = np.arange(-1, 1.0001, 0.04)
    mesh_x1 = np.repeat(points, 51)
    mesh_x2 = np.tile(points, 51)
    mesh_size = 51*51
    X_mesh = torch.from_numpy(np.stack((mesh_x1, mesh_x2), axis=1)).float()
    
    XA1_mesh = torch.cat( (torch.full((mesh_size, 1), 0, device=device), X_mesh), dim=1)
    XA2_mesh = torch.cat( (torch.full((mesh_size, 1), 1, device=device), X_mesh), dim=1)
    reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , 
                                        dtype=torch.float32, device=device)

    R1_mesh = np.zeros((mesh_size,1))
    R2_mesh = np.zeros((mesh_size,1))
    Cvar75_1_mesh = np.zeros((mesh_size,1))
    Cvar75_2_mesh = np.zeros((mesh_size,1))
    Cvar50_1_mesh = np.zeros((mesh_size,1))
    Cvar50_2_mesh = np.zeros((mesh_size,1))
    Cvar25_1_mesh = np.zeros((mesh_size,1))
    Cvar25_2_mesh = np.zeros((mesh_size,1))
    Cvar10_1_mesh = np.zeros((mesh_size,1))
    Cvar10_2_mesh = np.zeros((mesh_size,1))
    
    for rep_mesh in range(mesh_size):
        XA1_repeat = XA1_mesh[rep_mesh, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA2_repeat = XA2_mesh[rep_mesh, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        R1_Generator = np.sort( W_Gan(XA1_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator = np.sort( W_Gan(XA2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R1_mesh[rep_mesh, :] = R1_Generator.mean()
        R2_mesh[rep_mesh, :] = R2_Generator.mean()
        Cvar75_1_mesh[rep_mesh, :] = np.sum( R1_Generator[0:round((3*MonteCarlo_size/4))] )/MonteCarlo_size/0.75
        Cvar75_2_mesh[rep_mesh, :] = np.sum( R2_Generator[0:round((3*MonteCarlo_size/4))] )/MonteCarlo_size/0.75
        Cvar50_1_mesh[rep_mesh, :] = np.sum( R1_Generator[0:round((MonteCarlo_size/2))] )/MonteCarlo_size/0.5
        Cvar50_2_mesh[rep_mesh, :] = np.sum( R2_Generator[0:round((MonteCarlo_size/2))] )/MonteCarlo_size/0.5
        Cvar25_1_mesh[rep_mesh, :] = np.sum( R1_Generator[0:round((MonteCarlo_size/4))] )/MonteCarlo_size/0.25
        Cvar25_2_mesh[rep_mesh, :] = np.sum( R2_Generator[0:round((MonteCarlo_size/4))] )/MonteCarlo_size/0.25
        Cvar10_1_mesh[rep_mesh, :] = np.sum( R1_Generator[0:round((MonteCarlo_size/10))] )/MonteCarlo_size/0.1
        Cvar10_2_mesh[rep_mesh, :] = np.sum( R2_Generator[0:round((MonteCarlo_size/10))] )/MonteCarlo_size/0.1
        
    R_mesh = np.hstack((R1_mesh, R2_mesh))
    Cvar75_mesh = np.hstack((Cvar75_1_mesh, Cvar75_2_mesh))
    Cvar50_mesh = np.hstack((Cvar50_1_mesh, Cvar50_2_mesh))
    Cvar25_mesh = np.hstack((Cvar25_1_mesh, Cvar25_2_mesh))
    Cvar10_mesh = np.hstack((Cvar10_1_mesh, Cvar10_2_mesh))
    Mean_Est_ITR = np.argmax(R_mesh, axis=1) + 1
    Cvar75_Est_ITR = np.argmax(Cvar75_mesh, axis=1) + 1
    Cvar50_Est_ITR = np.argmax(Cvar50_mesh, axis=1) + 1
    Cvar25_Est_ITR = np.argmax(Cvar25_mesh, axis=1) + 1
    Cvar10_Est_ITR = np.argmax(Cvar10_mesh, axis=1) + 1

    return {'X_mesh':X_mesh, 'Mean_Est_ITR':Mean_Est_ITR, 'Cvar75_Est_ITR':Cvar75_Est_ITR, 
            'Cvar50_Est_ITR':Cvar50_Est_ITR, 'Cvar25_Est_ITR':Cvar25_Est_ITR, 'Cvar10_Est_ITR':Cvar10_Est_ITR}

##################### Plot ITR #########################
def figure_ITR(Data,mesh_result):
    X_test = Data['X_test'][:, -cov_dimension:]
    X_mesh = mesh_result['X_mesh'][:, -cov_dimension:]
    Cvar75_Est_ITR = mesh_result['Cvar75_Est_ITR']
    Cvar50_Est_ITR = mesh_result['Cvar50_Est_ITR']
    Cvar25_Est_ITR = mesh_result['Cvar25_Est_ITR']
    Cvar10_Est_ITR = mesh_result['Cvar10_Est_ITR']
    beta_1 = torch.tensor([0.5, -0.5]).to(device) ########## combine the coefficient of mu to beta
    beta_2 = torch.tensor([-0.5, 0.5]).to(device) 
    ER1 = torch.matmul(X_test,beta_1) + 2
    ER2 = torch.matmul(X_test,beta_2) + 2
    # ER = torch.stack((ER1, ER2), dim=1)
    # Mean_ITR = (torch.argmax(ER, axis=1).cpu().numpy() + 1)
    Cvar75 = torch.stack((ER1 - 0.4195317, ER2 - 0.04195317), dim=1)
    Cvar75_ITR = (torch.argmax(Cvar75, axis=1).cpu().numpy() + 1)
    Cvar50 = torch.stack((ER1 - 0.7911568, ER2 - 0.07911568), dim=1)
    Cvar50_ITR = (torch.argmax(Cvar50, axis=1).cpu().numpy() + 1)
    Cvar25 = torch.stack((ER1 - 1.258595, ER2 - 0.1258595), dim=1)
    Cvar25_ITR = (torch.argmax(Cvar25, axis=1).cpu().numpy() + 1)
    Cvar10 = torch.stack((ER1 - 1.72914, ER2 - 0.172914), dim=1)
    Cvar10_ITR = (torch.argmax(Cvar10, axis=1).cpu().numpy() + 1)
    
    points = np.arange(-1, 1.0001, 0.04)
    X1_grid, X2_grid = np.meshgrid(points, points)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10)) 
    axs[0, 0].scatter(X_test[Cvar75_ITR == 1, 0], X_test[Cvar75_ITR == 1, 1], color='royalblue', marker='x', s=20)
    axs[0, 0].scatter(X_test[Cvar75_ITR == 2, 0], X_test[Cvar75_ITR == 2, 1], color='coral', marker='o', s=15)
    axs[0, 0].scatter(X_mesh[Cvar75_Est_ITR == 1, 0], X_mesh[Cvar75_Est_ITR == 1, 1], color='blue',s=1, marker='o', alpha=0.2)
    axs[0, 0].scatter(X_mesh[Cvar75_Est_ITR == 2, 0], X_mesh[Cvar75_Est_ITR == 2, 1], color='red', s=1, marker='o', alpha=0.2)
    # axs[0, 0].plot([-1, 1], [-1 - 0.4195317 + 0.04237021, 1 - 0.4195317 + 0.04237021], color='black', linestyle='-')
    axs[0, 0].contour(X1_grid, X2_grid, Cvar75_Est_ITR.reshape(X1_grid.shape).T, 
                  levels=[1.5], colors='black', linestyles='dashed', linewidths=1.5)
    axs[0, 0].set_xlim(-1, 1)
    axs[0, 0].set_ylim(-1, 1)
    axs[0, 0].set_title(r'$\alpha = 0.75$')
    axs[0, 0].set_xlabel(r'$X_1$')
    axs[0, 0].set_ylabel(r'$X_2$', labelpad = -5)
    axs[0, 1].scatter(X_test[Cvar50_ITR == 1, 0], X_test[Cvar50_ITR == 1, 1], color='royalblue', marker='x', s=20)
    axs[0, 1].scatter(X_test[Cvar50_ITR == 2, 0], X_test[Cvar50_ITR == 2, 1], color='coral', marker='o', s=15)
    axs[0, 1].scatter(X_mesh[Cvar50_Est_ITR == 1, 0], X_mesh[Cvar50_Est_ITR == 1, 1], color='blue',s=1, marker='o', alpha=0.2)
    axs[0, 1].scatter(X_mesh[Cvar50_Est_ITR == 2, 0], X_mesh[Cvar50_Est_ITR == 2, 1], color='red', s=1, marker='o', alpha=0.2)
    # axs[0, 1].plot([-1, 1], [-1 - 0.7911568 + 0.07978846, 1 - 0.7911568 + 0.07978846], color='black', linestyle='-')
    axs[0, 1].contour(X1_grid, X2_grid, Cvar50_Est_ITR.reshape(X1_grid.shape).T, 
                  levels=[1.5], colors='black', linestyles='dashed', linewidths=1.5)
    axs[0, 1].set_xlim(-1, 1)
    axs[0, 1].set_ylim(-1, 1)
    axs[0, 1].set_title(r'$\alpha = 0.5$')
    axs[0, 1].set_xlabel(r'$X_1$')
    axs[0, 1].set_ylabel(r'$X_2$', labelpad = -5)
    axs[1, 0].scatter(X_test[Cvar25_ITR == 1, 0], X_test[Cvar25_ITR == 1, 1], color='royalblue', marker='x', s=20)
    axs[1, 0].scatter(X_test[Cvar25_ITR == 2, 0], X_test[Cvar25_ITR == 2, 1], color='coral', marker='o', s=15)
    axs[1, 0].scatter(X_mesh[Cvar25_Est_ITR == 1, 0], X_mesh[Cvar25_Est_ITR == 1, 1], color='blue',s=1, marker='o', alpha=0.2)
    axs[1, 0].scatter(X_mesh[Cvar25_Est_ITR == 2, 0], X_mesh[Cvar25_Est_ITR == 2, 1], color='red', s=1, marker='o', alpha=0.2)
    # axs[1, 0].plot([-1, 1], [-1 - 1.258595 + 0.1271106, 1 - 1.258595 + 0.1271106], color='black', linestyle='-')
    axs[1, 0].contour(X1_grid, X2_grid, Cvar25_Est_ITR.reshape(X1_grid.shape).T, 
                  levels=[1.5], colors='black', linestyles='dashed', linewidths=1.5)
    axs[1, 0].set_xlim(-1, 1)
    axs[1, 0].set_ylim(-1, 1)
    axs[1, 0].set_title(r'$\alpha = 0.25$')
    axs[1, 0].set_xlabel(r'$X_1$')
    axs[1, 0].set_ylabel(r'$X_2$', labelpad = -5)
    axs[1, 1].scatter(X_test[Cvar10_ITR == 1, 0], X_test[Cvar10_ITR == 1, 1], color='royalblue', marker='x', s=20)
    axs[1, 1].scatter(X_test[Cvar10_ITR == 2, 0], X_test[Cvar10_ITR == 2, 1], color='coral', marker='o', s=15)
    axs[1, 1].scatter(X_mesh[Cvar10_Est_ITR == 1, 0], X_mesh[Cvar10_Est_ITR == 1, 1], color='blue',s=1, marker='o', alpha=0.2)
    axs[1, 1].scatter(X_mesh[Cvar10_Est_ITR == 2, 0], X_mesh[Cvar10_Est_ITR == 2, 1], color='red', s=1, marker='o', alpha=0.2)
    # axs[1, 1].plot([-1, 1], [-1 - 1.72914 + 0.1754983, 1 - 1.72914 + 0.1754983], color='black', linestyle='-')
    axs[1, 1].contour(X1_grid, X2_grid, Cvar10_Est_ITR.reshape(X1_grid.shape).T, 
                  levels=[1.5], colors='black', linestyles='dashed', linewidths=1.5)
    axs[1, 1].set_xlim(-1, 1)
    axs[1, 1].set_ylim(-1, 1)
    axs[1, 1].set_title(r'$\alpha = 0.1$')
    axs[1, 1].set_xlabel(r'$X_1$')
    axs[1, 1].set_ylabel(r'$X_2$', labelpad = -5)
    plt.savefig(f"figure_{sample_size}.jpg", format='jpg')
    plt.savefig(f"figure_{sample_size}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

rep_index = 0
Data = generate_data(rep_index)
# W_Gan = train_generator(rep_index, Data).to(device)
W_Gan = DCG().to(device)
W_Gan.load_state_dict(torch.load(f"n{sample_size}_rep{rep_index}.pth", map_location=torch.device('cpu')))
mesh_result = mesh_ITR(W_Gan)
figure_ITR(Data,mesh_result)


