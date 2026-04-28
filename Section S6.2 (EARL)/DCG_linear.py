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
import random
# import ot
import matplotlib.pyplot as plt
import time
# from lifelines import KaplanMeierFitter
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Subset


# Change the path to the current file
######## script_path = os.path.dirname(os.path.abspath(__file__))
######## os.chdir(script_path)
######## os.getcwd()
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_rep = 100
sample_size = 200
train_size = math.ceil(0.8 * sample_size)
validation_size = math.ceil(0.2 * sample_size)
test_size = 500
total_size = test_size + sample_size
MonteCarlo_size = 10000
J = math.ceil(20000/train_size)
Cov_dimension = 2
K = 2
turning = 10
LearningR = 0.0005
reference_dimension = 4
G_width = 32
D_width = 32
epoch_size = 20000
patience = 2000
batch_size = math.ceil(0.05*train_size)

################## Define the deep neural network #################
class DCG(nn.Module):
    def __init__(self):
        super(DCG, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(Cov_dimension + (K-1) + reference_dimension, G_width),
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
            nn.Linear(Cov_dimension + K, D_width),
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

def train_generator(rep_index, Data):
    XA_train = Data['XA_train'].to(device)
    R_train = Data['R_train'].to(device)
    reference = Data['reference'].to(device)
    XA_validation = Data['XA_validation'].to(device)
    R_validation = Data['R_validation'].to(device)
    reference_validation = Data['reference_validation'].to(device)

    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    def Distribution_distence(XA_validation, reference_validation, R_validation, generator):
        generated = generator(XA_validation, reference_validation).detach().cpu().numpy()
        ws_distance = wasserstein_distance(R_validation.cpu().numpy().flatten(), generated.flatten())
        return ws_distance
    
    generator = DCG().to(device)
    discriminator = DCD().to(device)
    generator_optimizer = optim.RMSprop(generator.parameters(), lr=LearningR)
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=LearningR)
    
    one = torch.tensor(1, dtype=torch.float, device=device)
    minus_one = one * -1
    best_WD = float('inf')
    
    XA_train_repeated = XA_train.repeat(1, J).view(-1, Cov_dimension + (K-1))
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
        d_loss_real = discriminator(XA_train[indices,:], R_train[indices,:])
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(one)
        
        fake_images = generator(XA_train_repeated[repeated_indices,:], reference[repeated_indices,:]).to(device)   
        d_loss_fake = discriminator(XA_train_repeated[repeated_indices,:], fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(minus_one)
        
        d_loss_penalty = dis_gradient(discriminator,XA_train[indices,:], R_train[indices,:])
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
        perm = torch.randperm(XA_train.size(0), device=device)  
        indices = perm[: 2 * batch_size]
        repeated_indices = (J * indices[:, None] + offsets[None, :]).view(-1)
        # Train generator
        fake_images = generator(XA_train_repeated[repeated_indices,:], reference[repeated_indices,:]).to(device)
        g_loss = discriminator(XA_train_repeated[repeated_indices,:], fake_images)
        g_loss = g_loss.mean()
        g_loss.backward()
        generator_optimizer.step()
        # time3 = time.time()
        
        WS_D = Distribution_distence(XA_validation, reference_validation, R_validation, generator)
        # time4 = time.time()
        
        if WS_D < best_WD:
            best_WD= WS_D
            best_Wgenerator = generator.to(device)
            patience_counter = 0
            # Best_WDEpic = epoch
        else: 
            patience_counter += 1
        
        if epoch % 1000 == 0:
            print(f"epoch {epoch} WS_D {WS_D} best_WD {best_WD}")
        
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
    XX = truncnorm.rvs(-3, 3, loc = 0, scale = 1, size = (total_size,Cov_dimension)) # generate x1, x2 from normal distribution
    PA = np.hstack(( np.exp( XX[:,0].reshape(-1, 1) ) , np.repeat(1,total_size).reshape(-1, 1) ))
    PA = PA / PA.sum(axis=1).reshape(-1, 1) 
    choices = [0,1] 
    indicator_A = np.zeros(total_size, dtype=int).reshape(-1, 1) 
    for i in range(total_size):
        indicator_A[i] = np.random.choice(choices, p = PA[i])
    mu = 2 + XX.sum(axis=1).reshape(-1, 1)
    gamma_1 = (XX[:,0] + XX[:,1]).reshape(-1, 1)
    gamma_2 = (- XX[:,0] - XX[:,1]).reshape(-1, 1)
    # epsilon_sd = (np.sqrt( np.sum(XX**2,axis=1) )/Cov_dimension).reshape(-1, 1)
    # epsilon = np.zeros(total_size).reshape(-1, 1)
    # for i in range(total_size):
    #     epsilon[i] = truncnorm(-3, 3, 0, epsilon_sd[i]).rvs(1)
    epsilon = truncnorm.rvs(-3, 3, loc = 0, scale = 0.5, size = (total_size,1)).reshape(-1, 1)
        
    R_1 = mu + gamma_1 + epsilon
    R_2 = mu + gamma_2 + epsilon
    R = np.zeros_like(indicator_A,dtype="float64") 
    R[indicator_A == 1] = R_1[indicator_A == 1]
    R[indicator_A == 0] = R_2[indicator_A == 0]

    XA = np.hstack((indicator_A, XX))
    XA_train = torch.tensor(XA[:train_size], dtype=torch.float32)
    R_train = torch.tensor(R[:train_size], dtype=torch.float32)
    XA_validation = torch.tensor(XA[train_size:train_size+validation_size], dtype=torch.float32)
    R_validation = torch.tensor(R[train_size:train_size+validation_size], dtype=torch.float32)
    XX_test = torch.tensor(XX[-test_size:, :], dtype=torch.float32)

    reference =  torch.tensor( np.random.uniform(-1, 1, size=(XA_train.shape[0]*J,reference_dimension) ) , dtype=torch.float32)
    reference_validation = torch.tensor( np.random.uniform(-1, 1, size=(validation_size,reference_dimension) ) , dtype=torch.float32)
    return {'XA_train':XA_train, 'R_train':R_train, 'reference':reference, 'XA_validation':XA_validation, 
            'R_validation':R_validation, 'reference_validation':reference_validation, 'XX_test':XX_test}

def true_value(Mon_size):
    XX = truncnorm.rvs(-3, 3, loc = 0, scale = 1, size = (Mon_size,Cov_dimension))  # generate x1, x2 from normal distribution
    # E_sigma = np.mean( np.sqrt(np.sum(XX**2, axis=1)) /Cov_dimension)
    E_sigma = 0.5
    mu = 2 + XX.sum(axis=1).reshape(-1, 1)
    gamma_1 = (XX[:,0] + XX[:,1]).reshape(-1, 1)
    gamma_2 = (- XX[:,0] - XX[:,1]).reshape(-1, 1)
    Mean = np.hstack((gamma_1 + mu, gamma_2 + mu))
    T_value = np.max(Mean, axis=1).mean()
    T_Cvar75 = T_value - E_sigma * 0.4195317
    T_Cvar50 = T_value - E_sigma * 0.7911568
    T_Cvar25 = T_value - E_sigma * 1.258595
    T_Cvar10 = T_value - E_sigma * 1.72914
    return {'value':T_value, 'Cvar75':T_Cvar75, 'Cvar50':T_Cvar50, 'Cvar25':T_Cvar25, 'Cvar10':T_Cvar10}

##################### Test Procedure #########################
def test_ITR(Data,W_Gan):
    XX_test = Data['XX_test'][:, -Cov_dimension:].to(device)
    mu = 2 + XX_test.sum(axis=1).reshape(-1, 1)
    ER_1 = mu + (XX_test[:,0] + XX_test[:,1]).view(-1, 1)
    ER_2 = mu + (- XX_test[:,0] - XX_test[:,1]).view(-1, 1)
    ER = torch.cat((ER_1, ER_2), dim=1)
    T_ITR = torch.argmax(ER, axis=1).cpu().numpy()
    T_ITR = (1 - T_ITR).reshape(-1, 1)
    XA1_test = torch.cat((torch.full((test_size, 1), 1, device=device), XX_test), dim=1)
    XA2_test = torch.cat((torch.full((test_size, 1), 0, device=device), XX_test), dim=1)
    reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , 
                                        dtype=torch.float32, device=device)
    R1_test = np.zeros((test_size,1))
    R2_test = np.zeros((test_size,1))
    Cvar75_1_test = np.zeros((test_size,1))
    Cvar75_2_test = np.zeros((test_size,1))
    Cvar50_1_test = np.zeros((test_size,1))
    Cvar50_2_test = np.zeros((test_size,1))
    Cvar25_1_test = np.zeros((test_size,1))
    Cvar25_2_test = np.zeros((test_size,1))
    Cvar10_1_test = np.zeros((test_size,1))
    Cvar10_2_test = np.zeros((test_size,1))
    
    for rep_test in range(test_size):
        XA1_repeat = XA1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA2_repeat = XA2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        R1_Generator = np.sort( W_Gan(XA1_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator = np.sort( W_Gan(XA2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R1_test[rep_test, :] = R1_Generator.mean()
        R2_test[rep_test, :] = R2_Generator.mean()
        Cvar75_1_test[rep_test, :] = R1_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_2_test[rep_test, :] = R2_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar50_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar50_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar25_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar25_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar10_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/10))].mean()
        Cvar10_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/10))].mean()
        
    R_test = np.hstack((R1_test, R2_test))
    Cvar75_test = np.hstack((Cvar75_1_test, Cvar75_2_test))
    Cvar50_test = np.hstack((Cvar50_1_test, Cvar50_2_test))
    Cvar25_test = np.hstack((Cvar25_1_test, Cvar25_2_test))
    Cvar10_test = np.hstack((Cvar10_1_test, Cvar10_2_test))
    Mean_Est_ITR = np.argmax(R_test, axis=1).reshape(-1, 1)
    Cvar75_Est_ITR = np.argmax(Cvar75_test, axis=1).reshape(-1, 1)
    Cvar50_Est_ITR = np.argmax(Cvar50_test, axis=1).reshape(-1, 1)
    Cvar25_Est_ITR = np.argmax(Cvar25_test, axis=1).reshape(-1, 1)
    Cvar10_Est_ITR = np.argmax(Cvar10_test, axis=1).reshape(-1, 1)
    Mean_Est_ITR = (1 - Mean_Est_ITR).reshape(-1, 1)
    Cvar75_Est_ITR = (1 - Cvar75_Est_ITR).reshape(-1, 1)
    Cvar50_Est_ITR = (1 - Cvar50_Est_ITR).reshape(-1, 1)
    Cvar25_Est_ITR = (1 - Cvar25_Est_ITR).reshape(-1, 1)
    Cvar10_Est_ITR = (1 - Cvar10_Est_ITR).reshape(-1, 1)
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
    time1 = time.time()
    W_Gan = train_generator(rep_index, Data).to(device)
    time2 = time.time()
    test_result = test_ITR(Data,W_Gan)
    time3 = time.time()
    classrate[rep_index,:] = test_result['T_rate']
    bias[rep_index,:] = test_result['value_bias']
    Ttime[rep_index,0] = time2-time1
    Ttime[rep_index,1] = time3-time2

torch.save(classrate, f"n{total_size}_classrate.pth")
torch.save(bias, f"n{total_size}_bias.pth")
torch.save(Ttime, f"n{total_size}_Ttime.pth")

pd_classrate = pd.DataFrame(classrate)
pd_bias = pd.DataFrame(bias)
pd_Ttime = pd.DataFrame(Ttime)
pd_classrate.to_excel('classrate.xlsx',index=False,header=False)
pd_bias.to_excel('bias.xlsx',index=False,header=False)
pd_Ttime.to_excel('Ttime.xlsx',index=False,header=False)