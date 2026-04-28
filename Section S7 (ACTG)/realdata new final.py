# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:05:12 2024

@author: Hu Xiangbin
"""

#### LearningR = 0.001 width = 256 Layer = 3
#### LearningR = 0.0005 width = 256 Layer = 4

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import truncnorm
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_size = 2139
train_size = math.ceil(0.6 * sample_size)
validation_size = math.ceil(0.2 * sample_size)
test_size = sample_size - train_size - validation_size
MonteCarlo_size = 1000
J = math.ceil(20000/train_size)
cov_dimension = 12
K = 4
output_dimension = 1
turning = 10
reference_dimension = 15
G_width = 128
D_width = 128
LearningR = 0.00005
epoch_size = 20000
patience = 2000
batch_size = math.ceil(0.05*train_size) 
primdata = pd.read_csv('Realdata.csv').to_numpy() 
Comp_Est_ITR = pd.read_csv('ITR_Results_realdata.csv').to_numpy() 
rep_index = 0

def generate_data(): 
    
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    R = primdata[:,0].reshape(-1, 1)
    A = primdata[:,1].astype(int).reshape(-1, 1)
    X = primdata[:,2:(2+cov_dimension)]
    indicator_A = np.zeros((sample_size, K-1))
    indicator_A[ A[:, 0] > 1, (A[A[:, 0] > 1] - 2).squeeze()] = 1
    AX = np.hstack((indicator_A, X))
    A_test = A[:test_size, :]
    X_test = torch.tensor(AX[:test_size], dtype=torch.float32)
    R_test = torch.tensor(R[:test_size], dtype=torch.float32)
    X_train = torch.tensor(AX[test_size:test_size+train_size], dtype=torch.float32)
    R_train = torch.tensor(R[test_size:test_size+train_size], dtype=torch.float32)
    X_validation = torch.tensor(AX[-validation_size:, :], dtype=torch.float32)
    R_validation = torch.tensor(R[-validation_size:, :], dtype=torch.float32)
    
    reference =  torch.tensor( np.random.uniform(-1, 1, size=(X_train.shape[0]*J,reference_dimension) ) , dtype=torch.float32)
    reference_validation = torch.tensor( np.random.uniform(-1, 1, size=(validation_size,reference_dimension) ) , dtype=torch.float32)
    return {'X_train':X_train, 'R_train':R_train, 'reference':reference, 'X_validation':X_validation, 'R_validation':R_validation, 
            'reference_validation':reference_validation, 'A_test':A_test, 'X_test':X_test, 'R_test':R_test}

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

def train_generator():
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

    # torch.save(best_Wgenerator.state_dict(), f"n{sample_size}.pth")
    # singleP_residual_cdf(best_WDgenerator, e1, e2, e3, rep_index, 50000, Best_WDEpic, np.min(Y_train),np.max(Y_train))
    # print(f"best_WD {best_WD}")
    return best_Wgenerator

def test_ITR():
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)
    X_test = Data['X_test'][:, -cov_dimension:].to(device)
    A_test = Data['A_test']
    R_test = Data['R_test']
    XA1_test = torch.cat((torch.full((test_size, 3), 0, device=device), X_test), dim=1)
    XA2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 2), 0, device=device), X_test), dim=1)
    XA3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), 
                      torch.full((test_size, 1), 0, device=device), X_test), dim=1)
    XA4_test = torch.cat((torch.full((test_size, 2), 0, device=device), torch.full((test_size, 1), 1, device=device), X_test), dim=1)
    reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , 
                                        dtype=torch.float32, device=device)
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
        XA1_repeat = XA1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA2_repeat = XA2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA3_repeat = XA3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA4_repeat = XA4_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        R1_Generator = np.sort( W_Gan(XA1_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator = np.sort( W_Gan(XA2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R3_Generator = np.sort( W_Gan(XA3_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R4_Generator = np.sort( W_Gan(XA4_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
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
        
    RRR_test = np.hstack((R1_test, R2_test, R3_test, R4_test))
    Cvar75_test = np.hstack((Cvar75_1_test, Cvar75_2_test, Cvar75_3_test, Cvar75_4_test))
    Cvar50_test = np.hstack((Cvar50_1_test, Cvar50_2_test, Cvar50_3_test, Cvar50_4_test))
    Cvar25_test = np.hstack((Cvar25_1_test, Cvar25_2_test, Cvar25_3_test, Cvar25_4_test))
    Cvar10_test = np.hstack((Cvar10_1_test, Cvar10_2_test, Cvar10_3_test, Cvar10_4_test))
    Mean_Est_ITR = (np.argmax(RRR_test, axis=1) + 1).reshape(-1, 1)
    Cvar75_Est_ITR = (np.argmax(Cvar75_test, axis=1) + 1).reshape(-1, 1)
    Cvar50_Est_ITR = (np.argmax(Cvar50_test, axis=1) + 1).reshape(-1, 1)
    Cvar25_Est_ITR = (np.argmax(Cvar25_test, axis=1) + 1).reshape(-1, 1)
    Cvar10_Est_ITR = (np.argmax(Cvar10_test, axis=1) + 1).reshape(-1, 1)
    
    Index_Mean_Est = np.where(Mean_Est_ITR - A_test == 0)[0]
    Index_Cvar75_Est = np.where(Cvar75_Est_ITR - A_test == 0)[0]
    Index_Cvar50_Est = np.where(Cvar50_Est_ITR - A_test == 0)[0]
    Index_Cvar25_Est = np.where(Cvar25_Est_ITR - A_test == 0)[0]
    Index_Cvar10_Est = np.where(Cvar10_Est_ITR - A_test == 0)[0]
    Index_L1 = np.where( Comp_Est_ITR [0,:].reshape(-1, 1) - A_test == 0)[0]
    Index_AD = np.where( Comp_Est_ITR [1,:].reshape(-1, 1) - A_test == 0)[0]
    Index_RD = np.where( Comp_Est_ITR [2,:].reshape(-1, 1) - A_test == 0)[0]
    Index_SD = np.where( Comp_Est_ITR [3,:].reshape(-1, 1) - A_test == 0)[0]
    
    mean_list = [ torch.mean(R_test[Index_Mean_Est]).item(), torch.mean(R_test[Index_Cvar75_Est]).item(),
                  torch.mean(R_test[Index_Cvar50_Est]).item(), torch.mean(R_test[Index_Cvar25_Est]).item(),
                  torch.mean(R_test[Index_Cvar10_Est]).item(), torch.mean(R_test[Index_L1]).item(),
                  torch.mean(R_test[Index_AD]).item(), torch.mean(R_test[Index_RD]).item(), torch.mean(R_test[Index_SD]).item()]
    mean_array = np.array(mean_list).reshape(1, -1)
    return {'mean_array':mean_array, 'Mean_Est_ITR':Mean_Est_ITR,'Cvar75_Est_ITR':Cvar75_Est_ITR,
            'Cvar50_Est_ITR':Cvar50_Est_ITR, 'Cvar25_Est_ITR':Cvar25_Est_ITR, 'Cvar10_Est_ITR':Cvar10_Est_ITR}

def prob_ITR():
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    Y_baseline_test = primdata[-test_size:,14]
    X_test = Data['X_test'][:, -cov_dimension:].to(device)
    XA1_test = torch.cat((torch.full((test_size, 3), 0, device=device), X_test), dim=1)
    XA2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 2), 0, device=device), X_test), dim=1)
    XA3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), 
                      torch.full((test_size, 1), 0, device=device), X_test), dim=1)
    XA4_test = torch.cat((torch.full((test_size, 2), 0, device=device), torch.full((test_size, 1), 1, device=device), X_test), dim=1)
    reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , 
                                        dtype=torch.float32, device=device)
    Y1_Generator = np.zeros((test_size,MonteCarlo_size))
    Y2_Generator = np.zeros((test_size,MonteCarlo_size))
    Y3_Generator = np.zeros((test_size,MonteCarlo_size))
    Y4_Generator = np.zeros((test_size,MonteCarlo_size))
    for rep_test in range(test_size):
        XA1_repeat = XA1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA2_repeat = XA2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA3_repeat = XA3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA4_repeat = XA4_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        R1_Generator = np.sort( W_Gan(XA1_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator = np.sort( W_Gan(XA2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R3_Generator = np.sort( W_Gan(XA3_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R4_Generator = np.sort( W_Gan(XA4_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        Y1_Generator[rep_test,:] = (R1_Generator * 1494) - 634 + Y_baseline_test[rep_test] 
        Y2_Generator[rep_test,:] = (R2_Generator * 1494) - 634 + Y_baseline_test[rep_test]
        Y3_Generator[rep_test,:] = (R3_Generator * 1494) - 634 + Y_baseline_test[rep_test]
        Y4_Generator[rep_test,:] = (R4_Generator * 1494) - 634 + Y_baseline_test[rep_test]
    Prob_Generator_A1_200 = np.mean(Y1_Generator < 200, axis=1)
    Prob_Generator_A2_200 = np.mean(Y2_Generator < 200, axis=1)
    Prob_Generator_A3_200 = np.mean(Y3_Generator < 200, axis=1)
    Prob_Generator_A4_200 = np.mean(Y4_Generator < 200, axis=1)
    Prob_Generator_200 = np.vstack((Prob_Generator_A1_200, Prob_Generator_A2_200, Prob_Generator_A3_200, Prob_Generator_A4_200))
    Prob_Generator_A1_500 = np.mean(Y1_Generator < 500, axis=1)
    Prob_Generator_A2_500 = np.mean(Y2_Generator < 500, axis=1)
    Prob_Generator_A3_500 = np.mean(Y3_Generator < 500, axis=1)
    Prob_Generator_A4_500 = np.mean(Y4_Generator < 500, axis=1)
    Prob_Generator_500 = np.vstack((Prob_Generator_A1_500, Prob_Generator_A2_500, Prob_Generator_A3_500, Prob_Generator_A4_500))
    Mean_Est_ITR = test_results['Mean_Est_ITR']
    Cvar75_Est_ITR = test_results['Cvar75_Est_ITR']
    Cvar50_Est_ITR = test_results['Cvar50_Est_ITR']
    Cvar25_Est_ITR = test_results['Cvar25_Est_ITR']
    Cvar10_Est_ITR = test_results['Cvar10_Est_ITR']
    Q_Est_ITR = Comp_Est_ITR [0,:]
    AD_Est_ITR = Comp_Est_ITR [1,:]
    RD_Est_ITR = Comp_Est_ITR [2,:]
    SD_Est_ITR = Comp_Est_ITR [3,:]
    
    Mean_Est_Prob_200 = np.mean(Prob_Generator_200[(Mean_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar75_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar50_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar25_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar10_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Q_Est_Prob_200 = np.mean(Prob_Generator_200[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    AD_Est_Prob_200 = np.mean(Prob_Generator_200[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    RD_Est_Prob_200 = np.mean(Prob_Generator_200[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    SD_Est_Prob_200 = np.mean(Prob_Generator_200[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Prob_200 = np.array([Mean_Est_Prob_200, Cvar75_Est_Prob_200, Cvar50_Est_Prob_200, Cvar25_Est_Prob_200, Cvar10_Est_Prob_200, 
                              Q_Est_Prob_200, AD_Est_Prob_200, RD_Est_Prob_200, SD_Est_Prob_200])
    Mean_Est_Prob_500 = np.mean(Prob_Generator_500[(Mean_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar75_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar50_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar25_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar10_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Q_Est_Prob_500 = np.mean(Prob_Generator_500[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    AD_Est_Prob_500 = np.mean(Prob_Generator_500[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    RD_Est_Prob_500 = np.mean(Prob_Generator_500[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    SD_Est_Prob_500 = np.mean(Prob_Generator_500[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Prob_500 = np.array([Mean_Est_Prob_500, Cvar75_Est_Prob_500, Cvar50_Est_Prob_500, Cvar25_Est_Prob_500, Cvar10_Est_Prob_500, 
                              Q_Est_Prob_500, AD_Est_Prob_500, RD_Est_Prob_500, SD_Est_Prob_500])
    Est_Prob =  np.vstack((Est_Prob_200,Est_Prob_500))
    Y1_200 = np.sum(Y1_Generator * (Y1_Generator < 200), axis=1) / np.sum((Y1_Generator < 200), axis=1)
    Y2_200 = np.sum(Y2_Generator * (Y2_Generator < 200), axis=1) / np.sum((Y2_Generator < 200), axis=1)
    Y3_200 = np.sum(Y3_Generator * (Y3_Generator < 200), axis=1) / np.sum((Y3_Generator < 200), axis=1)
    Y4_200 = np.sum(Y4_Generator * (Y4_Generator < 200), axis=1) / np.sum((Y4_Generator < 200), axis=1)
    Y_200 = np.vstack((Y1_200, Y2_200, Y3_200, Y4_200))
    Y_200 = np.where(np.isnan(Y_200), 200, Y_200)
    Y1_500 = np.sum(Y1_Generator * (Y1_Generator < 500), axis=1) / np.sum((Y1_Generator < 500), axis=1)
    Y2_500 = np.sum(Y2_Generator * (Y2_Generator < 500), axis=1) / np.sum((Y2_Generator < 500), axis=1)
    Y3_500 = np.sum(Y3_Generator * (Y3_Generator < 500), axis=1) / np.sum((Y3_Generator < 500), axis=1)
    Y4_500 = np.sum(Y4_Generator * (Y4_Generator < 500), axis=1) / np.sum((Y4_Generator < 500), axis=1)
    Y_500 = np.vstack((Y1_500, Y2_500, Y3_500, Y4_500))
    Y_500 = np.where(np.isnan(Y_500), 500, Y_500)
    Mean_Est_Y_200 = np.mean(Y_200[(Mean_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Y_200 = np.mean(Y_200[(Cvar75_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Y_200 = np.mean(Y_200[(Cvar50_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Y_200 = np.mean(Y_200[(Cvar25_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Y_200 = np.mean(Y_200[(Cvar10_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Q_Est_Y_200 = np.mean(Y_200[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    AD_Est_Y_200 = np.mean(Y_200[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    RD_Est_Y_200 = np.mean(Y_200[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    SD_Est_Y_200 = np.mean(Y_200[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Y_200 = np.array([Mean_Est_Y_200, Cvar75_Est_Y_200, Cvar50_Est_Y_200, Cvar25_Est_Y_200, Cvar10_Est_Y_200, 
                              Q_Est_Y_200, AD_Est_Y_200, RD_Est_Y_200, SD_Est_Y_200])
    Mean_Est_Y_500 = np.mean(Y_500[(Mean_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Y_500 = np.mean(Y_500[(Cvar75_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Y_500 = np.mean(Y_500[(Cvar50_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Y_500 = np.mean(Y_500[(Cvar25_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Y_500 = np.mean(Y_500[(Cvar10_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Q_Est_Y_500 = np.mean(Y_500[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    AD_Est_Y_500 = np.mean(Y_500[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    RD_Est_Y_500 = np.mean(Y_500[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    SD_Est_Y_500 = np.mean(Y_500[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Y_500 = np.array([Mean_Est_Y_500, Cvar75_Est_Y_500, Cvar50_Est_Y_500, Cvar25_Est_Y_500, Cvar10_Est_Y_500, 
                              Q_Est_Y_500, AD_Est_Y_500, RD_Est_Y_500, SD_Est_Y_500])
    Est_Y =  np.vstack((Est_Y_200,Est_Y_500))
    return {'Est_Prob':Est_Prob, 'Est_Y':Est_Y}

def plot_ITR():
    
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)
    
    Y = primdata[:,0].reshape(-1, 1)
    A = primdata[:,1].reshape(-1, 1)
    X_test = Data['X_test'][:, -cov_dimension:].to(device)
    XA1_test = torch.cat((torch.full((test_size, 3), 0, device=device), X_test), dim=1)
    XA2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 2), 0, device=device), X_test), dim=1)
    XA3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), 
                      torch.full((test_size, 1), 0, device=device), X_test), dim=1)
    XA4_test = torch.cat((torch.full((test_size, 2), 0, device=device), torch.full((test_size, 1), 1, device=device), X_test), dim=1)
    reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , 
                                        dtype=torch.float32, device=device)

    R1_Generator = np.zeros((test_size, MonteCarlo_size))
    R2_Generator = np.zeros((test_size, MonteCarlo_size))
    R3_Generator = np.zeros((test_size, MonteCarlo_size))
    R4_Generator = np.zeros((test_size, MonteCarlo_size))
    
    for rep_test in range(test_size):
        XA1_repeat = XA1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA2_repeat = XA2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA3_repeat = XA3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA4_repeat = XA4_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        R1_Generator[rep_test,:] = np.sort( W_Gan(XA1_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator[rep_test,:] = np.sort( W_Gan(XA2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R3_Generator[rep_test,:] = np.sort( W_Gan(XA3_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R4_Generator[rep_test,:] = np.sort( W_Gan(XA4_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )

    Index_A1 = np.where( A == 1 ) [0]
    Index_A2 = np.where( A == 2 ) [0]
    Index_A3 = np.where( A == 3 ) [0]
    Index_A4 = np.where( A == 4 ) [0]
    kde_YA1 = gaussian_kde(Y[Index_A1].flatten())
    kde_YA2 = gaussian_kde(Y[Index_A2].flatten())
    kde_YA3 = gaussian_kde(Y[Index_A3].flatten())
    kde_YA4 = gaussian_kde(Y[Index_A4].flatten())
    kde_GR1 = gaussian_kde(R1_Generator.flatten())
    kde_GR2 = gaussian_kde(R2_Generator.flatten())
    kde_GR3 = gaussian_kde(R3_Generator.flatten())
    kde_GR4 = gaussian_kde(R4_Generator.flatten())
    range_YA1 = np.linspace( Y[Index_A1].min(), Y[Index_A1].max(), 1000)
    range_YA2 = np.linspace( Y[Index_A2].min(), Y[Index_A2].max(), 1000)
    range_YA3 = np.linspace( Y[Index_A3].min(), Y[Index_A3].max(), 1000)
    range_YA4 = np.linspace( Y[Index_A4].min(), Y[Index_A4].max(), 1000)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10)) 
    axs[0, 0].plot(range_YA1, kde_YA1(range_YA1), 'k-')
    axs[0, 0].plot(range_YA1, kde_GR1(range_YA1), 'r--')
    axs[0, 0].set_title('A=1', fontsize=7)
    axs[0, 0].tick_params(axis='x', labelsize=6)
    axs[0, 0].tick_params(axis='y', labelrotation=90, labelsize=8)
    axs[0, 0].grid(True)
    axs[0, 1].plot(range_YA2, kde_YA2(range_YA2), 'k-')
    axs[0, 1].plot(range_YA2, kde_GR2(range_YA2), 'r--')
    axs[0, 1].set_title('A=2', fontsize=7)
    axs[0, 1].tick_params(axis='x', labelsize=6)
    axs[0, 1].tick_params(axis='y', labelrotation=90, labelsize=8)
    axs[0, 1].grid(True)
    axs[1, 0].plot(range_YA3, kde_YA3(range_YA3), 'k-')
    axs[1, 0].plot(range_YA3, kde_GR3(range_YA3), 'r--')
    axs[1, 0].set_title('A=3', fontsize=7)
    axs[1, 0].tick_params(axis='x', labelsize=6)
    axs[1, 0].tick_params(axis='y', labelrotation=90, labelsize=8)
    axs[1, 0].grid(True)
    axs[1, 1].plot(range_YA4, kde_YA4(range_YA4), 'k-')
    axs[1, 1].plot(range_YA4, kde_GR4(range_YA4), 'r--')
    axs[1, 1].set_title('A=4', fontsize=7)
    axs[1, 1].tick_params(axis='x', labelsize=6)
    axs[1, 1].tick_params(axis='y', labelrotation=90, labelsize=8)
    axs[1, 1].grid(True)
    plt.suptitle('Kernel Density Estimation of the observation vs generation', y=0.95)
    plt.savefig( f"L2_W{G_width}_lr{LearningR}.png")
    plt.show()
    
Data = generate_data()
W_Gan = train_generator().to(device)
# W_Gan = DCG().to(device)
# W_Gan.load_state_dict(torch.load(f"n{sample_size}.pth", map_location=torch.device('cpu')))
test_results = test_ITR()
prob_results = prob_ITR()
plot_ITR()
np.savetxt(f"Ref{reference_dimension}_W{G_width}_lr{LearningR}EST_result.csv", test_results['mean_array'], delimiter=",")
np.savetxt(f"Ref{reference_dimension}_W{G_width}_lr{LearningR}prob.csv", prob_results['Est_Prob'], delimiter=",")
np.savetxt(f"Ref{reference_dimension}_W{G_width}_lr{LearningR}CondY.csv", prob_results['Est_Y'], delimiter=",")
