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
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde
import pyreadr
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_size = 639
train_size = math.ceil(0.8 * sample_size)
validation_size = math.ceil(0.2 * sample_size)
test_size = 94
MonteCarlo_size = 1000
J = math.ceil(20000/train_size)
K = 3
turning = 10
tabular_dimension = 7
X2_dimension = 4800
cov_dimension = X2_dimension + tabular_dimension
reference_dimension = 20 ## can be changed
G_width = 8192
D_width = 8192
LearningR = 0.00000005
latent_dimension = 10
G_width_image = 256
Embedding_width = 256
D_width_image = 8192
LearningR_image = 0.000001
epoch_size = 20000
patience = 2000
batch_size = math.ceil(0.05*train_size)
ROI_data = pyreadr.read_r("ADNI_ROI_data.RData")
# print(ROI_data.keys())
Tabular_data = pyreadr.read_r("ADNI_Tabular_data.RData")
df_data = Tabular_data["df_data"].copy()
df_data["subject_id"] = df_data["subject_id"].astype(str).str.strip()
train_ids = ROI_data["train_SubjectID"]["train_SubjectID"].astype(str).str.strip().to_numpy()
test_ids  = ROI_data["test_SubjectID"]["test_SubjectID"].astype(str).str.strip().to_numpy()
df_train = (df_data.set_index("subject_id").reindex(train_ids).reset_index())
df_test = (df_data.set_index("subject_id").reindex(test_ids).reset_index())
mask = ~df_test["APOE4_count1"].isna()
df_test = df_test[mask]
test_ids = test_ids[mask.values]
# Comp_Est_ITR = pd.read_csv('ITR_Results_realdata.csv').to_numpy()
rep_index = 0

def generate_data(): 
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    R = np.array(df_train["diff_score"]).reshape(-1, 1)
    indicator_A = df_train[["KEYMED_1", "KEYMED_2"]].to_numpy()
    X_1 = df_train[["PTGENDER", "entry_age","PTEDUCAT", "APOE4_count1", "APOE4_count2", "BCPREDX_1", "BCPREDX_2"]].to_numpy()
    X_2 = ROI_data["train_X"].values.reshape(ROI_data["train_X"].shape[0], -1)
    X = np.concatenate([X_1, X_2], axis=1)
    AX = np.hstack((indicator_A, X))
    AX1 = np.hstack((indicator_A, X_1))
    
    R_test = np.array(df_test["diff_score"]).reshape(-1, 1)
    indicator_A_test = df_test[["KEYMED_1", "KEYMED_2"]].to_numpy()
    X1_test = df_test[["PTGENDER", "entry_age","PTEDUCAT", "APOE4_count1", "APOE4_count2", "BCPREDX_1", "BCPREDX_2"]].to_numpy()
    X2_test = ROI_data["test_X"].values.reshape(ROI_data["test_X"].shape[0], -1)
    X2_test = X2_test[mask]
    X_test = np.concatenate([X1_test, X2_test], axis=1)

    X_train = torch.tensor(AX[0:train_size], dtype=torch.float32)
    X1_train = torch.tensor(AX1[0:train_size], dtype=torch.float32)
    R_train = torch.tensor(R[0:train_size], dtype=torch.float32)
    X2_train = torch.tensor(X_2[0:train_size], dtype=torch.float32)
    X_validation = torch.tensor(AX[-validation_size:, :], dtype=torch.float32)
    X1_validation = torch.tensor(AX1[-validation_size:, :], dtype=torch.float32)
    R_validation = torch.tensor(R[-validation_size:, :], dtype=torch.float32)
    X2_validation = torch.tensor(X_2[-validation_size:, :], dtype=torch.float32)
    
    reference =  torch.tensor( np.random.uniform(-1, 1, size=(X_train.shape[0]*J,reference_dimension) ) , dtype=torch.float32)
    reference_validation = torch.tensor( np.random.uniform(-1, 1, size=(validation_size,reference_dimension) ) , dtype=torch.float32)
    return {'X_train':X_train, 'X1_train':X1_train, 'R_train':R_train, 'X2_train':X2_train, 'reference':reference, 'X_validation':X_validation, 
            'X1_validation':X1_validation, 'R_validation':R_validation, 'X2_validation':X2_validation, 'reference_validation':reference_validation, 
            'indicator_A_test':indicator_A_test, 'X_test':X_test, 'X1_test':X1_test, 'X2_test':X2_test, 'R_test':R_test}

################## Define the deep neural network #################
class DCG(nn.Module):
    def __init__(self):
        super(DCG, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(cov_dimension + (K-1) + reference_dimension, G_width),
            nn.ReLU(),
            nn.Linear(G_width, G_width),
            nn.ReLU(),
            nn.Linear(G_width, G_width),
            nn.ReLU(),
            nn.Linear(G_width, 1),
        )
    def forward(self, x, a):
        xa = torch.cat((x, a), dim=-1)
        return self.main(xa)
    
class DCG_image(nn.Module):
    def __init__(self):
        super(DCG_image, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(X2_dimension, Embedding_width),
            nn.ReLU(),
            nn.Linear(Embedding_width, Embedding_width),
            nn.ReLU(),
            nn.Linear(Embedding_width, latent_dimension),
        )
        self.f = nn.Sequential(
            nn.Linear(latent_dimension + (K-1) + tabular_dimension + reference_dimension, G_width_image),
            nn.ReLU(),
            nn.Linear(G_width_image, G_width_image),
            nn.ReLU(),
            nn.Linear(G_width_image, 1),
        )
    def forward(self, x, m, eta):
        hm = self.h(m)
        total = torch.cat((x, hm, eta), dim=-1)
        return self.f(total)

class DCD(nn.Module):
    def __init__(self):
        super(DCD, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(cov_dimension + K, D_width),
            nn.ReLU(),
            nn.Linear(D_width, D_width),
            nn.ReLU(),
            nn.Linear(D_width, D_width),
            nn.ReLU(),
            nn.Linear(D_width, 1),
        )
    def forward(self, x, y):
        xy = torch.cat((x, y), dim=-1)
        return self.main(xy)
    
class DCD_image(nn.Module):
    def __init__(self):
        super(DCD_image, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(X2_dimension + tabular_dimension + K, D_width_image),
            nn.ReLU(),
            nn.Linear(D_width_image, D_width_image),
            nn.ReLU(),
            nn.Linear(D_width_image, D_width_image),
            nn.ReLU(),
            nn.Linear(D_width_image, 1),
        )
    def forward(self, x, m, y):
        xy = torch.cat((x, m, y), dim=-1)
        return self.main(xy)

# def dis_gradient_image(discriminator, X_train, Z_train, Y_train):

#     def relu_derivative(x): return (x > 0).float()

#     dis_input = torch.cat((X_train,Z_train,Y_train), dim=-1)
    
#     # forward check
#     a1 = discriminator.main[0](dis_input)
#     h1 = discriminator.main[1](a1)
#     a2 = discriminator.main[2](h1)
#     # h2 = discriminator.main[3](a2)
#     # output = discriminator.main[4](h2)
    
#     # backward calculate
#     grad_to_h2 = discriminator.main[4].weight
#     grad_to_a2 = relu_derivative(a2) * grad_to_h2
#     grad_to_h1 = torch.matmul(grad_to_a2, discriminator.main[2].weight)
#     grad_to_a1 = relu_derivative(a1) * grad_to_h1
#     grad = torch.matmul(grad_to_a1, discriminator.main[0].weight)

#     return grad.squeeze() 

def dis_gradient(discriminator, X_train, Y_train):

    def relu_derivative(x): return (x > 0).float()

    dis_input = torch.cat((X_train, Y_train), dim=-1)
    
    # forward check
    a1 = discriminator.main[0](dis_input)
    h1 = discriminator.main[1](a1)
    a2 = discriminator.main[2](h1)
    h2 = discriminator.main[3](a2)
    a3 = discriminator.main[4](h2)
    
    # backward calculate
    grad_to_h3 = discriminator.main[6].weight
    grad_to_a3 = relu_derivative(a3) * grad_to_h3
    grad_to_h2 = torch.matmul(grad_to_a3, discriminator.main[4].weight)
    grad_to_a2 = relu_derivative(a2) * grad_to_h2
    grad_to_h1 = torch.matmul(grad_to_a2, discriminator.main[2].weight)
    grad_to_a1 = relu_derivative(a1) * grad_to_h1
    grad = torch.matmul(grad_to_a1, discriminator.main[0].weight)

    return grad.squeeze() 

def dis_gradient_image(discriminator, X1_train, X2_train, Y_train):

    def relu_derivative(x): return (x > 0).float()

    dis_input = torch.cat((X1_train,X2_train,Y_train), dim=-1)
    
    # forward check
    a1 = discriminator.main[0](dis_input)
    h1 = discriminator.main[1](a1)
    a2 = discriminator.main[2](h1)
    h2 = discriminator.main[3](a2)
    a3 = discriminator.main[4](h2)
    # h3 = discriminator.main[5](a3)
    # output = discriminator.main[6](h3)
    
    # backward check
    grad_to_h3 = discriminator.main[6].weight
    grad_to_a3 = relu_derivative(a3) * grad_to_h3
    grad_to_h2 = torch.matmul(grad_to_a3, discriminator.main[4].weight)
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
    
    torch.save(best_Wgenerator.state_dict(), "W_Gan.pth")

    return best_Wgenerator

def train_generator_image():
    X1_train = Data['X1_train'].to(device)
    X2_train = Data['X2_train'].to(device)
    R_train = Data['R_train'].to(device)
    reference = Data['reference'].to(device)
    X1_validation = Data['X1_validation'].to(device)
    X2_validation = Data['X2_validation'].to(device)
    R_validation = Data['R_validation'].to(device)
    reference_validation = Data['reference_validation'].to(device)

    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    def Distribution_distence_image(X1_validation, X2_validation, reference_validation, R_validation, generator):
        generated = generator(X1_validation, X2_validation, reference_validation).detach().cpu().numpy()
        ws_distance = wasserstein_distance(R_validation.cpu().numpy().flatten(), generated.flatten())
        return ws_distance
    
    generator = DCG_image().to(device)
    discriminator = DCD_image().to(device)
    generator_optimizer = optim.RMSprop(generator.parameters(), lr=LearningR_image)
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=LearningR_image)

    one = torch.tensor(1, dtype=torch.float, device=device)
    minus_one = one * -1
    best_WD = float('inf')

    X1_train_repeated = X1_train.repeat(1, J).view(-1, tabular_dimension + (K-1))
    X2_train_repeated = X2_train.repeat(1, J).view(-1, X2_dimension)
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
        d_loss_real = discriminator(X1_train[indices,:], X2_train[indices,:], R_train[indices,:])
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(one)
        
        fake_images = generator(X1_train_repeated[repeated_indices,:], X2_train_repeated[repeated_indices,:], reference[repeated_indices,:]).to(device)   
        d_loss_fake = discriminator(X1_train_repeated[repeated_indices,:], X2_train_repeated[repeated_indices,:], fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(minus_one)
        
        d_loss_penalty = dis_gradient_image(discriminator,X1_train[indices,:],X2_train[indices,:],R_train[indices,:])
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
        perm = torch.randperm(X1_train.size(0), device=device)  
        indices = perm[: 2 * batch_size]
        repeated_indices = (J * indices[:, None] + offsets[None, :]).view(-1)
        # Train generator
        fake_images = generator(X1_train_repeated[repeated_indices,:],X2_train_repeated[repeated_indices,:],reference[repeated_indices,:]).to(device)
        g_loss = discriminator(X1_train_repeated[repeated_indices,:],X2_train_repeated[repeated_indices,:],fake_images)
        g_loss = g_loss.mean()
        g_loss.backward()
        generator_optimizer.step()
        # time3 = time.time()

        WS_D = Distribution_distence_image(X1_validation, X2_validation, reference_validation, R_validation, generator)
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

    torch.save(best_Wgenerator.state_dict(), "W_Gan_image.pth")
    # singleP_residual_cdf(best_WDgenerator, e1, e2, e3, rep_index, 50000, Best_WDEpic, np.min(Y_train),np.max(Y_train))
    # print(f"best_WD {best_WD}")
    return best_Wgenerator

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

def test_ITR():
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)
    X_test = torch.tensor(Data['X_test'], dtype=torch.float32).to(device)
    X1_test = torch.tensor(Data['X1_test'], dtype=torch.float32).to(device)
    X2_test = torch.tensor(Data['X2_test'], dtype=torch.float32).to(device)
    A_test = Data['indicator_A_test'] 
    A_test_label = np.where( (A_test[:,0] == 0) & (A_test[:,1] == 0), 1, 
                        np.where( (A_test[:,0] == 1) & (A_test[:,1] == 0), 2, 3) ).reshape(-1,1)
    R_test = torch.tensor(Data['R_test'], dtype=torch.float32).to(device)
    XA1_test = torch.cat((torch.full((test_size, 2), 0, device=device), X_test), dim=1)
    XA2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 1), 0, device=device), X_test), dim=1)
    XA3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), X_test), dim=1)
    X1A1_test = torch.cat((torch.full((test_size, 2), 0, device=device), X1_test), dim=1)
    X1A2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 1), 0, device=device), X1_test), dim=1)
    X1A3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), X1_test), dim=1)
    reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , dtype=torch.float32, device=device)
    
    R1_test = np.zeros((test_size,1))
    R2_test = np.zeros((test_size,1))
    R3_test = np.zeros((test_size,1))
    Cvar75_1_test = np.zeros((test_size,1))
    Cvar75_2_test = np.zeros((test_size,1))
    Cvar75_3_test = np.zeros((test_size,1))
    Cvar50_1_test = np.zeros((test_size,1))
    Cvar50_2_test = np.zeros((test_size,1))
    Cvar50_3_test = np.zeros((test_size,1))
    Cvar25_1_test = np.zeros((test_size,1))
    Cvar25_2_test = np.zeros((test_size,1))
    Cvar25_3_test = np.zeros((test_size,1))
    Cvar10_1_test = np.zeros((test_size,1))
    Cvar10_2_test = np.zeros((test_size,1))
    Cvar10_3_test = np.zeros((test_size,1))
    
    R1_test_image = np.zeros((test_size,1))
    R2_test_image = np.zeros((test_size,1))
    R3_test_image = np.zeros((test_size,1))
    Cvar75_1_test_image = np.zeros((test_size,1))
    Cvar75_2_test_image = np.zeros((test_size,1))
    Cvar75_3_test_image = np.zeros((test_size,1))
    Cvar50_1_test_image = np.zeros((test_size,1))
    Cvar50_2_test_image = np.zeros((test_size,1))
    Cvar50_3_test_image = np.zeros((test_size,1))
    Cvar25_1_test_image = np.zeros((test_size,1))
    Cvar25_2_test_image = np.zeros((test_size,1))
    Cvar25_3_test_image = np.zeros((test_size,1))
    Cvar10_1_test_image = np.zeros((test_size,1))
    Cvar10_2_test_image = np.zeros((test_size,1))
    Cvar10_3_test_image = np.zeros((test_size,1))
    for rep_test in range(test_size):
        XA1_repeat = XA1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA2_repeat = XA2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        XA3_repeat = XA3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X1A1_repeat = X1A1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X1A2_repeat = X1A2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X1A3_repeat = X1A3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X2_repeat = X2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        R1_Generator = np.sort( W_Gan(XA1_repeat,reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator = np.sort( W_Gan(XA2_repeat,reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R3_Generator = np.sort( W_Gan(XA3_repeat,reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R1_Generator_image = np.sort( W_Gan_image(X1A1_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator_image = np.sort( W_Gan_image(X1A2_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R3_Generator_image = np.sort( W_Gan_image(X1A3_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R1_test[rep_test, :] = R1_Generator.mean()
        R2_test[rep_test, :] = R2_Generator.mean()
        R3_test[rep_test, :] = R3_Generator.mean()
        Cvar75_1_test[rep_test, :] = R1_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_2_test[rep_test, :] = R2_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_3_test[rep_test, :] = R3_Generator[0:round((3*MonteCarlo_size/4))].mean()
        Cvar50_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar50_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar50_3_test[rep_test, :] = R3_Generator[0:round((MonteCarlo_size/2))].mean()
        Cvar25_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar25_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar25_3_test[rep_test, :] = R3_Generator[0:round((MonteCarlo_size/4))].mean()
        Cvar10_1_test[rep_test, :] = R1_Generator[0:round((MonteCarlo_size/10))].mean()
        Cvar10_2_test[rep_test, :] = R2_Generator[0:round((MonteCarlo_size/10))].mean()
        Cvar10_3_test[rep_test, :] = R3_Generator[0:round((MonteCarlo_size/10))].mean()
        R1_test_image[rep_test, :] = R1_Generator_image.mean()
        R2_test_image[rep_test, :] = R2_Generator_image.mean()
        R3_test_image[rep_test, :] = R3_Generator_image.mean()
        Cvar75_1_test_image[rep_test, :] = R1_Generator_image[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_2_test_image[rep_test, :] = R2_Generator_image[0:round((3*MonteCarlo_size/4))].mean()
        Cvar75_3_test_image[rep_test, :] = R3_Generator_image[0:round((3*MonteCarlo_size/4))].mean()
        Cvar50_1_test_image[rep_test, :] = R1_Generator_image[0:round((MonteCarlo_size/2))].mean()
        Cvar50_2_test_image[rep_test, :] = R2_Generator_image[0:round((MonteCarlo_size/2))].mean()
        Cvar50_3_test_image[rep_test, :] = R3_Generator_image[0:round((MonteCarlo_size/2))].mean()
        Cvar25_1_test_image[rep_test, :] = R1_Generator_image[0:round((MonteCarlo_size/4))].mean()
        Cvar25_2_test_image[rep_test, :] = R2_Generator_image[0:round((MonteCarlo_size/4))].mean()
        Cvar25_3_test_image[rep_test, :] = R3_Generator_image[0:round((MonteCarlo_size/4))].mean()
        Cvar10_1_test_image[rep_test, :] = R1_Generator_image[0:round((MonteCarlo_size/10))].mean()
        Cvar10_2_test_image[rep_test, :] = R2_Generator_image[0:round((MonteCarlo_size/10))].mean()
        Cvar10_3_test_image[rep_test, :] = R3_Generator_image[0:round((MonteCarlo_size/10))].mean()
    
    RRR_test = np.hstack((R1_test, R2_test, R3_test))
    Cvar75_test = np.hstack((Cvar75_1_test, Cvar75_2_test, Cvar75_3_test))
    Cvar50_test = np.hstack((Cvar50_1_test, Cvar50_2_test, Cvar50_3_test))
    Cvar25_test = np.hstack((Cvar25_1_test, Cvar25_2_test, Cvar25_3_test))
    Cvar10_test = np.hstack((Cvar10_1_test, Cvar10_2_test, Cvar10_3_test))
    RRR_test_image = np.hstack((R1_test_image, R2_test_image, R3_test_image))
    Cvar75_test_image = np.hstack((Cvar75_1_test_image, Cvar75_2_test_image, Cvar75_3_test_image))
    Cvar50_test_image = np.hstack((Cvar50_1_test_image, Cvar50_2_test_image, Cvar50_3_test_image))
    Cvar25_test_image = np.hstack((Cvar25_1_test_image, Cvar25_2_test_image, Cvar25_3_test_image))
    Cvar10_test_image = np.hstack((Cvar10_1_test_image, Cvar10_2_test_image, Cvar10_3_test_image))

    Mean_Est_ITR = (np.argmax(RRR_test, axis=1) + 1).reshape(-1, 1)
    Cvar75_Est_ITR = (np.argmax(Cvar75_test, axis=1) + 1).reshape(-1, 1)
    Cvar50_Est_ITR = (np.argmax(Cvar50_test, axis=1) + 1).reshape(-1, 1)
    Cvar25_Est_ITR = (np.argmax(Cvar25_test, axis=1) + 1).reshape(-1, 1)
    Cvar10_Est_ITR = (np.argmax(Cvar10_test, axis=1) + 1).reshape(-1, 1)
    Mean_Est_ITR_image = (np.argmax(RRR_test_image, axis=1) + 1).reshape(-1, 1)
    Cvar75_Est_ITR_image = (np.argmax(Cvar75_test_image, axis=1) + 1).reshape(-1, 1)
    Cvar50_Est_ITR_image = (np.argmax(Cvar50_test_image, axis=1) + 1).reshape(-1, 1)
    Cvar25_Est_ITR_image = (np.argmax(Cvar25_test_image, axis=1) + 1).reshape(-1, 1)
    Cvar10_Est_ITR_image = (np.argmax(Cvar10_test_image, axis=1) + 1).reshape(-1, 1)
    
    Index_Mean_Est = np.where(Mean_Est_ITR - A_test_label == 0)[0]
    Index_Cvar75_Est = np.where(Cvar75_Est_ITR - A_test_label == 0)[0]
    Index_Cvar50_Est = np.where(Cvar50_Est_ITR - A_test_label == 0)[0]
    Index_Cvar25_Est = np.where(Cvar25_Est_ITR - A_test_label == 0)[0]
    Index_Cvar10_Est = np.where(Cvar10_Est_ITR - A_test_label == 0)[0]
    Index_Mean_Est_image = np.where(Mean_Est_ITR_image - A_test_label == 0)[0]
    Index_Cvar75_Est_image = np.where(Cvar75_Est_ITR_image - A_test_label == 0)[0]
    Index_Cvar50_Est_image = np.where(Cvar50_Est_ITR_image - A_test_label == 0)[0]
    Index_Cvar25_Est_image = np.where(Cvar25_Est_ITR_image - A_test_label == 0)[0]
    Index_Cvar10_Est_image = np.where(Cvar10_Est_ITR_image - A_test_label == 0)[0]
    #Index_L1 = np.where( Comp_Est_ITR [0,:].reshape(-1, 1) - A_test_label == 0)[0]
    #Index_AD = np.where( Comp_Est_ITR [1,:].reshape(-1, 1) - A_test_label == 0)[0]
    #Index_RD = np.where( Comp_Est_ITR [2,:].reshape(-1, 1) - A_test_label == 0)[0]
    #Index_SD = np.where( Comp_Est_ITR [3,:].reshape(-1, 1) - A_test_label == 0)[0]
    
    mean_list = [ torch.mean(R_test[Index_Mean_Est]).item(), torch.mean(R_test[Index_Cvar75_Est]).item(), 
                  torch.mean(R_test[Index_Cvar50_Est]).item(), torch.mean(R_test[Index_Cvar25_Est]).item(), 
                  torch.mean(R_test[Index_Cvar10_Est]).item()]
    mean_list_image = [ torch.mean(R_test[Index_Mean_Est_image]).item(), torch.mean(R_test[Index_Cvar75_Est_image]).item(), 
                  torch.mean(R_test[Index_Cvar50_Est_image]).item(), torch.mean(R_test[Index_Cvar25_Est_image]).item(), 
                  torch.mean(R_test[Index_Cvar10_Est_image]).item()]
                  #torch.mean(R_test[Index_L1]).item(), torch.mean(R_test[Index_AD]).item(), 
                  #torch.mean(R_test[Index_RD]).item(), torch.mean(R_test[Index_SD]).item()]
    mean_array = np.array(mean_list).reshape(1, -1)
    mean_array_image = np.array(mean_list_image).reshape(1, -1)
    return {'mean_array':mean_array, 'mean_array_image':mean_array_image, 'Mean_Est_ITR':Mean_Est_ITR,'Cvar75_Est_ITR':Cvar75_Est_ITR,
            'Cvar50_Est_ITR':Cvar50_Est_ITR, 'Cvar25_Est_ITR':Cvar25_Est_ITR, 'Cvar10_Est_ITR':Cvar10_Est_ITR,
            'Mean_Est_ITR_image':Mean_Est_ITR_image, 'Cvar75_Est_ITR_image':Cvar75_Est_ITR_image,
            'Cvar50_Est_ITR_image':Cvar50_Est_ITR_image, 'Cvar25_Est_ITR_image':Cvar25_Est_ITR_image, 'Cvar10_Est_ITR_image':Cvar10_Est_ITR_image}

def prob_ITR():
    np.random.seed(rep_index)
    torch.manual_seed(rep_index)
    torch.cuda.manual_seed_all(rep_index)

    Y_baseline_test = np.array(df_test["MMSCORE"]).reshape(-1, 1)
    # X_test = torch.tensor(Data['X_test'], dtype=torch.float32).to(device)
    X1_test = torch.tensor(Data['X1_test'], dtype=torch.float32).to(device)
    X2_test = torch.tensor(Data['X2_test'], dtype=torch.float32).to(device)
    # XA1_test = torch.cat((torch.full((test_size, 2), 0, device=device), X_test), dim=1)
    # XA2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 1), 0, device=device), X_test), dim=1)
    # XA3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), X_test), dim=1)
    X1A1_test = torch.cat((torch.full((test_size, 2), 0, device=device), X1_test), dim=1)
    X1A2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 1), 0, device=device), X1_test), dim=1)
    X1A3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), X1_test), dim=1)
    reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , dtype=torch.float32, device=device)
    
    Y1_Generator = np.zeros((test_size,MonteCarlo_size))
    Y2_Generator = np.zeros((test_size,MonteCarlo_size))
    Y3_Generator = np.zeros((test_size,MonteCarlo_size))
    Y1_Generator_image = np.zeros((test_size,MonteCarlo_size))
    Y2_Generator_image = np.zeros((test_size,MonteCarlo_size))
    Y3_Generator_image = np.zeros((test_size,MonteCarlo_size))
    for rep_test in range(test_size):
        
        # XA1_repeat = XA1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        # XA2_repeat = XA2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        # XA3_repeat = XA3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X1A1_repeat = X1A1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X1A2_repeat = X1A2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X1A3_repeat = X1A3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        X2_repeat = X2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
        
        R1_Generator_image = np.sort( W_Gan_image(X1A1_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R2_Generator_image = np.sort( W_Gan_image(X1A2_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R3_Generator_image = np.sort( W_Gan_image(X1A3_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
        R1_Generator = R1_Generator_image
        R2_Generator = R2_Generator_image
        R3_Generator = R3_Generator_image

        Y1_Generator[rep_test,:] = (R1_Generator * 21) - 16 + Y_baseline_test[rep_test] 
        Y2_Generator[rep_test,:] = (R2_Generator * 21) - 16 + Y_baseline_test[rep_test]
        Y3_Generator[rep_test,:] = (R3_Generator * 21) - 16 + Y_baseline_test[rep_test]
        Y1_Generator_image[rep_test,:] = (R1_Generator_image * 21) - 16 + Y_baseline_test[rep_test] 
        Y2_Generator_image[rep_test,:] = (R2_Generator_image * 21) - 16 + Y_baseline_test[rep_test]
        Y3_Generator_image[rep_test,:] = (R3_Generator_image * 21) - 16 + Y_baseline_test[rep_test]

    Prob_Generator_A1_200 = np.mean(Y1_Generator <= 24, axis=1)
    Prob_Generator_A2_200 = np.mean(Y2_Generator <= 24, axis=1)
    Prob_Generator_A3_200 = np.mean(Y3_Generator <= 24, axis=1)
    Prob_Generator_200 = np.vstack((Prob_Generator_A1_200, Prob_Generator_A2_200, Prob_Generator_A3_200))
    Prob_Generator_A1_500 = np.mean(Y1_Generator <= 27, axis=1)
    Prob_Generator_A2_500 = np.mean(Y2_Generator <= 27, axis=1)
    Prob_Generator_A3_500 = np.mean(Y3_Generator <= 27, axis=1)
    Prob_Generator_500 = np.vstack((Prob_Generator_A1_500, Prob_Generator_A2_500, Prob_Generator_A3_500))
    Prob_Generator_A1_200_image = np.mean(Y1_Generator_image <= 24, axis=1)
    Prob_Generator_A2_200_image = np.mean(Y2_Generator_image <= 24, axis=1)
    Prob_Generator_A3_200_image = np.mean(Y3_Generator_image <= 24, axis=1)
    Prob_Generator_200_image = np.vstack((Prob_Generator_A1_200_image, Prob_Generator_A2_200_image, Prob_Generator_A3_200_image))
    Prob_Generator_A1_500_image = np.mean(Y1_Generator_image <= 27, axis=1)
    Prob_Generator_A2_500_image = np.mean(Y2_Generator_image <= 27, axis=1)
    Prob_Generator_A3_500_image = np.mean(Y3_Generator_image <= 27, axis=1)
    Prob_Generator_500_image = np.vstack((Prob_Generator_A1_500_image, Prob_Generator_A2_500_image, Prob_Generator_A3_500_image))
    Mean_Est_ITR = test_results['Mean_Est_ITR']
    Cvar75_Est_ITR = test_results['Cvar75_Est_ITR']
    Cvar50_Est_ITR = test_results['Cvar50_Est_ITR']
    Cvar25_Est_ITR = test_results['Cvar25_Est_ITR']
    Cvar10_Est_ITR = test_results['Cvar10_Est_ITR']
    Mean_Est_ITR_image = test_results['Mean_Est_ITR_image']
    Cvar75_Est_ITR_image = test_results['Cvar75_Est_ITR_image']
    Cvar50_Est_ITR_image = test_results['Cvar50_Est_ITR_image']
    Cvar25_Est_ITR_image = test_results['Cvar25_Est_ITR_image']
    Cvar10_Est_ITR_image = test_results['Cvar10_Est_ITR_image']
    #Q_Est_ITR = Comp_Est_ITR [0,:]
    #AD_Est_ITR = Comp_Est_ITR [1,:]
    #RD_Est_ITR = Comp_Est_ITR [2,:]
    #SD_Est_ITR = Comp_Est_ITR [3,:]
    Mean_Est_Prob_200 = np.mean(Prob_Generator_200[(Mean_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar75_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar50_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar25_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Prob_200 = np.mean(Prob_Generator_200[(Cvar10_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Mean_Est_Prob_200_image = np.mean(Prob_Generator_200_image[(Mean_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Prob_200_image = np.mean(Prob_Generator_200_image[(Cvar75_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Prob_200_image = np.mean(Prob_Generator_200_image[(Cvar50_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Prob_200_image = np.mean(Prob_Generator_200_image[(Cvar25_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Prob_200_image = np.mean(Prob_Generator_200_image[(Cvar10_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    #Q_Est_Prob_200 = np.mean(Prob_Generator_200[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #AD_Est_Prob_200 = np.mean(Prob_Generator_200[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #RD_Est_Prob_200 = np.mean(Prob_Generator_200[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #SD_Est_Prob_200 = np.mean(Prob_Generator_200[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Prob_200 = np.array([Mean_Est_Prob_200, Cvar75_Est_Prob_200, Cvar50_Est_Prob_200, 
                             Cvar25_Est_Prob_200, Cvar10_Est_Prob_200]) 
    Est_Prob_200_image = np.array([Mean_Est_Prob_200_image, Cvar75_Est_Prob_200_image, Cvar50_Est_Prob_200_image, 
                             Cvar25_Est_Prob_200_image, Cvar10_Est_Prob_200_image]) 
                              #Q_Est_Prob_200, AD_Est_Prob_200, RD_Est_Prob_200, SD_Est_Prob_200])
               
    Mean_Est_Prob_500 = np.mean(Prob_Generator_500[(Mean_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar75_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar50_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar25_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Prob_500 = np.mean(Prob_Generator_500[(Cvar10_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Mean_Est_Prob_500_image = np.mean(Prob_Generator_500_image[(Mean_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Prob_500_image = np.mean(Prob_Generator_500_image[(Cvar75_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Prob_500_image = np.mean(Prob_Generator_500_image[(Cvar50_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Prob_500_image = np.mean(Prob_Generator_500_image[(Cvar25_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Prob_500_image = np.mean(Prob_Generator_500_image[(Cvar10_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    #Q_Est_Prob_500 = np.mean(Prob_Generator_500[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #AD_Est_Prob_500 = np.mean(Prob_Generator_500[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #RD_Est_Prob_500 = np.mean(Prob_Generator_500[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #SD_Est_Prob_500 = np.mean(Prob_Generator_500[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Prob_500 = np.array([Mean_Est_Prob_500, Cvar75_Est_Prob_500, Cvar50_Est_Prob_500, 
                             Cvar25_Est_Prob_500, Cvar10_Est_Prob_500])
    Est_Prob_500_image = np.array([Mean_Est_Prob_500_image, Cvar75_Est_Prob_500_image, Cvar50_Est_Prob_500_image, 
                             Cvar25_Est_Prob_500_image, Cvar10_Est_Prob_500_image])
                              #Q_Est_Prob_500, AD_Est_Prob_500, RD_Est_Prob_500, SD_Est_Prob_500])
    Est_Prob =  np.vstack((Est_Prob_200,Est_Prob_500,Est_Prob_200_image,Est_Prob_500_image))
    
    Y1_200  = np.sum(Y1_Generator  * (Y1_Generator  <= 24), axis=1) / np.sum((Y1_Generator  <= 24), axis=1)
    Y2_200  = np.sum(Y2_Generator  * (Y2_Generator  <= 24), axis=1) / np.sum((Y2_Generator  <= 24), axis=1)
    Y3_200  = np.sum(Y3_Generator  * (Y3_Generator  <= 24), axis=1) / np.sum((Y3_Generator  <= 24), axis=1)
    Y_200  = np.vstack((Y1_200 , Y2_200 , Y3_200 ))
    Y_200  = np.where(np.isnan(Y_200 ), 24, Y_200 )
    Y1_500  = np.sum(Y1_Generator  * (Y1_Generator <= 27), axis=1) / np.sum((Y1_Generator  <= 27), axis=1)
    Y2_500  = np.sum(Y2_Generator  * (Y2_Generator  <= 27), axis=1) / np.sum((Y2_Generator  <= 27), axis=1)
    Y3_500  = np.sum(Y3_Generator  * (Y3_Generator  <= 27), axis=1) / np.sum((Y3_Generator  <= 27), axis=1)
    Y_500  = np.vstack((Y1_500 , Y2_500 , Y3_500 ))
    Y_500  = np.where(np.isnan(Y_500 ), 27, Y_500 )
    Mean_Est_Y_200  = np.mean(Y_200 [(Mean_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Y_200  = np.mean(Y_200 [(Cvar75_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Y_200  = np.mean(Y_200 [(Cvar50_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Y_200  = np.mean(Y_200 [(Cvar25_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Y_200  = np.mean(Y_200 [(Cvar10_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Est_Y_200  = np.array([Mean_Est_Y_200 , Cvar75_Est_Y_200 , Cvar50_Est_Y_200 , 
                          Cvar25_Est_Y_200 , Cvar10_Est_Y_200 ])
    Mean_Est_Y_500  = np.mean(Y_500 [(Mean_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Y_500  = np.mean(Y_500 [(Cvar75_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Y_500  = np.mean(Y_500 [(Cvar50_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Y_500  = np.mean(Y_500 [(Cvar25_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Y_500  = np.mean(Y_500 [(Cvar10_Est_ITR -1).reshape(-1),np.arange(test_size)])
    Est_Y_500  = np.array([Mean_Est_Y_500 , Cvar75_Est_Y_500 , Cvar50_Est_Y_500 , 
                          Cvar25_Est_Y_500 , Cvar10_Est_Y_500 ])
    
    Y1_200_image = np.sum(Y1_Generator_image * (Y1_Generator_image <= 24), axis=1) / np.sum((Y1_Generator_image <= 24), axis=1)
    Y2_200_image = np.sum(Y2_Generator_image * (Y2_Generator_image <= 24), axis=1) / np.sum((Y2_Generator_image <= 24), axis=1)
    Y3_200_image = np.sum(Y3_Generator_image * (Y3_Generator_image <= 24), axis=1) / np.sum((Y3_Generator_image <= 24), axis=1)
    Y_200_image = np.vstack((Y1_200_image, Y2_200_image, Y3_200_image))
    Y_200_image = np.where(np.isnan(Y_200_image), 24, Y_200_image)
    Y1_500_image = np.sum(Y1_Generator_image * (Y1_Generator_image<= 27), axis=1) / np.sum((Y1_Generator_image <= 27), axis=1)
    Y2_500_image = np.sum(Y2_Generator_image * (Y2_Generator_image <= 27), axis=1) / np.sum((Y2_Generator_image <= 27), axis=1)
    Y3_500_image = np.sum(Y3_Generator_image * (Y3_Generator_image <= 27), axis=1) / np.sum((Y3_Generator_image <= 27), axis=1)
    Y_500_image = np.vstack((Y1_500_image, Y2_500_image, Y3_500_image))
    Y_500_image = np.where(np.isnan(Y_500_image), 27, Y_500_image)
    Mean_Est_Y_200_image = np.mean(Y_200_image[(Mean_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Y_200_image = np.mean(Y_200_image[(Cvar75_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Y_200_image = np.mean(Y_200_image[(Cvar50_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Y_200_image = np.mean(Y_200_image[(Cvar25_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Y_200_image = np.mean(Y_200_image[(Cvar10_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    #Q_Est_Y_200 = np.mean(Y_200[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #AD_Est_Y_200 = np.mean(Y_200[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #RD_Est_Y_200 = np.mean(Y_200[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #SD_Est_Y_200 = np.mean(Y_200[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Y_200_image = np.array([Mean_Est_Y_200_image, Cvar75_Est_Y_200_image, Cvar50_Est_Y_200_image, 
                          Cvar25_Est_Y_200_image, Cvar10_Est_Y_200_image])
                              #Q_Est_Y_200, AD_Est_Y_200, RD_Est_Y_200, SD_Est_Y_200])
    Mean_Est_Y_500_image = np.mean(Y_500_image[(Mean_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar75_Est_Y_500_image = np.mean(Y_500_image[(Cvar75_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar50_Est_Y_500_image = np.mean(Y_500_image[(Cvar50_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar25_Est_Y_500_image = np.mean(Y_500_image[(Cvar25_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    Cvar10_Est_Y_500_image = np.mean(Y_500_image[(Cvar10_Est_ITR_image-1).reshape(-1),np.arange(test_size)])
    #Q_Est_Y_500 = np.mean(Y_500[(Q_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #AD_Est_Y_500 = np.mean(Y_500[(AD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #RD_Est_Y_500 = np.mean(Y_500[(RD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    #SD_Est_Y_500 = np.mean(Y_500[(SD_Est_ITR-1).reshape(-1),np.arange(test_size)])
    Est_Y_500_image = np.array([Mean_Est_Y_500_image, Cvar75_Est_Y_500_image, Cvar50_Est_Y_500_image, 
                          Cvar25_Est_Y_500_image, Cvar10_Est_Y_500_image])
                              #Q_Est_Y_500, AD_Est_Y_500, RD_Est_Y_500, SD_Est_Y_500])
    Est_Y =  np.vstack((Est_Y_200,Est_Y_500,Est_Y_200_image,Est_Y_500_image))
    return {'Est_Prob':Est_Prob, 'Est_Y':Est_Y}

# def plot_ITR():
#     np.random.seed(rep_index)
#     torch.manual_seed(rep_index)
#     torch.cuda.manual_seed_all(rep_index)
    
#     Y = np.array(df_train["diff_score"]).reshape(-1, 1)
#     A = df_train[["KEYMED_1", "KEYMED_2"]].to_numpy()
#     X_test = torch.tensor(Data['X_test'], dtype=torch.float32).to(device)
#     X1_test = torch.tensor(Data['X1_test'], dtype=torch.float32).to(device)
#     X2_test = torch.tensor(Data['X2_test'], dtype=torch.float32).to(device)
#     XA1_test = torch.cat((torch.full((test_size, 2), 0, device=device), X_test), dim=1)
#     XA2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 1), 0, device=device), X_test), dim=1)
#     XA3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), X_test), dim=1)
#     X1A1_test = torch.cat((torch.full((test_size, 2), 0, device=device), X1_test), dim=1)
#     X1A2_test = torch.cat((torch.full((test_size, 1), 1, device=device), torch.full((test_size, 1), 0, device=device), X1_test), dim=1)
#     X1A3_test = torch.cat((torch.full((test_size, 1), 0, device=device), torch.full((test_size, 1), 1, device=device), X1_test), dim=1)
#     reference_MonteCarlo = torch.tensor( np.random.uniform(-1, 1, size=(MonteCarlo_size,reference_dimension) ) , 
#                                         dtype=torch.float32, device=device)
#     R1_Generator = np.zeros((test_size, MonteCarlo_size))
#     R2_Generator = np.zeros((test_size, MonteCarlo_size))
#     R3_Generator = np.zeros((test_size, MonteCarlo_size))
#     R1_Generator_image = np.zeros((test_size,MonteCarlo_size))
#     R2_Generator_image = np.zeros((test_size,MonteCarlo_size))
#     R3_Generator_image = np.zeros((test_size,MonteCarlo_size))
    
#     for rep_test in range(test_size):
#         XA1_repeat = XA1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
#         XA2_repeat = XA2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
#         XA3_repeat = XA3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
#         X1A1_repeat = X1A1_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
#         X1A2_repeat = X1A2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
#         X1A3_repeat = X1A3_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
#         X2_repeat = X2_test[rep_test, :].unsqueeze(0).repeat(MonteCarlo_size, 1)
#         R1_Generator[rep_test,:] = np.sort( W_Gan(XA1_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
#         R2_Generator[rep_test,:] = np.sort( W_Gan(XA2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
#         R3_Generator[rep_test,:] = np.sort( W_Gan(XA3_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
#         R1_Generator_image[rep_test,:] = np.sort( W_Gan_image(X1A1_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
#         R2_Generator_image[rep_test,:] = np.sort( W_Gan_image(X1A2_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )
#         R3_Generator_image[rep_test,:] = np.sort( W_Gan_image(X1A3_repeat,X2_repeat, reference_MonteCarlo).to("cpu").detach().numpy().flatten() )

#     Index_A1 = np.where((A[:,0]==0.) & (A[:,1]==0.))[0]
#     Index_A2 = np.where((A[:,0]==1.) & (A[:,1]==0.))[0]
#     Index_A3 = np.where((A[:,0]==0.) & (A[:,1]==1.))[0]
#     kde_YA1 = gaussian_kde(Y[Index_A1].flatten())
#     kde_YA2 = gaussian_kde(Y[Index_A2].flatten())
#     kde_YA3 = gaussian_kde(Y[Index_A3].flatten())
#     kde_GR1 = gaussian_kde(R1_Generator.flatten())
#     kde_GR2 = gaussian_kde(R2_Generator.flatten())
#     kde_GR3 = gaussian_kde(R3_Generator.flatten())
#     kde_GR1_image = gaussian_kde(R1_Generator_image.flatten())
#     kde_GR2_image = gaussian_kde(R2_Generator_image.flatten())
#     kde_GR3_image = gaussian_kde(R3_Generator_image.flatten())
#     range_YA1 = np.linspace( Y[Index_A1].min(), Y[Index_A1].max(), 1000)
#     range_YA2 = np.linspace( Y[Index_A2].min(), Y[Index_A2].max(), 1000)
#     range_YA3 = np.linspace( Y[Index_A3].min(), Y[Index_A3].max(), 1000)

#     fig, axs = plt.subplots(1, 3, figsize=(15,5))
#     axs[0].plot(range_YA1, kde_YA1(range_YA1), 'k-', label='Observed')
#     axs[0].plot(range_YA1, kde_GR1(range_YA1), 'r--', label='Generated (X)')
#     axs[0].plot(range_YA1, kde_GR1_image(range_YA1), 'b:', label='Generated (Image)')
#     axs[0].set_title('A = 1')
#     axs[0].grid(True)
#     axs[1].plot(range_YA2, kde_YA2(range_YA2), 'k-')
#     axs[1].plot(range_YA2, kde_GR2(range_YA2), 'r--')
#     axs[1].plot(range_YA2, kde_GR2_image(range_YA2), 'b:')
#     axs[1].set_title('A = 2')
#     axs[1].grid(True)
#     axs[2].plot(range_YA3, kde_YA3(range_YA3), 'k-')
#     axs[2].plot(range_YA3, kde_GR3(range_YA3), 'r--')
#     axs[2].plot(range_YA3, kde_GR3_image(range_YA3), 'b:')
#     axs[2].set_title('A = 3')
#     axs[2].grid(True)
#     plt.legend()
#     plt.suptitle("Kernel Density: Observed vs Generated", y=1.02)
#     plt.savefig( "ADNI_CGlearning.png")
#     plt.tight_layout()
#     plt.show()

Data = generate_data()
W_Gan = train_generator().to(device)
W_Gan_image = train_generator_image().to(device)
# W_Gan = DCG().to(device)
# W_Gan.load_state_dict(torch.load('W_Gan.pth', map_location=device))
# W_Gan_image = DCG_image().to(device)
# W_Gan_image.load_state_dict(torch.load('W_Gan_image.pth', map_location=device))
test_results = test_ITR()
prob_results = prob_ITR()
np.savetxt(f"Ref{reference_dimension}_WEG{G_width_image}_WD{D_width_image}_lr{LearningR_image}EST_result_CG.csv", test_results['mean_array'], delimiter=",")
np.savetxt(f"Ref{reference_dimension}_WEG{G_width_image}_WD{D_width_image}_lr{LearningR_image}EST_result_UCG.csv", test_results['mean_array_image'], delimiter=",")
np.savetxt(f"Ref{reference_dimension}_WEG{G_width_image}_WD{D_width_image}_lr{LearningR_image}prob.csv", prob_results['Est_Prob'], delimiter=",")
np.savetxt(f"Ref{reference_dimension}_WEG{G_width_image}_WD{D_width_image}_lr{LearningR_image}CondY.csv", prob_results['Est_Y'], delimiter=",")
