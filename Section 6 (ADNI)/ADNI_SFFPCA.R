########################
#### load libraries ####
########################
library("fda")
library("readxl")
library("Matrix")
library("openxlsx")
#####################################
#### Some functions for iteration ###
#####################################

X_minus_AH <- function(n, X, A, H) { 
  X_minus_AH <- array(0, c(n, p, T) )
  for (t in 1 : T) X_minus_AH[, , t] <- X[, , t] - H[, , t] %*% t(A)
  return(X_minus_AH)
}

AB_array_multi <- function(A, B) {
  AB <- matrix(0, length(A[1, , 1]), length(B[1, , 1]) )
  for (t in 1 : T) AB <- AB + t(A[, , t]) %*% B[, , t]
  return(AB)
}

Alpha_ite <- function(n, X, B, H, HH_inv, Bs, BBs_inv) {
  X_minus_BH <- X_minus_AH(n, X, B, H)
  XH_BH <- AB_array_multi(X_minus_BH, H) 
  Alpha <- HH_inv %*% t(XH_BH) %*% Bs %*% BBs_inv
  return(Alpha)
}

Theta_score_ite <- function(Wl, K) {
  # Wl = W[,,l] n x tau_Bt
  Wl_svd <- svd( t(Wl) %*% Wl )
  Theta_q_l <- t( Wl_svd$u[, 1 : K] ) 
  score_n_l <- Wl %*% t(Theta_q_l)
  pro_l <- sum( (Wl_svd$d)[1 : K] ) / sum( (Wl_svd$d) )
  return(list(Theta_q_l, score_n_l, pro_l) )
}
##########################################################
##### Read ROI data from previous trained xlsx  ##########
##########################################################

traindir  <- "out_ADNI"
testdir  <- "out_ADNI_test"
trainfiles  <- list.files(traindir, pattern = "\\.csv$", full.names = TRUE)
testfiles  <- list.files(testdir, pattern = "\\.csv$", full.names = TRUE)
train_subjid <- sub("__lqdt\\.csv$", "", basename(trainfiles))
test_subjid <- sub("__lqdt\\.csv$", "", basename(testfiles))
train_n <- length(train_subjid) 
test_n <- length(test_subjid) 
xyz <- as.matrix(read_excel("ROI_center_mat.xlsx",col_names = FALSE))
p <- nrow(xyz)
time <- seq(0.01, 0.99, by = 0.02)
T <- length(time)
lambda1 <- 0.7
lambda2 <- 12
mu <- 0.001
step_max <- 5
B_step <- 2
train_X <- array(NA_real_, dim = c(train_n, p, T))
test_X <- array(NA_real_, dim = c(test_n, p, T))
for (i in 1:train_n) {train_X[i, , ] <- as.matrix(read.csv(trainfiles[i], header = FALSE))}
for (i in 1:test_n) {test_X[i, , ] <- as.matrix(read.csv(testfiles[i], header = FALSE))}
### remove mean ### 
train_X_mean <- matrix(0, p, T) 
for (t in 1 : T) train_X_mean[, t] <- colMeans(train_X[, ,t]) 
for (i in 1 : train_n) train_X[i, ,] <- train_X[i, , ] - train_X_mean 
test_X_mean <- matrix(0, p, T) 
for (t in 1 : T) test_X_mean[, t] <- colMeans(test_X[, ,t]) 
for (i in 1 : test_n) test_X[i, ,] <- test_X[i, , ] - test_X_mean 

##########################################################
#### generate basic functions M(s) to approximate f(s) ###
##########################################################
tau_Bs_k <- 4
break_num_s <- 3 
tau_Bs <- tau_Bs_k^3 ## number of elments of M(s): \tau
                     ## xyz: 3-dimensional spatial coordinates
                     ## p: number of ROIs
Bs1 <- bsplineS(x = xyz[, 1],seq(0, 1, length = break_num_s), tau_Bs_k + 2 - break_num_s)
Bs2 <- bsplineS(x = xyz[, 2],seq(0, 1, length = break_num_s), tau_Bs_k + 2 - break_num_s)
Bs3 <- bsplineS(x = xyz[, 3],seq(0, 1, length = break_num_s), tau_Bs_k + 2 - break_num_s)
Bs <- matrix(t( KhatriRao( KhatriRao(t(Bs1),t(Bs2)), t(Bs3) ) ) , p, tau_Bs )
Bs <- (svd( Bs %*% t(Bs) ))$u[, 1 : tau_Bs] * sqrt(p)
for (k in 1 : tau_Bs) Bs[, k] <- sign(Bs[1, k]) * Bs[, k]
BBs_inv <- solve( t(Bs) %*% Bs ) 


############################################################
#### generate splines functions Z(t) to estimate \phi(t) ###
############################################################

break_num_t <- 5  
tau_Bt <- 5  ## number of elments of Z(t): \omega
             ## time: quantile            
             ## T: number of quantile                
Bt <- bsplineS(time,breaks = seq(0, 1, length = break_num_t), norder = tau_Bt + 2 - break_num_t)
Bt <- (svd( Bt %*% t(Bt) ))$u[, 1 : tau_Bt] * sqrt(T)
for (k in 1 : tau_Bt) Bt[, k] <- sign(Bt[1, k]) * Bt[, k]
BBt_inv <- solve( t(Bt) %*% Bt ) 
q <- 3
K <- 2 

SFFPCA <- function(X, n, p, time, T, xyz, q, K, lambda1, lambda2, mu,
                   tau_Bt, step_max, B_step, Bs, Bt, BBs_inv, BBt_inv) {
  # X : functional data
  X_SFFPCA = X
  # n: sample size
  # p: dimension of X
  # time : quantile 
  # T: number of time points for each individual
  # xyz: 3-dimensional spatial coordinates
  # q: dimension of loadings
  # K: number of eigenfunctions
  # num.spline : number of spline basics
  # lambda1 : tunning parameter for group lasso
  # lambda2 : tunning parameter for pairwise lasso
  # mu: pre-given hyperparameter of the quadratic term in ADMM to update B
  # tau_Bt <- 5  ## number of elments of Z(t)
  # step_max: maximum number of iterations of SFFPCA
  # B_step: maximum number of ADMM to update B
  
  #####################################
  ### initial values for B, f(s), H ###
  #####################################
  XX <- AB_array_multi(X_SFFPCA, X_SFFPCA)
  XX_svd <- svd(XX)
  Bf0 <- XX_svd$u[, 1 : q] * sqrt(p)    
  Alpha0 <- t( BBs_inv %*% t(Bs) %*% Bf0 )
  Alpha0 <- t((svd( t(Alpha0) %*% Alpha0 ))$u[, 1 : q]) 
  f0 <- Bs %*% t(Alpha0)
  B0 <- Bf0 - f0
  H0 <- array(0, c(n, q, T))
  for (i in 1 : n) H0[i, , ] <- t(Bf0) %*% X_SFFPCA[i, , ] / p 
  
  
  ### weight for pairwise lasso ###
  Weight_s <- matrix(0, p, p)
  for (j in 1 : p) Weight_s[j, -j] <- apply(xyz[-j, ], 1, function(x) return(1/norm(x - xyz[j, ], type="2")) )
  Weight_s <- t(Weight_s) 
  
  
  ### set initial value ###
  B <- B0
  f <- f0
  H <- H0
  lambda1 <- lambda1
  lambda2 <- lambda2
  
  
  ### initial value for ADMM to update B ###
  D <- kronecker(diag(p) * (p) - matrix(rep(1, p*p), p, p), diag(q) )
  E <- rep(0, p*q)
  
  mu <- mu
  B_minus_B <-  array(0, c(p, p, q))
  for (j in 1 : p ) B_minus_B[, j, ]  <- t(apply(B, 1, function(x) x - B[j, ] ))  
  Gam <- B_minus_B
  nu <- array(0, c(p, p, q)) 
  
  
  ######################
  ### iteration step ###
  ######################
  W <- array(0, c(n, tau_Bt, q) )
  phi <- array(0,c(q, K, T) )
  Theta_q <- array(0, c(K, tau_Bt, q) )
  score_n <- array(0, c(n, q, K) )
  for (k in 1 : step_max) {
    wtH <- matrix(0, p * q, p * q)
    for (j in 1 : p) wtH[ ((j-1)*q+1) : (j*q) , ((j-1)*q+1) : (j*q) ] <- AB_array_multi(H, H) 
    X_minus_fH <- X_minus_AH(n, X_SFFPCA, f, H)
    XH_fH <- AB_array_multi(X_minus_fH, H) 
    wtH_X_minus_fH  <- c(t(XH_fH))
    
    HH_inv <- solve( AB_array_multi(H, H) )
    
    ############################
    ### update for B by (11) ###
    ############################
    for (l in 1 : B_step) {
      
      for (j in 1 : p) E[ ((j-1)*q+1)  :  (j*q) ] <- colSums(nu[j, -j,] - nu[-j, j, ] - Gam[j, -j, ] + Gam[-j, j, ]) 
      
      ## upadate wtB by (S5) in Suppl. B ##
      wtB <- solve(2 * wtH + mu * D/10) %*% (2 * wtH_X_minus_fH - E/10 ) 
      
      ## upadate B by (S5) in Suppl. B ##
      B <- t(matrix(wtB, q, p))
      B <- t(apply(B, 1, function(x) ifelse( (norm(x,type = "2")-lambda1) > 0, return((norm(x,type="2")-lambda1) * x / norm(x,type="2")), return(rep(0, q)) )) )
      
      ## update Gamma by (S5) in Suppl. B ##
      for (j in 1 : p ) B_minus_B[, j, ] <- t(apply(B, 1, function(x) x - B[j, ] ))  
      Gam <- nu / mu + B_minus_B
      for (j in 1 : p) {
        for (j1 in 1 : p) { 
          if (norm(c(Gam[j, j1, ]),type = "2") - lambda2*Weight_s[j, j1] > 0) { Gam[j, j1, ] <- (norm(Gam[j, j1, ],type = "2")-lambda2*Weight_s[j, j1]) * Gam[j, j1, ] / norm(Gam[j, j1, ],type = "2") }
          else {Gam[j, j1, ] <- rep(0,q)}
        }
      }
      
      ## update nu by (S5) in Suppl. B ###
      nu <- nu + mu * (B_minus_B - Gam)/10
    }
    
    ############################
    ### update f(s) by (12) ###
    ############################
    Alpha <- Alpha_ite(n, X_SFFPCA, B, H, HH_inv, Bs, BBs_inv)      
    f <- Bs %*% t(Alpha)
    for (l in 1 : q) f[, l] <- sign(f[1, l]) * f[, l]
    
    #######################
    ### update h by (9) ###
    #######################   
    for (t in 1 : T) H[, , t] <- X_SFFPCA[, , t] %*% (B+f) %*% solve( t(B+f) %*% (B+f) )
    
    ###############################################
    ### update \kesi and \phi(t) by (13) and 14 ###
    ###############################################  
    for (i in 1 : n) W[i, , ] <-  BBt_inv %*% t(Bt) %*% t(H[i, , ])
    for (l in 1 : q) {
      Theta_score_result_l <- Theta_score_ite(W[, , l], K)
      Theta_q[, , l] <- Theta_score_result_l[[1]]   
      score_n[, l, ] <- Theta_score_result_l[[2]]     
      phi[l, , ] <- t( Bt %*% t( Theta_q[, , l] ))
      H[, l, ] <-  score_n[, l, ] %*% phi[l, , ]
    }
    
  }
  
  return( list(B = B, f = f, Alpha = Alpha, Bs = Bs, 
               score_n = score_n, phi = phi, Theta_q = Theta_q, Bt = Bt) )
  
}

train_SFFPCA_result = SFFPCA(train_X, train_n, p, time, T, xyz, q, K, lambda1, lambda2, mu,
                       tau_Bt, step_max, B_step, Bs, Bt, BBs_inv, BBt_inv)

test_SFFPCA_result = SFFPCA(test_X, test_n, p, time, T, xyz, q, K, lambda1, lambda2, mu,
                       tau_Bt, step_max, B_step, Bs, Bt, BBs_inv, BBt_inv)

train_lowdim_score = train_SFFPCA_result$score_n
train_zeta <- matrix(NA_real_, train_n, q*K)
for (i in 1:train_n) { train_zeta[i, ] <- c(t(train_lowdim_score[i, , ])) }

test_lowdim_score = test_SFFPCA_result$score_n
test_zeta <- matrix(NA_real_, test_n, q*K)
for (i in 1:test_n) { test_zeta[i, ] <- c(t(test_lowdim_score[i, , ])) }

train_split_id <- strsplit(train_subjid, "__")
train_SubjectID <- sapply(train_split_id, `[`, 1)
train_ImageID   <- sapply(train_split_id, `[`, 2)
test_split_id <- strsplit(test_subjid, "__")
test_SubjectID <- sapply(test_split_id, `[`, 1)
test_ImageID   <- sapply(test_split_id, `[`, 2)

save(xyz, train_SubjectID, train_ImageID, train_X, train_zeta, test_SubjectID, test_ImageID, test_X, test_zeta, file="ADNI_ROI_data.RData")
