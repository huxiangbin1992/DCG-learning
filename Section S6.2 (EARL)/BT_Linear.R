library(rlist)
library(caret)
library(Matrix)
library(glmnet)
library(randomForest)
# library(xgboost)
library(tidyverse)
library(MASS)
library(truncnorm)
library(DynTxRegime)

#' Q-Learning
#' \code{qlearn} implements Q-Learning for the multiple treatment scenario with
#'    L1-regularization for parameter estimation
#' @param X A numeric matrix of covariates used for fitting 
#' @param A A numeric vector of treatments (coded {1, ..., K})
#' @param Y A numeric vector of observed outcomes
#' @param prob_A A numeric vector of known or estimated propensity scores, i-th element is Pr(a_i|x_i)
#' @param K Number of treatments
#' @return An object of class \code{cv.glmnet} containing coefficients of the Q-Learning fit

qlearn <- function(X, A, Y) {
  n <- nrow(X)
  p <- ncol(X)
  XA_interaction <- matrix (0,n,p)
  for(i in 1:n) {XA_interaction [i,] <- A[i]*X[i, ]}
  X_modified <- cbind(matrix(1,n,1), A, X, XA_interaction)
  prefit <- cv.glmnet(X_modified, Y, alpha = 1 , standardize.response = T) ### 10-fold CV for selection of lambda
  return(prefit)
}

qtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  prob_mat <- predict(fit, X, s="lambda.min", type="response")[,,1]
  prop_test <- prob_mat[cbind(1:n, 0.5*(3-A))]
  ##### prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  ITR = TITR(X)
  A_modified = rep(c(1,-1),n) 
  Xmat = X[rep(1:n, each = 2), ]
  XA_interaction <- matrix (0,2*n,p)
  for(i in 1:(2*n)) {XA_interaction [i,] <-  A_modified[i]*Xmat[i, ]}
  X_modified <- cbind(matrix(1,2*n,1), A_modified, Xmat, XA_interaction)
  Y_predict = predict(result, X_modified, s = "lambda.min", type = "response")
  Y_predict = t (matrix ( Y_predict, 2 ,  ))
  ITR_predict = apply(Y_predict, 1, which.max)
  ITR_predict = 3 - 2 * ITR_predict
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

#' (A)D-Learning
dlearn <- function(X, A, Y, prob_A) {
  n <- nrow(X)
  p <- ncol(X)
  Y_modified <- 2 * Y * A
  wts <- 1 / prob_A
  fit <- cv.glmnet(x = X, y = Y_modified, weights = wts, alpha = 1)
  return(fit)
}

dtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  prob_mat <- predict(fit, X, s="lambda.min", type="response")[,,1]
  prop_test <- prob_mat[cbind(1:n, 0.5*(3-A))]
  ##### prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  ITR = TITR(X)
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = ifelse(f_predict > 0, 1, -1)
  rate = 1 - mean(ITR_predict == ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

rdlearn <- function(X, A, Y, main_eff, prob_A) {
  n <- nrow(X)
  p <- ncol(X)
  Y_modified <- (Y-main_eff) * A
  wts <- 1 / prob_A
  prefit <- cv.glmnet(X, Y_modified, alpha = 1, standardize.response = T, weights = wts, family = "gaussian")
  return(prefit)
}

rdtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  ITR = TITR(X)
  prob_mat <- predict(fit, X, s="lambda.min", type="response")[,,1]
  prop_test <- prob_mat[cbind(1:n, 0.5*(3-A))]
  ##### prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = ifelse(f_predict > 0, 1, -1)
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

sdlearn <- function(X, A, Y, prob_A) {
  fit <- dlearn(X, A, Y, prob_A)
  Y_modified <- Y * A
  b <- coef(fit, "lambda.min")
  B <- as.matrix(list.cbind(b))
  X_int <- cbind(1,X) 
  XB <- X_int %*% t(B)
  resid <- 2 * Y * A - XB
  resid_squared <- resid^2
  ############## pred_all <- rowSums(A * XB)
  ############## resid_squared <- (Y - pred_all)^2
  XA <- cbind(X, as.factor(A))
  # Implementation - random forest for residual modeling and resulting SABD-Learning fit
  resid_fit <- randomForest(resid_squared ~ ., data = XA)
  resid_preds <- predict(resid_fit)
  # Modified weights and SABD-Learning fitting step
  wts_new <- 1 / resid_preds 
  fits <- cv.glmnet(X, Y_modified, alpha = 1,standardize.response = T, weights = wts_new, family = "gaussian")
  return(fits)
}

sdtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  ITR = TITR(X)
  prob_mat <- predict(fit, X, s="lambda.min", type="response")[,,1]
  prop_test <- prob_mat[cbind(1:n, 0.5*(3-A))]
  ##### prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = ifelse(f_predict > 0, 1, -1)
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

EARLtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  ITR = TITR(X)
  prob_mat <- predict(fit, X, s="lambda.min", type="response")[,,1]
  prop_test <- prob_mat[cbind(1:n, 0.5*(3-A))]
  ##### prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  EARL_DATAtest = data.frame(X)
  ITR_predict = 2 * optTx(result, EARL_DATAtest)$optimalTx - 1
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

# Generate DATA #######
generate_data <- function (rep_index){
  set.seed(rep_index)
  X = matrix ( rtruncnorm(sample_size * Cov_dim, a = -3, b = 3, mean = 0, sd = 1), sample_size, Cov_dim)
  mu = 2 + apply(X,1,sum)
  PA = 1/(1+exp(X[,1]))
  PA = cbind( PA, 1-PA)
  A = epsilon = Y = rep(0, sample_size)
  for (i in 1:sample_size){A[i] = sample(c(1,-1), size = 1, prob = PA[i,])}
  # epsilon_sd = sqrt(apply(X^2,1,sum))/Cov_dim
  epsilon_sd = rep(0.5,sample_size)
  for (i in 1:sample_size){epsilon[i] = rtruncnorm(1, a = -3*epsilon_sd[i], b = 3*epsilon_sd[i], mean = 0, sd = epsilon_sd[i])}
  Y[which(A==1)] = (mu + X[,1] + X[,2] + epsilon)[which(A==1)]
  Y[which(A==-1)] = (mu - X[,1] - X[,2] + epsilon)[which(A==-1)]
  # normalize the reward Y
  # min_Y= min(Y)
  # max_Y = max(Y)
  # Y_normalized = (Y - min_Y) / (max_Y - min_Y)
  A_train = A[1:train_size]
  X_train = X[1:train_size,]
  Y_train = Y[1:train_size]
  A_test = A[(train_size+1):sample_size]
  X_test = X[(train_size+1):sample_size,]
  Y_test = Y[(train_size+1):sample_size]
  return(list( A_train=A_train, X_train=X_train, Y_train=Y_train, A_test=A_test, X_test=X_test, Y_test=Y_test))
}

TITR <- function(X){
  mu = 2 + apply(X,1,sum)
  EY1 = mu + X[,1] + X[,2]
  EY2 = mu - X[,1] - X[,2]
  EY = cbind(EY1,EY2)
  ITR = apply(EY, 1, which.max)
  ITR = 3 - 2 * ITR
  return(ITR)
}

Tvalue <- function(Mon_size){
  set.seed(0)
  X = matrix ( rtruncnorm(Mon_size * Cov_dim, a = -3, b = 3, mean = 0, sd = 1), Mon_size, Cov_dim)
  mu = 2 + apply(X,1,sum)
  gamma_1 = mu + X[,1] + X[,2]
  gamma_2 = mu - X[,1] - X[,2]
  gamma = cbind(gamma_1,gamma_2) 
  value = mean(apply(gamma,1,max)) 
  return(value)
}

num_rep = 100
train_size = 400
test_size = 500
Cov_dim = 2
sample_size = train_size + test_size
T_value = Tvalue(5000000)
Results = matrix(0, num_rep, 12)

for(rep_index in 1:num_rep){
  data = generate_data(rep_index)
  set.seed(rep_index)
  Multilogistic.fit = cv.glmnet(data$X_train, data$A_train, family = "multinomial", type.multinomial = "grouped")
  prob_A_train = predict(Multilogistic.fit, data$X_train, s = "lambda.min", type = "response")[cbind(1:train_size,0.5*(3 - data$A_train),1)]
  ql_result = qlearn(data$X_train, data$A_train, data$Y_train)
  qtest_result = qtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, ql_result)
  d_result = dlearn(data$X_train, data$A_train, data$Y_train, prob_A_train)
  dtest_result = dtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, d_result)
  Maineffect.fit = cv.glmnet(data$X_train, data$Y_train, weights = 1/prob_A_train, standardize.response = T)
  main_eff = predict(Maineffect.fit, data$X_train, s = "lambda.min", type = "response") [cbind(1:train_size,1)]
  rd_result = rdlearn(data$X_train, data$A_train, data$Y_train, main_eff, prob_A_train)
  rdtest_result = rdtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, rd_result)
  sd_result = sdlearn(data$X_train, data$A_train, data$Y_train, prob_A_train)
  sdtest_result = sdtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, sd_result)
  
  EARL_A = 0.5*(data$A_train + 1)
  EARL_DATA = data.frame( EARL_A, data$Y_train,  data$X_train,data$X_train^2)
  # propensity model
  moPropen <- buildModelObj(model = ~X1+X2, solver.method = 'glm',
                            solver.args = list('family'='binomial'), predict.method = 'predict.glm', predict.args = list(type='response'))
  # cc_outcome model
  cc_moMain <- buildModelObj(model = ~X1.1+X2.1, solver.method = 'lm')
  cc_moCont <- buildModelObj(model = ~X1+X2, solver.method = 'lm')
  # outcome model
  moMain <- buildModelObj(model = ~X1+X2, solver.method = 'lm')
  moCont <- buildModelObj(model = ~X1+X2, solver.method = 'lm')
  cc_fitEARL <- earl(moPropen = moPropen, moMain = cc_moMain, moCont = cc_moCont, data = EARL_DATA, response = data$Y_train, txName = 'EARL_A',
                     regime = ~X1+X2, surrogate = 'hinge', kernel = 'linear', verbose = 0)
  fitEARL <- earl(moPropen = moPropen, moMain = moMain, moCont = moCont, data = EARL_DATA, response = data$Y_train, txName = 'EARL_A',
                  regime = ~X1+X2, surrogate = 'hinge', kernel = 'linear', verbose = 0)
  cc_EARL_result <- EARLtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, cc_fitEARL)
  EARL_result <- EARLtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, fitEARL)
  Results[rep_index,] = c(qtest_result$rate,qtest_result$bias,dtest_result$rate,dtest_result$bias,
                          rdtest_result$rate,rdtest_result$bias,sdtest_result$rate,sdtest_result$bias,
                          cc_EARL_result$rate,cc_EARL_result$bias,EARL_result$rate,EARL_result$bias)
  print(rep_index)
}
write.csv(Results, paste0("Results_",train_size,"linear.csv"), row.names = FALSE)


