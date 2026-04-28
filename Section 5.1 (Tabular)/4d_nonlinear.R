# multi_sd.R

library(rlist)
# library(caret) useless
library(Matrix)
library(glmnet)
library(randomForest)
# library(xgboost)
# library(tidyverse) useless
library(MASS)
library(truncnorm)

#' Q-Learning
#' 
#' \code{qlearn} implements Q-Learning for the multiple treatment scenario with
#'    L1-regularization for parameter estimation
#'    
#' @param X A numeric matrix of covariates used for fitting 
#' @param A A numeric vector of treatments (coded {1, ..., K})
#' @param Y A numeric vector of observed outcomes
#' @param prob_A A numeric vector of known or estimated propensity scores, i-th element is Pr(a_i|x_i)
#' @param K Number of treatments
#' 
#' @return An object of class \code{cv.glmnet} containing coefficients of the Q-Learning fit
#' 
qlearn <- function(X, A, Y, K) {
  n <- nrow(X)
  p <- ncol(X)
  A_modified <- matrix(0,n,K)
  A_modified [cbind(1:n,A)] <- 1
  A_modified = A_modified[, 1:(K-1)]
  XA_interaction <- matrix (0,n,(K-1)*p)
  for(i in 1:n) {XA_interaction [i,] <-  kronecker(A_modified [i, ], X[i, ])}
  X_modified <- cbind(matrix(1,n,1), A_modified, X, XA_interaction)
  prefit <- cv.glmnet(X_modified, Y, alpha = 1 , standardize.response = T) ### 10-fold CV for selection of lambda
  return(prefit)
}

qtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  ITR = TITR(X)
  Amat = matrix(rep(1:K,n), K * n ,1)
  Xmat = X[rep(1:n, each = K), ]
  A_modified <- matrix(0,(K*n),K)
  A_modified [cbind(1:(K*n),Amat)] <- 1
  A_modified = A_modified[, 1:(K-1)]
  XA_interaction <- matrix (0,K*n,(K-1)*p)
  for(i in 1:(K*n)) {XA_interaction [i,] <-  kronecker(A_modified [i, ], Xmat[i, ])}
  X_modified <- cbind(matrix(1,K*n,1), A_modified, Xmat, XA_interaction)
  Y_predict = predict(result, X_modified, s = "lambda.min", type = "response")
  Y_predict = t (matrix ( Y_predict, K ,  ))
  ITR_predict = apply(Y_predict, 1, which.max)
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

#' Simplex 
#' 
#' \code{simplex} creates simplex vertices for angle-based methods (from Qi et al. (2020)),
#' 
#' @param d Number of simplex dimensions (treatments - 1)
#' 
#' @return Matrix of simplex vertices where each row corresponds to a single vertex and the
#' columns are its coordinates
simplex <- function(d){
  A <- -(1 + sqrt(d+1)) / (d^1.5)
  B <- sqrt((d+1) / d)
  vertices <- matrix(0, d, d+1)
  vertices[, 1] <- d ^(-0.5) * matrix(1, d)
  vertices[, 2:(d+1)] <- A * matrix(1, d, d) + B * diag(1, d)
  return(t(vertices))  ##### There is a transpose here
}

#' AD-Learning
#' 
#' \code{dlearn} implements AD-Learning for the multiple treatment scenario with
#'    L1-regularization for parameter estimation
#'    
#' @param X A numeric matrix of covariates used for fitting 
#' @param A A numeric vector of treatments (coded {1, ..., K})
#' @param Y A numeric vector of observed outcomes
#' @param prob_A A numeric vector of known or estimated propensity scores, i-th element is Pr(a_i|x_i)
#' @param K Number of treatments
#' 
#' @return An object of class \code{cv.glmnet} containing coefficients of the AD-Learning fit
#' 
adlearn <- function(X, A, Y, prob_A, K) {
  n <- nrow(X)
  p <- ncol(X)
  vertices <- simplex(K-1)
  treatment_specific_vertices <- vertices[A, ]
  Y_modified <- (K / (K-1)) * Y * treatment_specific_vertices
  wts <- 1 / prob_A
  prefit <- cv.glmnet(X, Y_modified, alpha = 1, standardize.response = T, weights = wts, family = "mgaussian")
  return(prefit)
}

adtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  ITR = TITR(X)
  prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  vertices <- simplex(K-1)
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = apply( f_predict[,,1] %*% t(vertices), 1, which.max)
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

#' SABD-Learning

#' \code{sdlearn} Implements SABD-Learning in multiple treatment scenarios, using random forest 
#' and XGBoost for the residual modeling step. Additional residual modeling methods 
#' may be added to this function.
#' 
#' @param X A numeric matrix of covariates used for fitting 
#' @param A A numeric vector of treatments (coded {1, ..., K})
#' @param Y A numeric vector of observed outcomes
#' @param prob_A A numeric vector of known or estimated propensity scores
#' @param method Vector of all residual modeling methods to be used
#' @param xgb_params Parameters to be used for XGBoost in residual modeling
#' @param rf_params Parameters to be used for random forest in residual modeling
#' @param K Number of treatments
#' @return A list (possibly of length 1) of objects of class \code{cv.glmnet} containing coefficients 
#'         of SABD-Learning fits. This list is named by the residual modeling method.
#'         
sabdlearn <- function(X, A, Y, prob_A, K) {
  
  # X = data$X_train
  # Y = data$Y_train 
  # A = data$A_train
  # prob_A = prob_A_train
  
  # Initial AD-Learning Step
  vertices <- simplex(K-1)
  treatment_specific_vertices <- vertices[A, ]
  fit <- adlearn(X, A, Y, prob_A, K)
  Y_modified <- (K / (K-1)) * Y * treatment_specific_vertices
  
  b <- coef(fit, "lambda.min")
  #B <- as.matrix(cbind(b$y1, b$y2, b$y3))
  B <- as.matrix(list.cbind(b))
  X_int <- cbind(1, X)
  XB <- X_int %*% B
  pred_all <- rowSums(treatment_specific_vertices * XB)
  resid_squared <- ((K/(K-1))*Y - pred_all)^2
  XA <- cbind(X, as.factor(A))
  
  # Implementation - random forest for residual modeling and resulting SABD-Learning fit
  resid_fit <- randomForest(resid_squared ~ ., data = XA)
  resid_preds <- predict(resid_fit)
  
  # Modified weights and SABD-Learning fitting step
  wts_new <- 1 / resid_preds
  fits <- cv.glmnet(X, Y_modified, alpha = 1,standardize.response = T, weights = wts_new, family = "mgaussian")
  return(fits)
}

sadtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  ITR = TITR(X)
  prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  
  vertices <- simplex(K-1)
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  
  ITR_predict = apply( f_predict[,,1] %*% t(vertices), 1, which.max)
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

radlearn <- function(X, A, Y, main_eff, prob_A, K) {
  n <- nrow(X)
  p <- ncol(X)
  vertices <- simplex(K-1)
  treatment_specific_vertices <- vertices[A, ]
  Y_modified <- (Y-main_eff) * treatment_specific_vertices
  wts <- 1 / prob_A
  prefit <- cv.glmnet(X, Y_modified, alpha = 1, standardize.response = T, weights = wts, family = "mgaussian")
  return(prefit)
}

radtest <- function(X, A, Y, fit, result) {
  n <- nrow(X)
  p <- ncol(X)
  ITR = TITR(X)
  prop_test = predict(fit, X, s = "lambda.min", type = "response")[cbind(1:n,A,1)]
  vertices <- simplex(K-1)
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = apply( f_predict[,,1] %*% t(vertices), 1, which.max)
  rate = 1 - mean(ITR_predict==ITR)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prop_test)[Index] ) / mean( (1/prop_test)[Index] )
  bias = abs(T_value - Evalue)/T_value
  return(list(rate = rate,bias = bias))
}

# Generate DATA #######
generate_data <- function (rep_index){
  set.seed(rep_index)

  X_1 = matrix ( rtruncnorm(sample_size * 2, a = -3, b = 3, mean = 0, sd = 1), sample_size, 2)
  X_2 = matrix ( runif(sample_size * 2, min = -1, max = 1), sample_size, 2)
  X=cbind(X_1,X_2)
  mu = apply(exp(X),1,mean)
  PA = cbind( abs(X[,1]), 2*abs(X[,2]), 3*abs(X[,3]), 4*abs(X[,4]) )
  PA = sweep(PA, MARGIN = 1, STATS = apply(PA,1,sum), FUN = "/")
  A = epsilon = Y = rep(0, sample_size)
  for (i in 1:sample_size){A[i] = sample((1:4), size = 1, prob = PA[i,])}
  epsilon_sd = sqrt(apply(X^2,1,sum))/4
  for (i in 1:sample_size){epsilon[i] = rtruncnorm(1, a = -3*epsilon_sd[i], b = 3*epsilon_sd[i], mean = 0, sd = epsilon_sd[i])}

  Y[which(A==1)] = (mu + 2 * apply( X, 1, sum) + epsilon)[which(A==1)]
  Y[which(A==2)] = (mu + apply( sqrt(abs(X)), 1, sum)/2 + epsilon)[which(A==2)]
  Y[which(A==3)] = (mu + apply(X^2, 1, sum)/2 + epsilon)[which(A==3)]
  Y[which(A==4)] = (mu + apply(cos(X),1,sum)/2 + epsilon)[which(A==4)]
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
  mu = apply(exp(X),1,mean)
  EY1 = mu + 2 * apply( X, 1, sum)
  EY2 = mu + apply( sqrt(abs(X)), 1, sum)/2
  EY3 = mu + apply(X^2, 1, sum)/2
  EY4 = mu + apply(cos(X),1,sum)/2
  EY = cbind(EY1,EY2,EY3,EY4)
  ITR = apply(EY, 1, which.max)
  return(ITR)
}

Tvalue <- function(Mon_size){
  set.seed(0)
  X_1 = matrix ( rtruncnorm(Mon_size * 2, a = -3, b = 3, mean = 0, sd = 1), Mon_size, 2)
  X_2 = matrix ( runif(Mon_size * 2, min = -1, max = 1), Mon_size, 2)
  X=cbind(X_1,X_2)
  mu = apply(exp(X),1,mean)
  gamma_1 = mu + 2 * apply( X, 1, sum)
  gamma_2 = mu + apply( sqrt(abs(X)), 1, sum)/2
  gamma_3 = mu + apply(X^2, 1, sum)/2
  gamma_4 = mu + apply(cos(X),1,sum)/2
  gamma = cbind(gamma_1,gamma_2,gamma_3,gamma_4) 
  value = mean(apply(gamma,1,max)) 
  return(value)
}

num_rep = 100
train_size = 500
test_size = 500
sample_size = train_size + test_size
K = 4
T_value = Tvalue(5000000)
Results = matrix(0, num_rep, 8)

for(rep_index in 1:num_rep){
  data = generate_data(rep_index)
  set.seed(rep_index)
  Multilogistic.fit = cv.glmnet(data$X_train, data$A_train, family = "multinomial", type.multinomial = "grouped")
  prob_A_train = predict(Multilogistic.fit, data$X_train, s = "lambda.min", type = "response")[cbind(1:train_size,data$A_train,1)]
  ql_result = qlearn(data$X_train, data$A_train, data$Y_train, K)
  qtest_result = qtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, ql_result)
  ad_result = adlearn(data$X_train, data$A_train, data$Y_train, prob_A_train, K)
  adtest_result = adtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, ad_result)
  Maineffect.fit = cv.glmnet(data$X_train, data$Y_train, weights = 1/prob_A_train, standardize.response = T)
  main_eff = predict(Maineffect.fit, data$X_train, s = "lambda.min", type = "response") [cbind(1:train_size,1)]
  rad_result = radlearn(data$X_train, data$A_train, data$Y_train, main_eff, prob_A_train, K)
  radtest_result = radtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, rad_result)
  sad_result = sabdlearn(data$X_train, data$A_train, data$Y_train, prob_A_train, K)
  sadtest_result = sadtest(data$X_test, data$A_test, data$Y_test, Multilogistic.fit, sad_result)
  Results[rep_index,] = c(qtest_result$rate,qtest_result$bias,adtest_result$rate,adtest_result$bias,
                          radtest_result$rate,radtest_result$bias,sadtest_result$rate,sadtest_result$bias)
  print(rep_index)
}
write.csv(Results, paste0("Results_",train_size,".csv"), row.names = FALSE)

