# multi_sd.R

library(rlist)
# library(caret) useless
library(Matrix)
library(glmnet)
library(randomForest)
library(xgboost)
# library(tidyverse) useless
library(MASS)
library(truncnorm)
library(BART)

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

# X = data$X_test
# A = data$A_test
# Y = data$Y_test
# Y1 = data_Y1[(train_size+1):size]
# Y2 = data_Y2[(train_size+1):size]
# fit = Multilogistic.fit
# result = ql_result

qtest <- function(X, A, Y, Y1, Y2, prob_test, result) {
  n <- nrow(X)
  p <- ncol(X)
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
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prob_test)[Index] ) / mean( (1/prob_test)[Index] )
  results <- list()
  results$ITR = ITR_predict
  results$Evalue = Evalue
  return(results)
}

adtest <- function(X, A, Y, Y1, Y2, prob_test, result) {
  n <- nrow(X)
  p <- ncol(X)
  vertices <- simplex(K-1)
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = apply( f_predict[,,1] %*% t(vertices), 1, which.max)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prob_test)[Index] ) / mean( (1/prob_test)[Index] )
  results <- list()
  results$ITR = ITR_predict
  results$Evalue = Evalue
  return(results)
}

sadtest <- function(X, A, Y, Y1, Y2, prob_test, result) {
  n <- nrow(X)
  p <- ncol(X)
  vertices <- simplex(K-1)
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = apply( f_predict[,,1] %*% t(vertices), 1, which.max)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prob_test)[Index] ) / mean( (1/prob_test)[Index] )
  results <- list()
  results$ITR = ITR_predict
  results$Evalue = Evalue
  return(results)
}

radtest <- function(X, A, Y, Y1, Y2, prob_test, result) {
  n <- nrow(X)
  p <- ncol(X)
  vertices <- simplex(K-1)
  f_predict = predict(result, X, s = "lambda.min", type = "response")
  ITR_predict = apply( f_predict[,,1] %*% t(vertices), 1, which.max)
  Index = which(A == ITR_predict)
  Evalue = mean( (Y/prob_test)[Index] ) / mean( (1/prob_test)[Index] )
  results <- list()
  results$ITR = ITR_predict
  results$Evalue = Evalue
  return(results)
}

data(ACTG175)
set.seed(0)
size = 2139
train_size = ceiling(0.8*size)
test_size = size - train_size
K = 4
XC1 = (ACTG175$age - min(ACTG175$age)) / (max(ACTG175$age) - min(ACTG175$age)) 
XC2 = (ACTG175$wtkg - min(ACTG175$wtkg)) / (max(ACTG175$wtkg) - min(ACTG175$wtkg)) 
XC3 = (ACTG175$cd40 - min(ACTG175$cd40)) / (max(ACTG175$cd40) - min(ACTG175$cd40)) 
XC4 = (ACTG175$cd80 - min(ACTG175$cd80)) / (max(ACTG175$cd80) - min(ACTG175$cd80)) 
XC5 = (ACTG175$karnof - min(ACTG175$karnof)) / (max(ACTG175$karnof) - min(ACTG175$karnof)) 
data_XC = cbind(XC1, XC2, XC3, XC4, XC5)
data_XD = cbind(ACTG175$gender,ACTG175$race,ACTG175$homo,ACTG175$drugs,ACTG175$symptom,ACTG175$str2,ACTG175$hemo)
data_X = cbind(data_XC,data_XD)
data_A = ACTG175$arms + 1
data_Y = ((ACTG175$cd420-ACTG175$cd40)-min(ACTG175$cd420-ACTG175$cd40)) / (max(ACTG175$cd420-ACTG175$cd40)-min(ACTG175$cd420-ACTG175$cd40))
data_Y0 = ACTG175$cd40
data_Y1 = ACTG175$cd420
##data = list(Y_train = data_Y[1:train_size], A_train = data_A[1:train_size], X_train = data_X[1:train_size,], 
            ##Y_test = data_Y[(train_size+1):size], A_test = data_A[(train_size+1):size], X_test  = data_X[(train_size+1):size,])
data = list(Y_test = data_Y[1:test_size], A_test = data_A[1:test_size], X_test = data_X[1:test_size,], 
            Y_train = data_Y[(test_size+1):size], A_train = data_A[(test_size+1):size],  X_train = data_X[(test_size+1):size,])
prob_A_train = rep(0.25,train_size)
prob_A_test = rep(0.25,test_size)
Maineffect.fit = cv.glmnet(data$X_train, data$Y_train, weights = 1/prob_A_train, standardize.response = T)
main_eff = predict(Maineffect.fit, data$X_train, s = "lambda.min", type = "response") [cbind(1:train_size,1)]
ql_result = qlearn(data$X_train, data$A_train, data$Y_train, K)
ad_result = adlearn(data$X_train, data$A_train, data$Y_train, prob_A_train, K)
rad_result = radlearn(data$X_train, data$A_train, data$Y_train, main_eff, prob_A_train, K)
sad_result = sabdlearn(data$X_train, data$A_train, data$Y_train, prob_A_train, K)
qtest_result = qtest(data$X_test, data$A_test, data$Y_test, data_Y1[(train_size+1):size], data_Y2[(train_size+1):size], prob_A_test, ql_result)
adtest_result = adtest(data$X_test, data$A_test, data$Y_test, data_Y1[(train_size+1):size], data_Y2[(train_size+1):size], prob_A_test, ad_result)
radtest_result = radtest(data$X_test, data$A_test, data$Y_test, data_Y1[(train_size+1):size], data_Y2[(train_size+1):size], prob_A_test, rad_result)
sadtest_result = sadtest(data$X_test, data$A_test, data$Y_test, data_Y1[(train_size+1):size], data_Y2[(train_size+1):size], prob_A_test, sad_result)
Evalue_Results = rbind(qtest_result$Evalue,adtest_result$Evalue,radtest_result$Evalue,sadtest_result$Evalue)
ITR_Results =  rbind(qtest_result$ITR,adtest_result$ITR,radtest_result$ITR,sadtest_result$ITR)
write.csv(cbind(data_Y,data_A,data_X,data_Y0,data_Y1), paste0("Realdata.csv"), row.names = FALSE)
write.csv(Evalue_Results, paste0("Results_realdata.csv"), row.names = FALSE)
write.csv(ITR_Results, paste0("ITR_Results_realdata.csv"), row.names = FALSE)

