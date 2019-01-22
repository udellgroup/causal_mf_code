# library(testthat)
library(assertthat)
library(softImpute)
library(caret)
library(mvtnorm)
library(RSpectra)
library(glmnet)
library(logisticPCA)
library(rpca)
library(rARPACK)
library(purrr)
# library(gridExtra)
# library(grid)

make_V <- function(r, p){
  matrix(rnorm(r*p), nrow = p, ncol = r)
}

compute_logit <- function(x){
  1/(1 + exp(-x))
}

design_matrix <- function(V, n, r, p){
  # this function constructs the design list with U, V, X components 
  #   U: n * r matrix with entries from standard gaussian distribution
  #   V: n * p matrix given in the input 
  #   X: X = UV^T

  design = vector("list")
  design$U = matrix(rnorm(n*r), nrow = n, ncol = r)
  design$V = V
  design$X = design$U %*% t(design$V)
  
  assert_that(are_equal(dim(design$U), c(n, r)))
  assert_that(are_equal(dim(design$V), c(p, r)))
  assert_that(are_equal(dim(design$X), c(n, p)))

  design
}

design_matrix_binary <- function(V, n, r, p){
  # this function constructs the design list with U, V, X components 
  #   U: n * r matrix with entries from standard gaussian distribution
  #   V: n * p matrix given in the input 
  #   X: X = UV^T
  #   this function is the same as design_matrix()

  design = vector("list")
  design$U = matrix(rnorm(n*r), nrow = n, ncol = r)
  design$V = V
  design$X = design$U %*% t(design$V)
  
  
  assert_that(are_equal(dim(design$U), c(n, r)))
  assert_that(are_equal(dim(design$V), c(p, r)))
  assert_that(are_equal(dim(design$X), c(n, p)))

  design
}

perturbation_gaussian <- function(design, noise_sd = 5){
  # add Gaussian noise to the UV^T matrix, which creates Gaussian noisy proxies 
  n = nrow(design$X)
  p = ncol(design$X)
  design$X + matrix(rnorm(n*p, 0, noise_sd), nrow = n, ncol = p)
}

perturbation_binary <- function(design, prop = 0.2){
  # add Bernoulli noise to the UV^T matrix, which creates Binary noisy proxies
  #   the prop argument is a dummy argument that doesn't have any effect; it's included here 
  #   so that perturbation_binary() and perturbation_gaussian() have the same number of inputs
  n = nrow(design$X)
  p = ncol(design$X)
  prob = compute_logit(design$X)
  Xm = matrix(ifelse(runif(n*p) <= c(prob), 1, 0), n, p)
  Xm
}

treatment_assignment <- function(design, treat_coef){
  # assign the treatment according to
  #   P(T = 1 | U) = 1/(1 + exp(-U*treat_coef))

  n = nrow(design$X)
  assert_that(are_equal(ncol(design$U), length(treat_coef)))
  prob = 1/(1 + exp(-design$U %*% treat_coef))
  Treatment <- ifelse(runif(n, 0, 1) <= c(prob), 1, 0)
  as.matrix(Treatment)
}


recover_pca_binary_cv <- function(design, r_seq){
  # binary matrix factorization on the noisy proxy matrix design$Xm 
  #     with rank chosen by cross validation from r_seq 

  cv.fit = cv.lsvd(design$Xm, ks = r_seq)
  best_r = which.min(cv.fit)
  logPCA = logisticSVD(design$Xm, k = best_r, main_effects = F)
  Uhat = logPCA$A
  list(Uhat = Uhat, best_r = best_r)
}

compute_folds <- function(Xm, nfolds = 5){
  # create cross-validation folds for gaussian matrix factorization 

  n = nrow(Xm); p = ncol(Xm)
  nfold_train = createFolds(1:n, nfolds, returnTrain = T)
  pfold_train = createFolds(1:p, nfolds, returnTrain = T)
  nfold_test = lapply(nfold_train, function(x) setdiff(1:n, x))
  pfold_test = lapply(pfold_train, function(x) setdiff(1:p, x))
  list(nfold_train = nfold_train, pfold_train = pfold_train, 
       nfold_test = nfold_test, pfold_test = pfold_test)
}

cross_valid <- function(design, r, warm, folds, nfolds = 5){
  # compute the cross validation error for gaussian matrix factorization with different ranks 
  
  assert_that(length(folds$nfold_train) == nfolds)
  
  nfold_train = folds$nfold_train
  pfold_train = folds$pfold_train
  nfold_test = folds$nfold_test
  pfold_test = folds$pfold_test
  
  error_folds = numeric(nfolds)
  fit_folds = list()
  
  for (f in 1:nfolds){
    temp_data = design$Xm
    temp_data[nfold_test[[f]], pfold_test[[f]]] = NA
    fit = softImpute(temp_data, rank.max = r, type = "als", maxit = 1000, warm.start = warm)
    pred = impute(fit, i = rep(nfold_test[[f]], length(pfold_test[[f]])), 
                  j = rep(pfold_test[[f]], each = length(nfold_test[[f]])))
    assert_that(length(c(design$Xm[nfold_test[[f]], pfold_test[[f]]])) == length(pred))
    error = mean((c(design$Xm[nfold_test[[f]], pfold_test[[f]]]) - pred)^2, na.rm = T)
    error_folds[f] = error
    fit_folds[[f]] = fit
  }
  list(error = mean(error_folds), fit = fit_folds[[which.min(error_folds)]])
}

recover_pca_gaussian_cv <- function(design, r_seq, nfolds = 5){
  # Gaussian matrix factorization on the noisy proxy matrix design$Xm 
  #     with rank chosen by cross validation from r_seq 
  #     the matrix factorization is carried out by the softImpute package  

  cv_error = numeric(length(r_seq))
  warm_list = list()
  folds = compute_folds(design$Xm)
  
  for (r in 1:length(r_seq)){
    if (r == 1){
      temp = cross_valid(design, r_seq[r], warm = NULL, folds = folds, nfolds = nfolds)
      cv_error[r] = temp$error
      warm_list[[r]] = temp$fit
    } else{
      temp = cross_valid(design, r_seq[r], warm = warm_list[[r-1]], folds = folds, nfolds = nfolds)
      cv_error[r] = temp$error
      warm_list[[r]] = temp$fit
    }
  }
  
  best_r = r_seq[which.min(cv_error)]
  warm = warm_list[[which.min(cv_error)]]
  result = softImpute(design$Xm, rank.max = best_r, type = "als", maxit = 1000, warm.start = warm)
  list(Uhat = result$u, best_r = best_r)
}


simulate_response <- function(design, true_te, out_coef){
  # simulate the response variable Y = true_te*T + out_coef*U + error where errors come from i.i.d N(0, 1)

  n = nrow(design$X)
  assert_that(are_equal(length(out_coef), ncol(design$U)))
  Y = true_te*design$Treatment + design$U %*% out_coef + as.matrix(rnorm(n))
  Y
}

regress_unbiased <- function(design){
  # causal effect estimates from linear regression Y ~ U + T

  model <- lm(design$Y ~ design$U + design$Treatment + 0)
  coefficients(model)["design$Treatment"]
}

regress_corrupted_ols <- function(design){
  # causal effect estimates from linear regression Y ~ Xm + T

  model <- lm(design$Y ~ design$Xm + design$Treatment + 0)
  coefficients(model)["design$Treatment"]
}

regress_corrupted_lasso <- function(design){
  # causal effect estimates from lasso regression Y ~ U + T

  p.factor = rep(1, ncol(design$Xm) + 1)
  p.factor[length(p.factor)] = 0
  # print(p.factor)
  cvfit = cv.glmnet(x = cbind(design$Xm, design$Treatment), y = design$Y, 
                    intercept = F, penalty.factor = p.factor, alpha = 1,
                    )
  # print(coef(cvfit, s = "lambda.min"))
  coef(cvfit, s = "lambda.min")[ncol(design$Xm) + 2]
}

regress_corrupted_ridge <- function(design){
  # causal effect estimates from ridge regression Y ~ U + T

  p.factor = rep(1, ncol(design$Xm) + 1)
  p.factor[length(p.factor)] = 0
  cvfit = cv.glmnet(x = cbind(design$Xm, design$Treatment), y = design$Y,
                   intercept = F, penalty.factor = p.factor, alpha = 0,
                    )
  coefs = coef(cvfit, s = "lambda.min")
  coefs[length(coefs)]
}


regress_svd <- function(design){
  # causal effect estimates from linear regression Y ~ Uhat + T where Uhat
  #   is from matrix factorization 

  model <- lm(design$Y ~ design$Uhat + design$Treatment + 0)
  coefficients(model)["design$Treatment"]
}


impute_by_mode <- function(data){
  # impute each missing value in the data frame by the column mode 
  data2 = data
  for (j in 1:ncol(data)){
    freq = sort(table((data[, j])), decreasing = T)
    data2[, j][is.na(data[, j])] = as.integer(names(freq[1]))
  }
  data2
}


main_cv <- function(nseq, pseq, treat_coef, out_coef, true_te = 2, true_r = 3, r_seq = c(1,3,5,7,9), noise_level = 5, rep_number = 1, 
                    funs = c(regress_unbiased, regress_svd, regress_corrupted_lasso, regress_corrupted_ridge, regress_corrupted_ols), 
                 perturbation. = perturbation_binary,  recover_pca. = recover_pca_binary_cv){
  ### Input
  # nseq, pseq: the list of sample size and dimension of proxies
  # treat_coef: the coef of generating the treatment used in treatment_assignment()
  # out_coef: the coef of generating the response variable used in simulate_response()
  # true_te: the true treatment effect, used in simulate_response()
  # r_seq: the candidate ranks used in cross validation for matrix factorization 
  # noise_level: the standard deviation of gaussian noise used in perturbation_gaussian(); when perturbation. = perturbation_binary, 
  #   this argument doesn't have any effect
  # rep_number: the number of repetitions of the experiments for each dimension setting 
  # funs: the list of regression functions for causal effect estimation that uses a subset of Y, Uhat, U, X
  #   regress_unbiased OLS regression Y ~ U + T with U as the true confounder 
  #   regress_svd OLS regression Y ~ Uhat + T with Uhat from recover_pca.
  #   regress_corrupted_lasso lasso regression Y ~ X + T with X as the noisy proxy generated from perturbation.
  #   regress_corrupted_ridge ridge regression Y ~ X + T with X as the noisy proxy generated from perturbation.
  #   regress_corrupted_ols OLS regression Y ~ X + T with X as the noisy proxy generated from perturbation.
  # perturbation.: the function used to generate the noisy proxy, it can be either perturbation_binary or perturbation_gaussian
  # recover_pca.: the function used to factorize the noisy proxy matrix 

  ### Output 
  # estimate: a list of length len(nseq) where each component corresponds to the estimation result for each n, p dimension setting;
  #   each component is a len(funs) by rep_number matrix where the (i, j)th entry corresponds to the causal effect estimate for the
  #   ith regression function in funs at the jth repetition of the experiments
  # best_r: a len(nseq) by rep_number matrix whose (i, j) entry records the best rank chosen by cross validation in the ith dimension
  #   setting in the jth repetition of the experiments 
  
  len_p = length(pseq)
  
  estimate_result = list()
  best_r = matrix(0, length(nseq), rep_number)
  
  for (i in seq_along(nseq)){
    cat("the", i, ' th p:', pseq[i],"\n")
    cat("the", i, ' th n:', nseq[i],"\n")
   
    n = nseq[i]
    p = pseq[i]
    
    estimate_result[[i]] <- matrix(NA, length(funs), rep_number)
    rownames(estimate_result[[i]]) <- c("regress_unbiased", 
                                   "regress_svd", "regress_corrupted_lasso", 
                                   "regress_corrupted_ridge", "regress_corrupted_ols")

    ### the following part is needed only when n < p so that OLS doesn't work 
    # funs_lowdim = funs
    # funs_highdim = c(regress_unbiased, 
    #            regress_svd, regress_corrupted_lasso, regress_corrupted_ridge)

    # if (n < p) {
    #   funs2 = funs_highdim
  
    # } else {
    #   funs2 = funs_lowdim
    # }
    
    
    V = make_V(true_r, p)
    
    for (j in 1:rep_number){
      cat("the", j, " th repetition", "\n")
      
      design = design_matrix(V, n, true_r, p)
      design$Xm = perturbation.(design, noise_level)
      design$Treatment <- treatment_assignment(design, treat_coef)
      design$Y <- simulate_response(design, true_te, out_coef)
      
      pca_factor = recover_pca.(design, r_seq)
      design$Uhat <- pca_factor$Uhat
      best_r[i, j] <-  pca_factor$best_r
      
      ### the following part is needed only when n < p so that OLS doesn't work 
      # tmp <- unlist(lapply(funs2, function(f) f(design)))
      # if (length(tmp) < length(funs_lowdim)){
      #   tmp = c(tmp, NA)
      # }
      # estimate_result[[i]][, j] <- tmp
      estimate_result[[i]][, j] <- unlist(lapply(funs, function(f) f(design)))
    }
    
  }
  list(estimate = estimate_result, best_r = best_r)
}



main_cv_missing <- function(nseq, pseq, treat_coef, out_coef, true_te = 2, true_r = 3, r_seq = c(1,3,5,7,9), miss_frac = 0.3, noise_level = 5, rep_number = 1, 
                    funs = c(regress_unbiased, regress_svd, regress_corrupted_lasso, regress_corrupted_ridge, regress_corrupted_ols), 
                    perturbation. = perturbation_binary,  recover_pca. = recover_pca_binary_cv){
  ### Input  
  # nseq, pseq: the list of sample size and dimension of proxies
  # treat_coef: the coef of generating the treatment used in treatment_assignment()
  # out_coef: the coef of generating the response variable used in simulate_response()
  # true_te: the true treatment effect, used in simulate_response()
  # r_seq: the candidate ranks used in cross validation for matrix factorization 
  # miss_frac: the fraction of missing values in the proxy matrix 
  # noise_level: the standard deviation of gaussian noise used in perturbation_gaussian(); when perturbation. = perturbation_binary, 
  #   this argument doesn't have any effect
  # rep_number: the number of repetitions of the whole experiments
  # funs: the list of regression functions for causal effect estimation that uses a subset of Y, Uhat, U, X
  #   regress_unbiased OLS regression Y ~ U + T with U as the true confounder 
  #   regress_svd OLS regression Y ~ Uhat + T with Uhat from recover_pca.
  #   regress_corrupted_lasso lasso regression Y ~ X + T with X as the noisy proxy generated from perturbation.
  #   regress_corrupted_ridge ridge regression Y ~ X + T with X as the noisy proxy generated from perturbation.
  #   regress_corrupted_ols OLS regression Y ~ X + T with X as the noisy proxy generated from perturbation.
  # perturbation.: the function used to generate the noisy proxy, it can be either perturbation_binary or perturbation_gaussian
  # recover_pca.: the function used to factorize the noisy proxy matrix 

  ### Output 
  # estimate: a list of length len(nseq) where each component corresponds to the estimation result for each n, p dimension setting;
  #   each component is a len(funs) by rep_number matrix where the (i, j)th entry corresponds to the causal effect estimate for the
  #   ith regression function in funs at the jth repetition of the experiments
  # best_r: a len(nseq) by rep_number matrix whose (i, j) entry records the best rank chosen by cross validation in the ith dimension
  #   setting in the jth repetition of the experiments 
  
  len_p = length(pseq)
  
  estimate_result = list()
  best_r = matrix(0, length(nseq), rep_number)
  
  for (i in seq_along(nseq)){
    cat("the", i, ' th p:', pseq[i],"\n")
    cat("the", i, ' th n:', nseq[i],"\n")
    
    n = nseq[i]
    p = pseq[i]
    
    estimate_result[[i]] <- matrix(NA, length(funs), rep_number)
    rownames(estimate_result[[i]]) <- c("regress_unbiased", 
                                        "regress_svd", "regress_corrupted_lasso", 
                                        "regress_corrupted_ridge", "regress_corrupted_ols")
    
    
    ### the following part is needed only when n < p so that OLS doesn't work 
    # funs_lowdim = funs
    # funs_highdim = c(regress_unbiased, 
    #                  regress_umvue, regress_svd, regress_corrupted_lasso, regress_corrupted_ridge)
    # if (n < p) {
    #   funs2 = funs_highdim
      
    # } else {
    #   funs2 = funs_lowdim
    # }
    
    
    V = make_V(true_r, p)
    
    for (j in 1:rep_number){
      cat("the", j, " th repetition", "\n")
      
      design = design_matrix(V, n, true_r, p)
      design$Xm = perturbation.(design, noise_level)
      design$Xm = gen_missing(design$Xm, miss_frac)
      design$Treatment <- treatment_assignment(design, treat_coef)
      design$Y <- simulate_response(design, true_te, out_coef)
      
      pca_factor = recover_pca.(design, r_seq)
      design$Uhat <- pca_factor$Uhat
      best_r[i, j] <-  pca_factor$best_r

      design$Xm = impute_by_mode(design$Xm)
      
      ### the following part is needed only when n < p so that OLS doesn't work 
      # tmp <- unlist(lapply(funs2, function(f) f(design)))
      # if (length(tmp) < length(funs_lowdim)){
      #   tmp = c(tmp, NA)
      # }
      # estimate_result[[i]][, j] <- tmp
      estimate_result[[i]][, j] <- unlist(lapply(funs, function(f) f(design)))
    }
    
  }
  list(estimate = estimate_result, best_r = best_r)
}




gen_missing <- function(Xm, miss_frac = 0.3){
  if (miss_frac == 0){
    return(Xm)
  }
  n = nrow(Xm); p = ncol(Xm)
  missing = matrix(runif(n*p) < miss_frac, n, p)
  Xm[missing] = NA
  Xm
}









