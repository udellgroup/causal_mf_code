sigmoid  = function(x){
  1/(1 + exp(-x))
}



fullmatch_estimator <- function(match, data){
  # estimate the ate based on the matching pair output 
  groups = levels(match)
  ate_group = numeric(length(groups))
  for (i in 1:length(groups)){
    group = groups[i]
    elements = as.numeric(names(match[match == group]))
    group_data = data[elements, ]
    ate_group[i] = mean(group_data$y[group_data$t == 1]) - mean(group_data$y[group_data$t == 0])
  }
  mean(ate_group)
}
proxy_cols <- function(n_views, confounder = "gestat10"){
  # this function generates the column names for the noisy confounders:
  #   gestat10, gestat10_2, ..., gestat10_n
  if (n_views > 1){
    total_corrup = c(confounder, sapply(confounder, function(x) sapply(1:(n_views-1), function(n) paste0(x, "_", n+1))))
  } else {
    total_corrup = confounder
  }
  total_corrup
}


propensity_match <- function(n_views_seq, rep_seq, fraction, Date, miss = 0){
  ate_mf = matrix(0, length(n_views_seq), length(rep_seq))
  ate_corrup = matrix(0, length(n_views_seq), length(rep_seq))
  
  for (i in 1:length(n_views_seq)){
    print(paste("the", i, "th views:", n_views_seq[i]))
    for (j in 1:length(rep_seq)){
      print(paste("the", j, "th repetition"))
      
      U = read.csv(paste0("Output/", Date, "noise_", n_views_seq[i],
                          "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))
      if (miss == 0){
        comp_data = read.csv(paste0("Data/",  "noise_", n_views_seq[i],
                                    "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))[, -1]
      }
      
      if (miss == 0){
        total_corrup = proxy_cols(n_views_seq[i])
        X = comp_data[total_corrup];  ### possibly impute here
        y = comp_data[, 1]; t = comp_data[, 2];
        # gestat10 = orig_data["gestat10"]
      }
      
      data_mf = data.frame(y = y, t = t, U = U)
      data_corrup = cbind(y, t, X)
      
      data = data_mf
      model = glm(t ~ ., data = data, family = binomial)
      match  = fullmatch(model, data = data)
      mf = fullmatch_estimator(match, data)
      
      data = data_corrup
      model = glm(t ~ ., data = data, family = binomial)
      match  = fullmatch(model, data = data)
      corrup = fullmatch_estimator(match, data)
      
      ate_mf[i, j] = mf
      ate_corrup[i, j] = corrup
    }
  }
  list(mf = ate_mf, corrup = ate_corrup)
}

full_match <- function(n_views_seq, rep_seq, fraction, Date, miss = 0){
  ate_mf = matrix(0, length(n_views_seq), length(rep_seq))
  ate_corrup = matrix(0, length(n_views_seq), length(rep_seq))
  for (i in 1:length(n_views_seq)){
    print(paste("the", i, "th views:", n_views_seq[i]))
    result = foreach(j = 1:length(rep_seq), .combine = cbind, 
                     .packages = "optmatch",
                     .export =  c("proxy_cols", "fullmatch_estimator")) %dopar% {
      print(paste("the", j, "th repetition"))
      
      U = read.csv(paste0("Output/", Date, "noise_", n_views_seq[i],
                          "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))
      if (miss == 0){
        comp_data = read.csv(paste0("Data/",  "noise_", n_views_seq[i],
                                    "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))[, -1]
      }
      
      if (miss == 0){
        total_corrup = proxy_cols(n_views_seq[i], confounder = "gestat10")
        X = comp_data[total_corrup];  ### possibly impute here
        y = comp_data[, 1]; t = comp_data[, 2];
        # gestat10 = orig_data["gestat10"]
      }
      
      data_mf = data.frame(y = y, t = t, U = U)
      data_corrup = cbind(y, t, X)
      
      data = data_mf
      match = fullmatch(as.formula(paste("t ~", Reduce(function(x, y) paste(x, "+", y), colnames(data_mf)[-c(1:2)]))), data = data, method = "mahalanobis")
      mf = fullmatch_estimator(match, data)
      data = data_corrup
      match = fullmatch(as.formula(paste("t ~", Reduce(function(x, y) paste(x, "+", y), total_corrup))), data = data, method = "mahalanobis")
      corrup = fullmatch_estimator(match, data)
      result = matrix(c(mf, corrup), 2, 1)
      rownames(result) = c("mf", "corrup")
      result
    }
    ate_mf[i, ] = result[1, ]
    ate_corrup[i, ] = result[2, ]
  }
  list(mf = ate_mf, corrup = ate_corrup)
}

compute_pw_by_views <- function(n_views_seq, rep_seq, miss, fraction, Date){
  # this function generates ate estimates list for two methods with complete noise data
  #   propensity score reweighting with propensity estimated from logistic regression on the U from matrix factorization
  #   propensity score reweighting with propensity estimated from logistic regression on the corrupted data
  
  ate_mf = matrix(0, length(n_views_seq), length(rep_seq))
  ate_corrup = matrix(0, length(n_views_seq), length(rep_seq))
  
  for (i in 1:length(n_views_seq)){
    print(paste("the", i, "th views:", n_views_seq[i]))
    for (j in 1:length(rep_seq)){
      print(paste("the", j, "th repetition"))
      
      U = read.csv(paste0("Output/", Date, "noise_", n_views_seq[i],
                          "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))
      if (miss == 0){
        comp_data = read.csv(paste0("Data/",  "noise_", n_views_seq[i],
                                    "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))[, -1]
      }
      
      if (miss == 0){
        total_corrup = proxy_cols(n_views_seq[i])
        X = comp_data[total_corrup];  ### possibly impute here
        y = comp_data[, 1]; t = comp_data[, 2];
      }
      
      prop_score = propensity_score(U, t, X)
      score = prop_score$mf
      ate_mf[i, j] = mean(y*t/score) - mean(y*(1-t)/(1-score))
      score = prop_score$corrup
      ate_corrup[i, j] = mean(y*t/score) - mean(y*(1-t)/(1-score))
    }
  }
  list(mf = ate_mf, corrup = ate_corrup)
}

compute_dr_by_views <- function(n_views_seq, rep_seq, miss, fraction, Date){
  # this function generates ate estimates list for doubly robust methods with complete noise data
  #   both regresion and propensity score are estimated by logistic regression on U from matrix factorization
  #   both regresion and propensity score are estimated by logistic regression on corrupted data
  
  ate_mf = matrix(0, length(n_views_seq), length(rep_seq))
  ate_corrup = matrix(0, length(n_views_seq), length(rep_seq))
  
  for (i in 1:length(n_views_seq)){
    print(paste("the", i, "th views:", n_views_seq[i]))
    for (j in 1:length(rep_seq)){
      print(paste("the", j, "th repetition"))
      
      
      U = read.csv(paste0("Output/", Date, "noise_", n_views_seq[i],
                          "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))
      if (miss == 0){
        comp_data = read.csv(paste0("Data/",  "noise_", n_views_seq[i],
                                    "views_", fraction*100, "fraction_rep", rep_seq[j], ".csv"))[, -1]
      }
      
      if (miss == 0){
        total_corrup = proxy_cols(n_views_seq[i])
        X = comp_data[total_corrup];  ### possibly impute here
        y = comp_data[, 1]; t = comp_data[, 2];
      }
      
      prop_score = propensity_score(U, t, X)
      
      # mf 
      data = data.frame(y = y, t = t, U = U)
      lr = glm(y ~ ., data = data, family = binomial)
      score = as.matrix(cbind(rep(1, dim(data)[1]), data[, -c(1:2)]))%*%coefficients(lr)[-2]
      m1 = sigmoid(score + coefficients(lr)[2])
      m0 = sigmoid(score)
      ps = prop_score$mf
      ate_dr_mf = mean(t*y/ps - (t - ps)/ps*m1) - mean((1-t)*y/(1-ps) + (t-ps)/(1-ps)*m0)
      
      # corrup
      data = data.frame(y = y, t = t, X = X)
      lr = glm(y ~ ., data = data, family = binomial)
      score = as.matrix(cbind(rep(1, dim(data)[1]), data[, -c(1:2)]))%*%coefficients(lr)[-2]
      m1 = sigmoid(score + coefficients(lr)[2])
      m0 = sigmoid(score)
      ps = prop_score$corrup
      ate_dr_corrup = mean(t*y/ps - (t - ps)/ps*m1) - mean((1-t)*y/(1-ps) + (t-ps)/(1-ps)*m0)
      
      ate_mf[i, j] = ate_dr_mf
      ate_corrup[i , j] = ate_dr_corrup   
      
    }
  }
  list(dr_mf = ate_mf, dr_corrup = ate_corrup)
}

propensity_score <- function(U, t, X){
  data_mf = as.data.frame(cbind(t, U))
  pw_mf = glm(t ~ ., family = binomial, data = data_mf)
  pw_mf = predict(pw_mf, type = "response")
  
  data_corrup = as.data.frame(cbind(t, X))
  pw_corrup = glm(t ~ ., family = binomial, data = data_corrup)
  pw_corrup = predict(pw_corrup, type = "response") 
  
  list(mf = pw_mf, corrup = pw_corrup)
}

mice_impute <- function(x, n_views, m){
  mice.out = mice(x, m = m, print = F)$imp
  
  cols = proxy_cols(n_views)
  assert_that(length(cols) == length(mice.out))
  
  imputed_data = vector("list", m)
  
  for (i in 1:length(imputed_data)){
    imputed_data[[i]] = x
    for (col in cols){
      temp = unlist(mice.out[[col]][i])
      imputed_data[[i]][col][is.na(imputed_data[[i]][col])] = temp
    }
  }
  imputed_data
}
main_mice <- function(n_views_seq, rep_seq, m, miss, fraction, Date = "1_7/"){
  # this function generates mice imputation for missing data with different n_views and rep settings  
  # m is the number of multiple imputations for each dataset 
  # the saved rds file is a length-m list of the imputed datasets
  est = matrix(0, length(n_views_seq), length(rep_seq))
  for (i in 1:length(n_views_seq)){
    print(paste("the", i, "th n_views:", n_views_seq[i]))
    
    est_temp = foreach(j = 1:length(rep_seq), .packages = c("mice", "assertthat"), 
            .export = c("mice_impute", "proxy_cols", "lr", "sigmoid", "est")) %dopar% {
      print(paste("the", j, "th rep"))
      comp_data = read.csv(paste0("Data/missing/",  "noise_", n_views_seq[i],
                                  "views_", fraction*100, "fraction_rep", rep_seq[j], "missing_", miss, ".csv"))[, -1]
      
      corrup_data = cbind(comp_data[, c(1:2)], comp_data[, proxy_cols(n_views_seq[i])])
      
      x = corrup_data[, -c(1:2)]
      result = mice_impute(x, n_views_seq[i], m) 
      temp = rep(0, m)
      for (i in 1:length(result)){
        temp[i] = lr(cbind(comp_data[, 1:2], result[[i]]))
      }
      mean(temp)
    }
    est[i, ] = unlist(est_temp)
  }
  est
}
lr <- function(data){
  model <- glm(y ~ ., data = data, family = binomial)
  score = as.matrix(cbind(rep(1, dim(data)[1]), data[, -c(1:2)]))%*%coefficients(model)[-2]
  mean(sigmoid(score + coefficients(model)[2]) - sigmoid(score))
}

impute_by_mode <- function(data){
  data2 = data
  missing = sapply(names(data), function(x) sum(is.na(data[x])) > 0)
  for (name in names(data)[missing]){
    freq = sort(table((data[name])), decreasing = T)
    data2[name][is.na(data[name])] = as.integer(names(freq[1]))
  }
  data2
}
main_mode <- function(n_views_seq, rep_seq, m, miss, fraction, Date = "1_7/"){
  est = matrix(0, length(n_views_seq), length(rep_seq))
  for (i in 1:length(n_views_seq)){
    print(paste("the", i, "th n_views:", n_views_seq[i]))
    for (j in 1:length(rep_seq)){
      print(paste("the", j, "th rep"))
      
      comp_data = read.csv(paste0("Data/missing/",  "noise_", n_views_seq[i],
                                  "views_", fraction*100, "fraction_rep", rep_seq[j], "missing_", miss, ".csv"))[, -1]
      
      corrup_data = cbind(comp_data[, c(1:2)], comp_data[, proxy_cols(n_views_seq[i])])
      impute_data = cbind(corrup_data[, 1:2], impute_by_mode(corrup_data[, -c(1:2)]))
      est[i, j] = lr(impute_data)
    }
  }
  est
}
main_mf <- function(n_views_seq, rep_seq, m, miss, fraction, Date = "1_7/"){
  est = matrix(0, length(n_views_seq), length(rep_seq))
  for (i in 1:length(n_views_seq)){
    print(paste("the", i, "th n_views:", n_views_seq[i]))
    for (j in 1:length(rep_seq)){
      print(paste("the", j, "th rep"))
      comp_data = read.csv(paste0("Data/missing/",  "noise_", n_views_seq[i],
                                  "views_", fraction*100, "fraction_rep", rep_seq[j], "missing_", miss, ".csv"))[, -1]
      U = read.csv(paste0("Output/",  Date, "noise_", n_views_seq[i],
                          "views_", fraction*100, "fraction_rep", rep_seq[j], "miss", miss, "_U.csv"))
      est[i, j] = lr(cbind(comp_data[, 1:2], U)) 
    }
  }
  est
}
