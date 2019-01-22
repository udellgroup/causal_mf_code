source("Code/utility.R")
#########################
#  No missing value 
#########################

library(MatchIt)
library(optmatch)
library(doParallel)
options("optmatch_max_problem_size" = Inf)
sub_dir = "twins_total/miss_by_views/"

n_views_seq = c(5)
rep_seq = 1;
fraction = 0.5
Date = "1_7/"
miss = 0

# propensity score matching
pm = propensity_match(n_views_seq, rep_seq, fraction, Date, miss = 0)

# matching on the covariates
cores = 3
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)
getDoParWorkers()
fm = full_match(n_views_seq, rep_seq, fraction, Date, miss = 0)
stopCluster(cl)

# doubly robust 
dr = compute_dr_by_views(n_views_seq, rep_seq, miss = 0, fraction = 0.5, Date = "1_7/")

# propensity score reweighting 
pw = compute_pw_by_views(n_views_seq, rep_seq, miss = 0, fraction = 0.5, Date = "1_7/")

########################
# Missing Value 
########################
library(mice)

# mice 
cores = 3
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)
getDoParWorkers()

n_views_seq = c(5)
rep_seq = 1;
fraction = 0.5
Date = "1_7/"
miss = 30
m = 5  # number of imputation 
mice = main_mice(n_views_seq, rep_seq, m, miss, fraction, Date = "1_7/")
stopCluster(cl)

# impute by mode 
mode = main_mode(n_views_seq, rep_seq, m, miss, fraction, Date = "1_7/")

# matrix factorization 
mf = main_mf(n_views_seq, rep_seq, m, miss, fraction, Date = "1_7/")

























