source("helper.R")

args <- commandArgs(TRUE)
rep_number = as.numeric(args[1])
perturbation = get(args[2])
recover_pca = get(args[3])
save_name = args[4]
Date = args[5]
# nseq = seq(150, 1500, by = 150)
# pseq = nseq - 50
nseq = c(30, 40)
pseq = nseq - 5
treat_coef = c(1, 2, 2, 2, 2)
out_coef = c(-2, 3, -2, -3, -2)




result = main_cv_missing(nseq, pseq, treat_coef, out_coef, true_te = 2, true_r = 5, r_seq = c(1,3,5,7,9), noise_level = 5, rep_number = rep_number, 
                 funs = c(regress_unbiased, regress_svd, regress_corrupted_lasso, regress_corrupted_ridge, regress_corrupted_ols), 
                 perturbation. = perturbation,  recover_pca. = recover_pca)

output_dir = paste0("../Output/", Date, "/")
if (!dir.exists(output_dir)){
	dir.create(output_dir)
}

saveRDS(result, paste0("../Output/", Date, "/", save_name, "_hd_missing.rds"))