##########
#  Code
##########
helper.R: a collection of all utility codes for the simulations

hd.R: causal effect with matrix factorization for high dimensional setting where p/n --> 1
ld.R: causal effect with matrix factorization for lower dimensional setting where p/n = 1/2

hd_missing.R: causal effect with matrix factorization in the presence of missing values for high dimensional setting where p/n --> 1
ld_missing.R: causal effect with matrix factorization in the presence of missing values for lower dimensional setting where p/n = 1/2

run.sh: the sh file for running all experiments parallelly in hd.R, ld.R, hd_missing.R, ld_missing.R with both Gaussian and Binary proxy variables.  
example: 
	time parallel -j 8 --eta \
	'echo "run dimensional setting {1} with {2} data";R CMD BATCH --vanilla \
	"--args 3 perturbation_{2} recover_pca_{2}_cv {2}\
	1_22" {1} {2}_{1}.Rout'  ::: hd.R ld.R hd_missing.R ld_missing.R  ::: gaussian binary
	remark: 
	parallel -j this argument specifies how many tasks are run in parallel
	--args: the argument fed into the R files above, see any of them for details; in this example
		3 represents 3 repetitions of all experiments
		perturbation_{2} means running the main_cv() or main_cv_missing() with perturbation_gaussian and perturbation_binary as perturbation. 
		recover_pca_{2}_cv means running the main_cv() or main_cv_missing() with recover_pca_gaussian_cv and recover_pca_binary_cv as recover_pca.
		{2} specifies part the names of the saved results 
		1_22 specifies the subdirectory in Output where the results are stored
	{1} specify the R files to run
	{2}_{1}.Rout: specify the name of the Rout logging files 

_.Rout: the logging files recording the execution of the R codes

##########
#  Output 
##########
The output directory storing the results from running the simulations. The directory is organized in terms of the Date, where Date is specified in the Code/_.sh file. For example, in the example Code/run.sh file, the Date is specified to be 1_22, so the Output/1_22 contains all output from running the experiments in the example Code/run.sh file.

##########
# R version
##########
R version 3.4.3
R packages:
urrr_0.2.4      rARPACK_0.11-0   rpca_0.2.3       logisticPCA_0.2  
glmnet_2.0-13   RSpectra_0.12-0  mvtnorm_1.0-7    caret_6.0-78
softImpute_1.4  assertthat_0.2.0


