time parallel -j 8 --eta \
'echo "run dimensional setting {1} with {2} data";R CMD BATCH --vanilla \
"--args 3 perturbation_{2} recover_pca_{2}_cv {2}\
 1_22" {1} {2}_{1}.Rout'  ::: hd.R ld.R hd_missing.R ld_missing.R  ::: gaussian binary

