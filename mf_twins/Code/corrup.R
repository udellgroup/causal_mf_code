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

noise <- function(data, n_views, corrup_p = 0.2, confounder = "gestat10"){
  # create noisy proxy variables for the original dataset.
  # input:
  #   data --> the original data
  #   n_views --> the number of proxies
  #   the fraction of entries that are perturbed 
  x = data[, -c(1:2)]
  n = dim(x)[1]; d = dim(x)[2]
  
  na_frac = apply(is.na(x), 2, sum)/dim(x)[1]
  total_names = confounder
  xx = x
  if (n_views > 1){
    for (i in 1:(n_views - 1)){
      x2 = x[confounder]
      names(x2) = paste0(names(x[confounder]),"_", i + 1)
      total_names = c(total_names, names(x2))
      xx = cbind(xx, x2)
    }
  }
  
  # corrup is a 0-1 matrix recording which entry is perturbed 
  corrup = ifelse(runif(n*length(confounder)*n_views) < corrup_p, T, F)
  corrup = matrix(corrup, nrow = n)
  corrup = as.data.frame(corrup)
  colnames(corrup) = total_names
  
  for (name in total_names){
    values =  unname(unlist(unique(xx[name]))) # number of unique levels for each proxy   
    n_corrup = sum(corrup[, name])
    xx[corrup[, name], name] = sample(values, n_corrup, replace = T)
  }
  data = cbind(data[, 1:2], xx)
  # if (write_file){
  #   write.csv(data,file_path)
  # }
  data
}

noise_gen <- function(data, n_views_seq, corrup_seq, rep_seq, dir_path = paste0("Data/"), write_file = T){
  # create noisy proxy variables for the original dataset.
  # input: 
  #   data: the original dataset 
  #   n_views_seq: the sequence of the number of proxies 
  #   corrup_seq: the sequence of fraction of entries that are corrupted
  #   dir_path: the directory that stores the output noisy proxy data 
  # remark: note that we first generate the largest noisy
  # dataset and then smaller noisy dataset is obtained from only keeping the first a few columns of the
  # largest noisy dataset; this way, we keep the noise patterns the same across the dimensions 
  
  if (!dir.exists(file.path(dir_path))){
    dir.create(dir_path)
  }
  
  conf_name = "gestat10"
  name_rest = setdiff(names(data), conf_name)
  data_rest = data[name_rest]
  
  for (r in rep_seq){
    print(paste("rep", r))
    for (corrup_p in corrup_seq){
      print(paste("corruption", corrup_p))
      data2 = noise(data, max(n_views_seq), corrup_p = corrup_p, confounder = "gestat10")
      for (n_views in n_views_seq){
        print(paste("n_views", n_views))
        file_path = paste0(dir_path, "noise_", n_views, "views_", corrup_p*100, "fraction_rep", r,".csv")
        
        if (n_views == 1){
          select_names = conf_name
        } else {
          select_names = c(conf_name, sapply(1:(n_views-1), function(x) paste0("gestat10","_", x + 1)))
        }
        
        data3 = cbind(data_rest[1:2], data2[select_names])
        write.csv(data3,file_path)
      }
    }   
  }
}


miss_gen <- function(miss_seq, rep_seq, n_views, fraction, complete_data_dir, missing_data_dir){
  # generate missing dataset 
  if (!dir.exists(file.path(complete_data_dir))){
    stop("the complete data directory doesn't exist")
  }
  
  if (!dir.exists(file.path(missing_data_dir))){
    dir.create(missing_data_dir)
  }
  
  for (rep in rep_seq){
    print(paste("reptition", rep))
    for (missing_prop in miss_seq){
      print(paste("missing prop", missing_prop))
      
      data = read.csv(paste0(complete_data_dir, "noise_", n_views, "views_", fraction, "fraction_rep", rep, ".csv"))[, -1]
      
      total_corrup = proxy_cols(n_views)
      n = dim(data[total_corrup])[1]; d = dim(data[total_corrup])[2]
      missing = ifelse(runif(n*d) <= missing_prop, T, F)
      missing = as.data.frame(matrix(missing, n, d))
      colnames(missing) = total_corrup
      
      for (name in total_corrup){
        data[missing[, name], name] = NA
      }
      
      write.csv(data, paste0(missing_data_dir, "noise_", n_views, "views_", fraction, "fraction_rep", rep, "missing_", missing_prop*100,
                             ".csv"))
      
    }
  }
}

#############################
#   No missing value                                              
#############################                                           
### Add noise corruption
data = read.csv(paste0("Data/comp_data.csv"))[, -1]
n_views_seq = c(5, 10)
rep_seq = 1:2
corrup_seq = 0.5
noise_gen(data, n_views_seq, corrup_seq, rep_seq)

###########################
#     Missing Value 
###########################

data_dir = "Data/"
output_dir = "Data/missing/"
for (n_views in c(5, 10)){
  miss_gen(0.3, 1:2, n_views, 50, data_dir, output_dir)
}

