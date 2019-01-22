using JLD
using CSV
using Logging
using RCall
using ArgParse
@everywhere using DataFrames
@everywhere using LowRankModels
@everywhere using ParallelDataTransfer

const global corrup = ["gestat10"]

@everywhere import LowRankModels.embedding_dim
@everywhere embedding_dim(l::Array{Any,1}) = sum(map(embedding_dim, l))

function construct_loss_noise(n_views::Int64, corrup::Array{String, 1})
	# this function constructs the loss functions for the matrix factorization;
	#	here the confounder is "gestat10", which is a categorical variable, so multinomial loss with
	#	10 levels are used 
	# Input:
	# 	corrup --> the list of confounders, it's "gestat10" in this example 
	#	n_views --> the number of proxies available for each confounder 
	# Output:
	#	loss: the list of loss function used in matrix factorization
	#	cols: the column names in the data frame corresponding to all proxies 

	# total_corrup is a list containing the 
	total_corrup = []
	for n in corrup
	   append!(total_corrup, [string(n, "_", i+1) for i in 1:(n_views - 1)] )
	end
	total_corrup = append!(total_corrup, corrup)

	#loss = fill(MultinomialLoss(10), length(total_corrup))
	loss = fill(MultinomialLoss(10), length(total_corrup))
	cols = total_corrup

	[loss, cols]
end



function construct_x_noise(x_data::DataFrame, cols::Array{Any, 1})
	# extract the proxy matrix from the whole dataset
	#	cols is the list of all noisy proxies 
	cols = parse.(cols)
	x = x_data[cols]
	x
end


function mc(x::DataFrame, r_seq::Array{Int64, 1}, obs::Array{Tuple{Int64,Int64},1};
	offset = false, initialization = false, nomissing = false)
	train_error = zeros(length(r_seq)); test_error = zeros(length(r_seq))
	for r in 1:length(r_seq)
	    info("the ($r)th r_seq: $(r_seq[r])")
	    model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[r];
	               offset = false, scale = false, obs = obs)
	    train_err, test_err, train_glrms, test_glrms = cross_validate(model, nfolds=5)
	    train_error[r] = mean(train_err); test_error[r] = mean(test_err);
	end
	minval, minind = findmin(test_error)
	model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[minind];
	               offset = false, scale = false, obs = obs)
	U, V, ch = fit!(model)
	U = convert(Array{Float64, 2}, U)
	[U', V', ch, r_seq[minind], test_error] 
end


function mc_cv(x::DataFrame, r_seq::Array{Int64, 1}, obs::Array{Tuple{Int64,Int64},1}, loss::Array{Any, 1}; 
	offset = false, initialization = false)

	train_error = zeros(length(r_seq)); test_error = zeros(length(r_seq))
	for r in 1:length(r_seq)
		info("the ($r)th r_seq: $(r_seq[r])")
	    model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[r];
	               offset = offset, obs = obs)
	    if initialization 
	    	train_err, test_err, train_glrms, test_glrms = cross_validate(model, nfolds=5; init = init_svd!)
	    else
	    	train_err, test_err, train_glrms, test_glrms = cross_validate(model, nfolds=5)
	    end
	    train_error[r] = mean(train_err); test_error[r] = mean(test_err); 
	end
	minval, minind = findmin(test_error)
	model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[minind];
	               offset = offset, obs = obs)
	U, V, ch = fit!(model)
	U = convert(Array{Float64, 2}, U)
	[U', V', ch, r_seq[minind], test_error]
end



function mc_cv_parallel(x::DataFrame, r_seq::Array{Int64, 1}, obs::Array{Tuple{Int64,Int64},1}, loss::Array{Any, 1}; 
	offset = false, initialization = false)
	# this function implements matrix factorization with the rank r chosen by cross-validation where cross-validation 
	#	is implemented in parallel 
	# input:
	#	x: the proxy matrix to be factorized
	#	r_seq: the candidate ranks used in matrix factorization; and the best one is chosen by cross validation 
	#	obs: the list of nonmissing observations
	#	loss: the list of loss functions used in matrix factorization where each element corresponds to a variable
	#		in the proxy matrix
	#	offset: whether bias term is used in matrix factorization
	# 	initialization: whether initialization is used in matrix factorization 

	
	info("send data to workers\n")
	sendto(workers(), x = x, r_seq = r_seq, obs = obs, loss = loss)

	info("create sharedarray\n")
	train_error = SharedArray{Float64, 1}(length(r_seq))
	test_error = SharedArray{Float64, 1}(length(r_seq))

	# cross-validation 
	@sync @parallel for r in 1:length(r_seq)
		info("the ($r)th r_seq: $(r_seq[r])")
	    model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[r];
	               offset = offset, obs = obs)
	    if initialization 
	    	train_err, test_err, train_glrms, test_glrms = cross_validate(model, nfolds=5; init = init_svd!)
	    else
	    	train_err, test_err, train_glrms, test_glrms = cross_validate(model, nfolds=5)
	    end
	    train_error[r] = mean(train_err); test_error[r] = mean(test_err); 
	end
	# choose the best r that achieves the minimum testing error 
	minval, minind = findmin(test_error)
	info("fit the best model\n")
	model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[minind];
	               offset = offset, obs = obs)
	U, V, ch = fit!(model)
	U = convert(Array{Float64, 2}, U)
	[U', V', ch, r_seq[minind], test_error]
end

function mc_impute_parallel(x::DataFrame, r_seq::Array{Int64, 1}, obs::Array{Tuple{Int64,Int64},1}, loss::Array{Any, 1}; 
	offset = false, initialization = true)

	# this function implements matrix factorization in the presence of missing values with the rank r chosen by cross-validation where cross-validation 
	#	is implemented in parallel 
	# input:
	#	x: the proxy matrix to be factorized; it can contain missing values 
	#	r_seq: the candidate ranks used in matrix factorization; and the best one is chosen by cross validation 
	#	obs: the list of nonmissing observations
	#	loss: the list of loss functions used in matrix factorization where each element corresponds to a variable
	#		in the proxy matrix
	#	offset: whether bias term is used in matrix factorization
	# 	initialization: whether initialization is used in matrix factorization 
	# output:
	#	completion --> the imputed matrix
	#		if x[i, j] is missing, then completion[i, j] is computed from matrix factorization
	#		if x[i, j] is nonmissing, then completion[i, j] = x[i, j]
	
	info("send data to workers\n")
	sendto(workers(), x = x, r_seq = r_seq, obs = obs, loss = loss)

	info("create sharedarray\n")
	train_error = SharedArray{Float64, 1}(length(r_seq))
	test_error = SharedArray{Float64, 1}(length(r_seq))
	@sync @parallel for r in 1:length(r_seq)
		info("the ($r)th r_seq: $(r_seq[r])")
	    model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[r];
	               offset = offset, obs = obs)
	    if initialization 
	    	train_err, test_err, train_glrms, test_glrms = cross_validate(model, nfolds=5; init = init_svd!)
	    else
	    	train_err, test_err, train_glrms, test_glrms = cross_validate(model, nfolds=5)
	    end
	    train_error[r] = mean(train_err); test_error[r] = mean(test_err); 
	end
	minval, minind = findmin(test_error)
	info("fit the best model\n")
	model = GLRM(x, loss, QuadReg(.01), QuadReg(.01), r_seq[minind];
	               offset = offset, obs = obs)
	U, V, ch = fit!(model)
	U = convert(Array{Float64, 2}, U)

	n, p = size(x);
	x_imputed = impute(model)
	completion = fill(0, (n, p));
	for i in 1:n
	    for j in 1:p 
	        completion[i, j] = ismissing(x[i, j])? x_imputed[i, j] : x[i, j]
	    end 
	end

	[U', V', completion, ch, r_seq[minind], test_error]
end

function main(r_seq::Array{Int64, 1}, data_path::String, output_path::String, main_type::String; initialization = true)
	if !isdir(output_path)
	    mkdir(output_path)
	end 

	# extract the information for each data file from the data names
	#	structure of the name of the files:
	#		e.g., noise_50views_50fraction_rep22.csv
	#	for this data file, 50 noisy proxies are available, 50% of entries in the proxy matrix are randomly perturbed, 
	#	rep22 refers to the 22th repetition of the experiment with this 50views_50fraction setting.
	numbers = matchall(r"\d+", data_path)
	numbers = map(x->parse(Int64,x),numbers)
	# n_views: number of proxies
	# corrup_p: the fraction of entries that are corrupted with noise 
	# rep: the repetition number for the n_view and corrup_p settings
	n_views = numbers[end-2]
	corrup_p = numbers[end-1]/100
	rep = numbers[end]

	info("n_views: $n_views\n")
	info("corrup_p: $corrup_p\n")

	loss, cols = construct_loss_noise(n_views, corrup)
	data = readtable(data_path)[:, 2:end];
	x = construct_x_noise(data, cols);
	size_x = size(x)
	info("size of x:", "($size_x[1], $size_x[2])", "\n")

	# obs is the list of nonmissing observations 
	obs = observations(x);

	result, t, bytes, gctime = @timed mc_cv_parallel(x, r_seq, obs, loss; initialization = initialization)
	
	save(string(output_path, string("noise_", n_views,"views_", Int64(corrup_p*100), "fraction_rep", rep, ".jld")), "U", result[1], "V", result[2], "ch", result[3], "r", result[4]
	, "test_error", result[5], "t", t, "bytes", bytes, "gctime", gctime)
	[result, t, bytes, gctime]
end

function main_missing(r_seq::Array{Int64, 1}, data_path::String, output_path::String, main_type::String; initialization = true)
	if !isdir(output_path)
	    mkdir(output_path)
	end 

	numbers = matchall(r"\d+", data_path)
	numbers = map(x->parse(Int64,x),numbers)
	n_views = numbers[end-3]
	corrup_p = numbers[end-2]/100
	rep = numbers[end-1]
	miss_prop = numbers[end]

	info("n_views: $n_views\n")
	info("corrup_p: $corrup_p\n")

	loss, cols = construct_loss_noise(n_views, corrup)
	data = readtable(data_path)[:, 2:end];
	x = construct_x_noise(data, cols);
	size_x = size(x)
	info("size of x:", "($size_x[1], $size_x[2])", "\n")
	obs = observations(x);

	result, t, bytes, gctime = @timed mc_impute_parallel(x, r_seq, obs, loss; initialization = initialization)
	
	save(string(output_path, string("noise_", n_views,"views_", Int64(corrup_p*100), "fraction_rep", 
		rep, "miss", miss_prop, ".jld")), "U", result[1], "V", result[2], "A", result[3], "ch", result[4], "r", result[5]
	, "test_error", result[6], "t", t, "bytes", bytes, "gctime", gctime)
	[result, t, bytes, gctime]
end




function parse_commandline()
	# this function specifies the commandline input arguments 
    s = ArgParseSettings()

    @add_arg_table s begin
    	"--data"
            help = "the data file"
            arg_type = String
        "--date"
            help = "date of the experiment"
            arg_type = String
	end

    return parse_args(s)
 end


function parse_result(n_views_seq::Array{Int64, 1}, rep_seq::Array{Int64, 1}, fraction::Int64, sub_dir::String; output_dir = "../Output/")
	# this function summarizes the output from all experiments specified by n_views_seq, fraction, rep_seq

	run_time = Array{Float64}(length(n_views_seq), length(rep_seq))
	best_rank = Array{Float64}(length(n_views_seq), length(rep_seq))
	test_error_ratio = Array{Float64}(length(n_views_seq), length(rep_seq))
	test_error_min = Array{Float64}(length(n_views_seq), length(rep_seq))
	history_obj_ratio = Array{Float64}(length(n_views_seq), length(rep_seq))
	history_obj_end = Array{Float64}(length(n_views_seq), length(rep_seq))
	for i in 1:length(n_views_seq)
	    for j in 1:length(rep_seq)
	        file_name = string(output_dir, sub_dir, "noise_", n_views_seq[i], "views_", fraction, "fraction_rep", rep_seq[j], ".jld")
	        result = load(file_name)
	        run_time[i, j] = result["t"] 
	        best_rank[i, j] = result["r"]
	        test_error = convert(Array{Float64, 1}, result["test_error"])
	        test_error_ratio[i, j] = maximum(test_error)/minimum(test_error)
	        test_error_min[i, j] = minimum(test_error)
	        history_obj_ratio[i, j] = result["ch"].objective[1]/result["ch"].objective[end]
	        history_obj_end[i, j] = result["ch"].objective[end]
	        U = convert(DataFrame, result["U"])
	        CSV.write(string(output_dir, sub_dir, "noise_", n_views_seq[i], "views_", fraction, "fraction_rep", rep_seq[j], ".csv"), U)
	    end 
	end
	[run_time, best_rank, test_error_ratio, test_error_min, history_obj_ratio, history_obj_end] 
end



function parse_result_by_missing(miss_prop::Array{Int64, 1}, rep_seq::Array{Int64, 1}, n_views::Int64, noise_fraction::Int64, sub_dir::String; output_dir = "/home/xm77/causal/JL_sim/Output/twins/")
	run_time = Array{Float64}(length(miss_prop), length(rep_seq))
	best_rank = Array{Float64}(length(miss_prop), length(rep_seq))
	test_error_ratio = Array{Float64}(length(miss_prop), length(rep_seq))
	test_error_min = Array{Float64}(length(miss_prop), length(rep_seq))
	history_obj_ratio = Array{Float64}(length(miss_prop), length(rep_seq))
	history_obj_end = Array{Float64}(length(miss_prop), length(rep_seq))
	for i in 1:length(miss_prop)
		    for j in 1:length(rep_seq)
		    	file_name = string(output_dir, sub_dir, "noise_", n_views, "views_", noise_fraction, 
		    			        	"fraction_rep", rep_seq[j], "miss", miss_prop[i],".jld")
	            result = load(file_name)
		        run_time[i, j] = result["t"] 
		        best_rank[i, j] = result["r"]
		        test_error = convert(Array{Float64, 1}, result["test_error"])
		        test_error_ratio[i, j] = maximum(test_error)/minimum(test_error)
		        test_error_min[i, j] = minimum(test_error)
		        history_obj_ratio[i, j] = result["ch"].objective[1]/result["ch"].objective[end]
		        history_obj_end[i, j] = result["ch"].objective[end]
		        V = convert(DataFrame, result["V"])
		        output_name_V = string(output_dir, sub_dir, "noise_", n_views, "views_", 
		  		        	noise_fraction, "fraction_rep", rep_seq[j], "miss", miss_prop[i],"_V",".csv")
		        A = convert(DataFrame, result["A"])
		        output_name_A = string(output_dir, sub_dir, "noise_", n_views, "views_", 
		        		        	noise_fraction, "fraction_rep", rep_seq[j], "miss", miss_prop[i],"_A",".csv")
		    
		    	CSV.write(output_name_V, V)
		    	CSV.write(output_name_A, A)
		        U = convert(DataFrame, result["U"])
		        output_name_U = string(output_dir, sub_dir, "noise_", n_views, "views_", 
		        		        	noise_fraction, "fraction_rep", rep_seq[j], "miss", miss_prop[i],"_U",".csv") 
		        CSV.write(output_name_U, U) 
		    end 
	end
	[run_time, best_rank, test_error_ratio, test_error_min, history_obj_ratio, history_obj_end] 
end
