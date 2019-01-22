# number of cores used in parallel computing 
addprocs(3)

const global code_dir = "../Code/"
const global data_dir = "../Data/"
const global output_dir = "../Output/"
include(string(code_dir, "helper.jl"))


# command line argument:
#	Date: the name of the sub-directory under the output_dir that is used for the ouput; it's usually 
#		named in the Date when the experiment runs
#	data_name: the name of the proxy data file used in matrix factorization 
parsed_args = parse_commandline()
const global Date = string(parsed_args["date"], "/")
const global data_name = parsed_args["data"]

data_path = string(data_dir, data_name)
output_path = string(output_dir, Date)
# r_seq: the candidate ranks used in matrix factorization where the best rank is chosen via cross-validation 
r_seq = [1, 2, 3, 4]

if !isdir(output_path)
    mkdir(output_path)
end 

# logging 
log_name = string(output_path, split(split(data_path, "/")[end], ".")[1], "_log.txt")
if isfile(log_name)
   rm(log_name)
end
Logging.configure(level = INFO, filename = log_name) 

result, t, bytes, gctime = main(r_seq, data_path, output_path, "parallel"; initialization = true)
