# number of cores used in parallel computing 
addprocs(3)

const global code_dir = "../Code/"
const global data_dir = "../Data/"
const global output_dir = "../Output/"
include(string(code_dir, "helper.jl"))

# parsed_args = parse_commandline()
# const global Date = string(parsed_args["date"], "/")
# const global data_name = parsed_args["data"]
parsed_args = parse_commandline()
const global Date = string(parsed_args["date"], "/")
const global data_name = parsed_args["data"]

data_path = string(data_dir, "missing/", data_name)
output_path = string(output_dir, Date)
r_seq = [1, 2, 3, 4, 5]

if !isdir(output_path)
    mkdir(output_path)
end 

log_name = string(output_path, split(split(data_path, "/")[end], ".")[1], "_log.txt")
if isfile(log_name)
   rm(log_name)
end
Logging.configure(level = INFO, filename = log_name)  

result, t, bytes, gctime = main_missing(r_seq, data_path, output_path, "parallel"; initialization = true)
