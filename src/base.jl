# basic supporting stuff

macro check_arg_dims(c)
	:( ($(c)) ? nothing : throw(ArgumentError("Incompatible dimensions of arguments.")) )
end

