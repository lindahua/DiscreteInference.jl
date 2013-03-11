# Exact inference on chains

# a chain model
abstract Chain

# a chain model every each node takes value from the same finite set
abstract RegularChain <: Chain

###########################################################
#
#  SimpleChain
#
#  a simple chain is a regular chain, where the joint
#  potential is defined to be
#
#   sum_i a_i(x_i) + sum_i b(x_i, x_{i+1})
#
#   Optionally, the first node can have an additional
#   potential a_f(x_1)
#
#   the last node can have an additional potential
#   a_l(x_n)
#   
##########################################################

type SimpleChain <: RegularChain
    a::Matrix{Float64}      # unary potentials [k x n]
    b::Matrix{Float64}      # binary potentials [k x k]
    
    function SimpleChain(a::Matrix{Float64}, b::Matrix{Float64})
        K = size(a, 1)
        @check_arg_dims size(b) == (K, K)
        new(a, b)
    end
end

length(chain::SimpleChain) = size(chain.a, 2)
cardinality(chain::SimpleChain) = size(chain.a, 1)


function simple_chain_evaluate(r::Vector{Int32}, a::Matrix{Float64}, b::Matrix{Float64})
    
    n = length(r)
    
    # sum over unary potentials
    
    v1 = 0.
    for t = 1 : n
        v1 += a[r[t], t]
    end
    
    # sum over binary potentials
    
    v2 = 0.
    for t = 1 : n-1
        v2 += b[r[t], r[t+1]]
    end
    
    return v1 + v2
end


function simple_chain_viterbi_solve!(r::Vector{Int32}, a::Matrix{Float64}, b::Matrix{Float64})

    K::Int = size(a, 1)
    n::Int = length(r)
    
    # forward pass: F[v,t] = max_u F[u,t-1] + b[u,v] + a[v,t]

    F = Array(Float64, K, n)
    P = Array(Int32, K, n-1)  # for back-tracing
    
    @devec F[:,1] = a[:,1]
    
    for t = 2 : n
        for v = 1 : K
            # indmax(F[:,t-1] + b[:,v])
            mind = 1
            mval = F[1,t-1] + b[1,v]
            for i = 2 : K
                cval = F[i,t-1] + b[i,v]
                if cval > mval
                    mind = i
                    mval = cval
                end
            end
            
            F[v,t] = mval + a[v,t]
            P[v,t-1] = convert(Int32, mind)
        end
    end
    
    # backward trace
    
    i::Int32 = convert(Int32, indmax(F[:,n]))
    objv = F[i,n]
    r[n] = i
    
    for t = n-1 : -1 : 1
        i = P[i, t]
        r[t] = i
    end
    
    return objv::Float64
end

function evaluate(chain::SimpleChain, r::Vector{Int32})
    @check_arg_dims length(r) == length(chain)
    simple_chain_evaluate(r, chain.a, chain.b)
end

function viterbi_solve!(r::Vector{Int32}, chain::SimpleChain)
    @check_arg_dims length(r) == length(chain)
    simple_chain_viterbi_solve!(r, chain.a, chain.b)
end


# generic functions

function viterbi_solve(chain::Chain)
    r = Array(Int32, length(chain))
    v = viterbi_solve!(r, chain)
    r, v::Float64
end


