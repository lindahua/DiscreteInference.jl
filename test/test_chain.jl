# test inference on chains

using DiscreteInference
using Test

# simple chain

chain = SimpleChain([5. 3. 2. 6.; 0. 2. 8. 1.], [3. 1.; 1. 5.])

@test length(chain) == 4
@test cardinality(chain) == 2

r, v = viterbi_solve(chain)
@test isequal(r, Int32[1, 2, 2, 1])
@test v == 28
@test evaluate(chain, r) == 28
