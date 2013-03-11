module DiscreteInference
    using Devectorize
    
    import Base.length, Base.show
    
    export 
        # common names
        cardinality, evaluate,
    
        # types
        Chain, RegularChain, SimpleChain, 
        
        # algorithms
        viterbi_solve!, viterbi_solve
    
    
    include("base.jl")
    include("chain.jl")
end
