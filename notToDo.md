file: /Users/Melchior@Magi/University/PhD/DeepEGB/configs/default.yaml
what_to_do_or_explain:{
    1. extend the unary_operators to all analytical functions from mathematical analysis
    2. extend the list of seed_expressions for V and xi with references to literature (can use arxiv mcp)
}

file: /Users/Melchior@Magi/University/PhD/DeepEGB/src/deepegb/analysis/analyze.py
what_to_do_or_explain:{
    1. where does r/(-8nT) comes from in analyze_egb_model function? need to 
        1.1. explain this
        1.2. correct this because this relation must not hold for arbitrary case as far as i am concerned
    2. need to make default values of xi_expr argument 
        2.1. to be defined same way
        2.2. to be nontrivial by default because we are aimed by default to EGB gravity
    3. how correct it is to set N=55 by default? N must be defined by \epsilon=1 and it can be optimised from constants of the model. the idea is to keep N in range [50,60] as literature states
    4. not clean what n_decades is and how good idea it is to have default T_reh_GeV to be 1.0e15. any literature backing this up?
}

file: /Users/Melchior@Magi/University/PhD/DeepEGB/src/deepegb/analysis/plot.py
what_to_do_or_explain:{
    1. how correct it is to set N=55 by default? N must be defined by \epsilon=1 and it can be optimised from constants of the model. the idea is to keep N in range [50,60] as literature states
    2. in description part of the file next to 6-panel figure need to also add the gravitational wave background spectrum plot too
    3. not clear what n_decades and n_k stand for
    4. not clear why phi_range varys from plot to plot

}

file: /Users/Melchior@Magi/University/PhD/DeepEGB/src/deepegb/physics/kernel.jl
what_to_do_or_explain:{
    1. all functions must use same target values. for example here we see in function egb_chi2_v_only target_ns::Real = 0.965 while it must pull the default value from same place the other python scripts do. same for all other variables/arguments
    2. terrible choise of equations for physical \epsilon, n_s, r, and pretty much everything else. this are approximations that are too broad. need real equations from literature and it would be great for this equations to be analytical and not approximate
}

file: /Users/Melchior@Magi/University/PhD/DeepEGB/src/deepegb/physics/diagnostics.py
what_to_do_or_explain:{
    1. default values for example in chi2_full_breakdown may be different than in other files. need to make them all to be the same accross all files
}

file: /Users/Melchior@Magi/University/PhD/DeepEGB/src/deepegb/physics/egb_background.py
what_to_do_or_explain:{
    1. why is hubble_from_constraint using some predefined denom, constants, why is it picking the positive real root closest to the GR seed? this is some terrible mistake rather than a useful function
    2. same for _step_rhs function. unclear function, when analytic ones are known
    3. this whole block must be rewritten in a way to utilise real analytical equations from literature, not this made-up approximations that have nothing in common with reality
}

file: /Users/Melchior@Magi/University/PhD/DeepEGB/src/deepegb/physics/egb_perturbations.py
what_to_do_or_explain:{
    1. why using slow roll in calculations if we can use full system? everywhere where we can use full system with exact equations - we should use them.
}

file: /Users/Melchior@Magi/University/PhD/DeepEGB/src/deepegb/physics/relic_gw.py
what_to_do_or_explain:{
    1. unclear choise of equations. in each function need to have a reference to a paper/equation. where possible we must use analytical equations and not approximations (especially unclear approximations without any explanation where it comes from)
}