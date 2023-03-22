using WaveOpticsPropagation
using Test
using Zygote
using ChainRulesTestUtils
using IndexFunArrays, NDTools, FourierTools

@testset "WaveOpticsPropagation.jl" begin
    include("utils.jl")
    include("angular_spectrum.jl")
end

return nothing

