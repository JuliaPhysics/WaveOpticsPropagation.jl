using WaveOpticsPropagation
using Test
using Zygote
using ChainRulesTestUtils
using IndexFunArrays, NDTools, FourierTools
using FiniteDifferences

@testset "WaveOpticsPropagation.jl" begin
    include("utils.jl")
    include("angular_spectrum.jl")
    include("scalable_angular_spectrum.jl")
    include("fraunhofer.jl")
end

return nothing

