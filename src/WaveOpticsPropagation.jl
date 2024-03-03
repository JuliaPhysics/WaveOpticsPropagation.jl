module WaveOpticsPropagation

using EllipsisNotation
using FFTW
using ChainRulesCore
using Zygote
using NDTools
using FourierTools
using IndexFunArrays
using CUDA

"""
    Params{M, M2}

Calls such as `AS = Angular_Spectrum(field, z, λ, L)` will store a `Params` object in `AS.params` that 
contains the physical parameters of the field and the propagated field. 
This is useful to keep track of the physical parameters of the field and the propagated field. 

Has fields to store physical parameters
- `y`: is the y-axis, so the first dimension of the field
- `x`: is the x-axis, so the second dimension of the field
- `yp`: is the y-axis of the propagated field.
- `xp`: is the x-axis of the propagated field.
- `L`: is the physical size of the initial field.
- `Lp`: is the physical size of the propagated field.

"""
struct Params{M, M2}
    y::M
    x::M
    yp::M
    xp::M
    L::M2
    Lp::M2
end

include("utils.jl")
include("propagation.jl")
include("angular_spectrum.jl")
include("shifted_angular_spectrum.jl")
include("scalable_angular_spectrum.jl")
include("fraunhofer.jl")
include("beams.jl")
include("conv.jl")
include("shifted_SAS.jl")



end
