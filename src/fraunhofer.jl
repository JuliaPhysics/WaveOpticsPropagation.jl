export fraunhofer
export Fraunhofer


"""
    fraunhofer(field, z, λ, L)

Returns the electrical field with physical length `L` and wavelength `λ` 
propagated with the Fraunhofer propagation (a single FFT) by the propagation distance `z`.
This is based on a far field approximation `z >> λ`

This method is efficient but to save memory and avoiding recalculating some arrays (such as the phase kernel), see [`Fraunhofer`](@ref). 

# Arguments
* `field`: Input field
* `z`: propagation distance
* `λ`: wavelength of field
* `L`: field size indicating field size

# Keyword Arguments
* `skip_final_phase=true` skip the final phase which is multiplied to the propagated field at the end 


# Example
```jldoctest
julia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;

julia> res, t = fraunhofer(field, 4f-3, 632f-9, 100f-6)
(ComplexF64[0.00390625 + 0.0im 0.003905073506757617 - 9.586417581886053e-5im … 0.003901544725522399 + 0.0001916706096380949im 0.003905073506757617 + 9.58640594035387e-5im; 0.003905073506757617 - 9.586417581886053e-5im 0.003901544725522399 - 0.0001916706096380949im … 0.003905073506757617 + 9.58640594035387e-5im 0.00390625 - 1.1641532182693481e-10im; … ; 0.003901544725522399 + 0.00019167049322277308im 0.003905073506757617 + 9.586406667949632e-5im … 0.003887440310791135 + 0.0003828795161098242im 0.0038956659846007824 + 0.00028736155945807695im; 0.003905073506757617 + 9.586417581886053e-5im 0.00390625 - 1.5902765215791703e-12im … 0.0038956659846007824 + 0.0002873614430427551im 0.003901544725522399 + 0.0001916706096380949im], (L = 0.006471681f0,))

julia> t.L / 100f-6
64.71681f0

julia> 4f-3 *  632f-9 * 256 / (100f-6)^2
64.71681f0
```

"""
function fraunhofer(U, z, λ, L; skip_final_phase=true)
    @assert size(U, 1) == size(U, 2)
    L_new = λ * z / L * size(U, 1)
	Ns = size(U)[1:2]
    
    p = Zygote.@ignore plan_fft(U, (1,2))
    
    if skip_final_phase
        out = fftshift(p * ifftshift(U)) ./ √(size(U, 1) * size(U, 2))
    else    
        k = eltype(U)(2π) / λ
        # output coordinates
        y = similar(U, real(eltype(U)), (Ns[1], 1))
    	Zygote.@ignore y .= (fftpos(L, Ns[1], CenterFT))
    	x = similar(U, real(eltype(U)), (1, Ns[2]))
    	Zygote.@ignore x .= (fftpos(L, Ns[2], CenterFT))'
        phasefactor = (-1im) .* exp.(1im * k / (2 * z) .* (x.^2 .+ y.^2)) 
        out = phasefactor .* fftshift(p * ifftshift(U)) ./ √(size(U, 1) * size(U, 2))
    end
    
    return out
end

"""
    Fraunhofer(U, z, λ, L; skip_final_phase=true)


This returns a function for efficient reuse of pre-calculated kernels.
See [`fraunhofer`](@ref) for the full documentation.


# Example
```jldoctest
julia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;

julia> f, t = Fraunhofer(field, 4f-3, 632f-9, 100f-6);

julia> f(field)
(ComplexF32[0.00390625f0 + 0.0f0im 0.0039050735f0 - 9.5864176f-5im … 0.0039015447f0 + 0.00019167061f0im 0.0039050735f0 + 9.586406f-5im; 0.0039050735f0 - 9.5864176f-5im 0.0039015447f0 - 0.00019167061f0im … 0.0039050735f0 + 9.586406f-5im 0.00390625f0 - 1.1641532f-10im; … ; 0.0039015447f0 + 0.0001916705f0im 0.0039050735f0 + 9.586407f-5im … 0.0038874403f0 + 0.00038287952f0im 0.003895666f0 + 0.00028736156f0im; 0.0039050735f0 + 9.5864176f-5im 0.00390625f0 - 1.5902765f-12im … 0.003895666f0 + 0.00028736144f0im 0.0039015447f0 + 0.00019167061f0im], (L = 0.006471681f0,))

julia> t.L / 100f-6
64.71681f0

julia> 4f-3 *  632f-9 * 256 / (100f-6)^2
64.71681f0
```
"""
function Fraunhofer(U, z, λ, L; skip_final_phase=true)
    @assert size(U, 1) == size(U, 2)
    L_new = λ * z / L
	Ns = size(U)[1:2]
   
    k = eltype(U)(2π) / λ
    # output coordinates
    y = similar(U, real(eltype(U)), (Ns[1], 1))
    y .= (fftpos(L, Ns[1], CenterFT))
    x = similar(U, real(eltype(U)), (1, Ns[2]))
    x .= (fftpos(L, Ns[2], CenterFT))'
    yp = similar(U, real(eltype(U)), (Ns[1], 1))
    yp .= (fftpos(L_new, Ns[1], CenterFT))
    xp = similar(U, real(eltype(U)), (1, Ns[2]))
    xp .= (fftpos(L_new, Ns[2], CenterFT))'

    if skip_final_phase
        phasefactor = nothing
    else    
        phasefactor = (-1im) .* exp.(1im * k / (2 * z) .* (x.^2 .+ y.^2)) 
    end

    buffer = zero.(U)

    FFTplan = plan_fft!(buffer, (1,2))
    params = Params(y, x, yp, xp, L, L_new)
    return FraunhoferOp(buffer, phasefactor, params, FFTplan)
end

struct FraunhoferOp{B, PF, P, M, M2}
    buffer::B
    phasefactor::PF
    params::Params{M, M2}
    FFTplan::P
end

function (fraunhofer::FraunhoferOp)(field)
    buffer = fraunhofer.buffer
    ifftshift!(buffer, field)
    fraunhofer.FFTplan * buffer
    buffer ./= √(size(field, 1) * size(field, 2))
    out = fftshift(buffer)
    if !isnothing(fraunhofer.phasefactor)
        out .*= fraunhofer.phasefactor
    end
    return out
end


function ChainRulesCore.rrule(f::FraunhoferOp, U)
    field_and_tuple = f(U)

    function f_pullback(ȳ)
        buffer = f.buffer
        y2 = ȳ
        if !isnothing(f.phasefactor)
            y2 = y2 .* conj.((f.phasefactor))
        end
        ifftshift!(buffer, y2)
        buffer .*= √(size(U, 1) * size(U, 2))
        res = fftshift(inv(f.FFTplan) * buffer)
        return NoTangent(), res
    end
    return field_and_tuple, f_pullback
end
