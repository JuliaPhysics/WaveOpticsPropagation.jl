export Fraunhofer


"""
    fraunhofer(field, z, λ, L)

"""
function fraunhofer(U, z, λ, L; skip_final_phase=true)
    @assert size(U, 1) == size(U, 2)
    L_new = λ * z / L * size(U, 1)
	Ns = size(U)[1:2]
    
    p = ChainRulesCore.@ignore_derivatives plan_fft(U, (1,2))
    
    if skip_final_phase
        out = fftshift(p * ifftshift(U)) ./ √(size(U, 1) * size(U, 2))
    else    
        k = eltype(U)(2π) / λ
        # output coordinates
        y = similar(U, real(eltype(U)), (Ns[1], 1))
        ChainRulesCore.@ignore_derivatives y .= (fftpos(L, Ns[1], CenterFT))
    	x = similar(U, real(eltype(U)), (1, Ns[2]))
        ChainRulesCore.@ignore_derivatives x .= (fftpos(L, Ns[2], CenterFT))'
        phasefactor = (-1im) .* exp.(1im * k / (2 * z) .* (x.^2 .+ y.^2)) 
        out = phasefactor .* fftshift(p * ifftshift(U)) ./ √(size(U, 1) * size(U, 2))
    end
    
    return out
end

"""
    Fraunhofer(U, z, λ, L; skip_final_phase=true)


This returns a function for efficient reuse of pre-calculated kernels.
This function then returns the electrical field with physical length `L` and wavelength `λ` 
propagated with the Fraunhofer propagation (a single FFT) by the propagation distance `z`.
This is based on a far field approximation.

# Arguments
* `U`: Input field
* `z`: propagation distance
* `λ`: wavelength of field
* `L`: field size indicating field size

# Keyword Arguments
* `skip_final_phase=true` skip the final phase which is multiplied to the propagated field at the end 
See [`fraunhofer`](@ref) for the full documentation.


# Example
```jldoctest
julia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;

julia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;

julia> f = Fraunhofer(field, 4f-3, 632f-9, 100f-6);

julia> res = f(field);

julia> f.params.Lp[1] / 100f-6
64.71681f0

julia> 4f-3 * 632f-9 * 256 / (100f-6)^2
64.71681f0
```
"""
function Fraunhofer(U::AbstractArray{CT}, _z::Number, _λ, _L; skip_final_phase=true) where CT
    λ = real(CT)(_λ)
    z = real(CT)(_z)
    L = real(CT).(_L isa Number ? (_L, _L) : _L)
    L_new = λ .* z ./ L .* size(U)[1:2]
	Ns = size(U)[1:2]
   
    k = eltype(U)(2π) / λ
    # output coordinates
    y = similar(U, real(eltype(U)), (Ns[1], 1))
    y .= (fftpos(L[1], Ns[1], CenterFT))
    x = similar(U, real(eltype(U)), (1, Ns[2]))
    x .= (fftpos(L[2], Ns[2], CenterFT))'
    yp = similar(U, real(eltype(U)), (Ns[1], 1))
    yp .= (fftpos(L_new[1], Ns[1], CenterFT))
    xp = similar(U, real(eltype(U)), (1, Ns[2]))
    xp .= (fftpos(L_new[2], Ns[2], CenterFT))'

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
