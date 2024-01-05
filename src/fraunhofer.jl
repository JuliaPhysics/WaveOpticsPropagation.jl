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
* `skip_final_phase` skip the final phase which is multiplied to the propagated field at the end 

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
        phasefactor = exp.(1im * k / (2 * z) .* (x.^2 .+ y.^2)) 
        out = phasefactor .* fftshift(p * ifftshift(U)) ./ √(size(U, 1) * size(U, 2))
    end
    
    return out, (;L=L_new) 
end

"""
    Fraunhofer(U, z, λ, L; skip_final_phase=true)


This returns a function for efficient reuse of pre-calculated kernels.
See [`fraunhofer`](@ref) for the full documentation.

"""
function Fraunhofer(U, z, λ, L; skip_final_phase=true)
    @assert size(U, 1) == size(U, 2)
    L_new = λ * z / L * size(U, 1)
	Ns = size(U)[1:2]
    
    if skip_final_phase
        phasefactor = nothing
    else    
        k = eltype(U)(2π) / λ
        # output coordinates
        y = similar(U, real(eltype(U)), (Ns[1], 1))
    	y .= (fftpos(L, Ns[1], CenterFT))
    	x = similar(U, real(eltype(U)), (1, Ns[2]))
    	x .= (fftpos(L, Ns[2], CenterFT))'
        phasefactor = 1 / (1im * λ * z) .* exp.(1im * k / (2 * z) .* (x.^2 .+ y.^2)) 
    end

    buffer = zero.(U)

    FFTplan = plan_fft!(buffer, (1,2))

    return FraunhoferOp{typeof(L), typeof(buffer), typeof(phasefactor), 
                        typeof(FFTplan)}(buffer, phasefactor, L_new, FFTplan)
end

struct FraunhoferOp{T, B, PF, P}
    buffer::B
    phasefactor::PF
    L::T
    FFTplan::P
end

function (fraunhofer::FraunhoferOp)(field)
    buffer = fraunhofer.buffer
    ifftshift!(buffer, field)
    fraunhofer.FFTplan * buffer
    if !isnothing(fraunhofer.phasefactor)
        buffer .*= fraunhofer.phasefactor
    end
    buffer ./= √(size(field, 1) * size(field, 2))
    out = fftshift(buffer)
    return out, (; fraunhofer.L)
end


function ChainRulesCore.rrule(f::FraunhoferOp, U)
    field_and_tuple = f(U)

    function f_pullback(ȳ)
        buffer = f.buffer
        ifftshift!(buffer, ȳ.backing[1])
        buffer .*= √(size(U, 1) * size(U, 2))
        res = fftshift(inv(f.FFTplan) * buffer)
        if !isnothing(f.phasefactor)
            res .*= conj.(f.phasefactor)
        end
        return NoTangent(), res
    end
    return field_and_tuple, f_pullback
end
