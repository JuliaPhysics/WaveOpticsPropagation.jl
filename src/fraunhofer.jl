export fraunhofer
export Fraunhofer


"""
    fraunhofer(field, z, λ, L)

Returns the electrical field with physical length `L` and wavelength `λ` propagated with the angular spectrum 
method of plane waves (AS) by the propagation distance `z`.
This is based on a far field approximation `z ≫ λ`

# Arguments
* `field`: Input field
* `z`: propagation distance
* `λ`: wavelength of field
* `L`: field size indicating field size

# Keyword Arguments
* `skip_final_phase` skip the final phase which is multiplied to the propagated field at the end 

"""
function fraunhofer(field, z, λ, L)

end


function Fraunhofer(U, z, λ, L; skip_final_phase=true)
    @assert size(U, 1) == size(U, 2)
    L_new = λ * z / L * size(U, 1)
	Ns = size(U)[1:2]

    k = eltype(U)(2π) / λ
    # output coordinates
    y = similar(U, real(eltype(U)), (Ns[1], 1))
	y .= (fftpos(L, Ns[1], CenterFT))
	x = similar(U, real(eltype(U)), (1, Ns[2]))
	x .= (fftpos(L, Ns[2], CenterFT))'
    
    if skip_final_phase
        phasefactor = nothing
    else    
        phasefactor = 1 / (1im * λ * z) .* exp.(1im * k / (2 * z) .* (x.^2 .+ y.^2)) 
    end

    buffer = zero.(U)

    FFTplan = plan_fft!(buffer, (1,2))

    return FraunhoferOp{typeof(L), typeof(buffer), typeof(phasefactor), typeof(FFTplan)}(buffer, phasefactor, L_new, FFTplan)
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
