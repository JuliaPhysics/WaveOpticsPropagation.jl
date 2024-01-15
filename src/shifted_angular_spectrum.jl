export shifted_angular_spectrum 
export ShiftedAngularSpectrum


function _prepare_shifted_angular_spectrum(field::AbstractArray{CT}, z, λ, L, α;
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1.0)) where {CT<:Complex}

    fftdims = (1,2)
    T = real(CT)
    λ = T(λ)
    z = _transform_z(T, z)
    L = T.(L isa Number ? (L, L) : L)

    sxy = .+ sin.(α)
    txy = .+ tan.(α)

    bandlimit_border = real(CT).(bandlimit_border)
    L_new = padding ? pad_factor .* L : L

	# applies zero padding
    if ndims(field) == 2
        pad_factor2 = pad_factor
    elseif ndims(field) == 3
        pad_factor2 = (pad_factor * size(field, 1), pad_factor * size(field, 2), size(field, 3))
    end
	field_new = padding ? pad(field, pad_factor2) : field
	
	# helpful propagation variables
	(; k, f_x, f_y, x, y) = Zygote.@ignore _propagation_variables(field_new, λ, L_new)
	
	# transfer function kernel of angular spectrum
    #H = exp.(1im .* k .* z .* (sqrt.(CT(1) .- abs2.(f_x .* λ .- sxy[2]) .- abs2.(f_y .* λ .- sxy[1])))
    #        .+ 1im .* 2 .* real(CT)(π) .* z .* (txy[2] .* f_x .+ txy[1] .* f_y))
    
    H = exp.(1im .* k .* z .* (sqrt.(CT(1) .- abs2.(f_x .* λ .+ sxy[2]) .- abs2.(f_y .* λ .+ sxy[1]))
                              .+ λ .* (txy[2] .* f_x .+ txy[1] .* f_y)))
    #H = exp.(1im .* k .* z .* sqrt.(CT(1) .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)))
	
	# bandlimit according to Matsushima
	# as addition we introduce a smooth bandlimit with a Hann window
	# and fuzzy logic 
	
    W = let
        if bandlimit
	        # bandlimit filter
            χ = 1 / λ^2 .- abs2.(f_x .+ sxy[2] ./ λ) .- abs2.(f_y .+ sxy[1] ./ λ)

            Ωx = z * (txy[2] .- (f_x .+ sxy[2] ./ λ) ./ (sqrt.(χ)))
            Ωy = z * (txy[1] .- (f_y .+ sxy[1] ./ λ) ./ (sqrt.(χ)))

            Δf = abs.(f_x[1] - f_x[2])
            W = ( (1 / L_new[2]) .<= abs.(1 ./ 2 ./ Ωx)) .* ( (1 / L_new[1]) .<= abs.(1 ./ 2 ./ Ωy)) 
        else
            # use an array here too, to avoid type instabilities
            W = similar(field, real(eltype(field)), size(f_y, 1), size(f_x, 2))
            W .= 1
        end
	end
   
    shift = txy .* z

    ya = similar(field_new, real(eltype(field)), (size(field_new, 1), 1))
    ya .= (fftpos(L_new[1], size(field_new, 1), CenterFT)) .+ shift[1]
    xa = similar(field_new, real(eltype(field)), (1, size(field_new, 2)))
    xa .= (fftpos(L_new[2], size(field_new, 2), CenterFT))' .+ shift[2]
    
    ramp_before = ifftshift(exp.(1im .* 2 .* T(π) ./ λ .* (sxy[2] .* x .+ sxy[1] .* y)), (1,2))
    ramp_after = ifftshift(exp.(1im .* 2 .* T(π) ./ λ .* (sxy[2] .* xa .+ sxy[1] .* ya)), (1,2))
    return (;field_new, H, W, fftdims, ramp_before, ramp_after)
end


"""
    shifted_angular_spectrum(field, z, λ, L, α; kwargs...)

Returns the electrical field with physical length `L` and wavelength `λ` propagated with the shifted angular spectrum 
method of plane waves (AS) by the propagation distance `z`.
`α` is the shift angle with respect to the optical axis.

This method is efficient but to avoid recalculating some arrays (such as the phase kernel), see [`ShiftedAngularSpectrum`](@ref). 

# Arguments
* `field`: Input field
* `z`: propagation distance. Can be a single number or a vector of `z`s (Or `CuVector`). In this case the returning array has one dimension more.
* `λ`: wavelength of field
* `L`: field size (can be a scalar or a tuple) indicating field size
* `α` is the shift angle with respect to the optical axis.


# Keyword Arguments
* `padding=true`: applies padding to avoid convolution wraparound
* `pad_factor=2`: padding of 2. Larger numbers are not recommended since they don't provide better results.
* `bandlimit=true`: applies the bandlimit to avoid circular wraparound due to undersampling 
    of the complex propagation kernel [1]
* `bandlimit_border=(0.8, 1)`: applies a smooth bandlimit cut-off instead of hard-edge. 


# Examples
```jldoctest
julia> field = zeros(ComplexF32, (4,4)); field[3,3] = 1
```


# References
* Matsushima, Kyoji. "Shifted angular spectrum method for off-axis numerical propagation." Optics Express 18.17 (2010): 18453-18463.
"""
function shifted_angular_spectrum(field::AbstractArray{CT, 2}, z, λ, L, α; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1.0)) where {CT<:Complex}
   
    @assert size(field, 1) == size(field, 2) "input field needs to be quadradically shaped and not $(size(field, 1)), $(size(field, 2))"

    (; field_new, H, W, fftdims, ramp_before, ramp_after) = _prepare_shifted_angular_spectrum(field, z, λ, L, real(CT).(α); padding, 
                                              pad_factor, bandlimit, bandlimit_border)

	# propagate field
    field_new_is = ifftshift(field_new, fftdims) ./ (ramp_before)
#    ramp_after = 1
    field_out = fftshift(ramp_after .* ifft(fft(field_new_is, fftdims) .* H .* W, fftdims), fftdims)
    field_out_cropped = padding ? crop_center(field_out, size(field)) : field_out
	
	# return final field and some other variables
    return field_out_cropped, (; H, L, W, ramp_before, ramp_after)
end


scalable_angular_spectrum(field::AbstractArray, z, λ, L; kwargs...) = throw(ArgumentError("Provided field needs to have a complex elementype"))
