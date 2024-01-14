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

    sxy = sin.(α)
    txy = tan.(α)

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
	(; k, f_x, f_y) = Zygote.@ignore _propagation_variables(field_new, λ, L_new)
	
	# transfer function kernel of angular spectrum
    H = exp.(1im .* k .* z .* (sqrt.(CT(1) .- abs2.(f_x .* λ .- sxy[2]) .- abs2.(f_y .* λ .- sxy[1])))
            .+ 1im .* 2 .* real(CT)(π) .* z .* (txy[2] .* f_x .+ txy[1] .* f_y))
	
	# bandlimit according to Matsushima
	# as addition we introduce a smooth bandlimit with a Hann window
	# and fuzzy logic 
	Δu =   1 ./ L_new
	u_limit = 1 ./ (sqrt.((2 .* Δu .* z).^2 .+ 1) .* λ)
	
    W = let
        if bandlimit
	        # bandlimit filter
            u_limit1 = u_limit isa AbstractArray ? u_limit[1:1, ..] : u_limit[1] 
            u_limit2 = u_limit isa AbstractArray ? u_limit[2:2, ..] : u_limit[2] 

            W = .*(hann.(scale.(abs2.(f_y) ./ u_limit1 .^2 .+ abs2.(f_x) * λ^2, 
                                bandlimit_border[1], bandlimit_border[2])),
                   hann.(scale.(abs2.(f_x) ./ u_limit2 .^2 .+ abs2.(f_y) * λ^2, 
                             bandlimit_border[1], bandlimit_border[2])))
        else
            # use an array here too, to avoid type instabilities
            W = similar(field, real(eltype(field)), size(f_y, 1), size(f_x, 1))
            W .= 1
        end
	end

    return (;field_new, H, W, fftdims)
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

    (; field_new, H, W, fftdims) = _prepare_shifted_angular_spectrum(field, z, λ, L, α; padding, 
                                              pad_factor, bandlimit, bandlimit_border)

	# propagate field
	field_out = fftshift(ifft(fft(ifftshift(field_new, fftdims), fftdims) .* H .* W, fftdims), fftdims)
	
    field_out_cropped = padding ? crop_center(field_out, size(field)) : field_out
	
	# return final field and some other variables
    return field_out_cropped, (; H, L, W)
end


scalable_angular_spectrum(field::AbstractArray, z, λ, L; kwargs...) = throw(ArgumentError("Provided field needs to have a complex elementype"))
