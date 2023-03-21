export angular_spectrum
export Angular_Spectrum

"""
	angular_spectrum(field, z, λ, L)

Returns the electrical field with physical length `L` and wavelength `λ` propagated with the angular spectrum 
method of plane waves (AS) by the propagation distance `z`.


# Arguments
* `field`: Input field
* `z`: propagation distance
* `λ`: wavelength of field
* `L`: field size (can be a scalar or a tuple) indicating field size


# Keyword Arguments
* `padding=true`: applies padding to avoid convolution wraparound
* `pad_factor=2`: padding of 2. Larger numbers are not recommended since they don't provide better results.
* `bandlimit=true`: applies the bandlimit to avoid circular wraparound due to undersampling 
    of the complex propagation kernel [1]
* `bandlimit_border=(0.8, 1)`: applies a smooth bandlimit cut-off instead of hard-edge. 


# References
* Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673.
"""
function angular_spectrum(field::Matrix{T}, z, λ, L; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1)) where T
   
    bandlimit_border = real(T).(bandlimit_border)
    L_new = padding ? pad_factor .* L : L

	# applies zero padding
	field_new = padding ? pad(field, pad_factor) : field
	
	# helpful propagation variables
	(; k, f_x, f_y) = _propagation_variables(field_new, λ, L_new)
	
	# transfer function kernel of angular spectrum
	H = exp.(1im .* k .* z .* sqrt.(T(1) .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)))
	
	# bandlimit according to Matsushima
	# as addition we introduce a smooth bandlimit with a Hann window
	# and fuzzy logic 
	Δu =   1 / L_new
	u_limit = 1 / (sqrt((2 * Δu * z)^2 + 1) * λ)
	f_x_limit = sqrt(inv(1/u_limit^2 + λ^2))
	
    W = let
        if bandlimit
	        # bandlimit filter
	        # smoothing at 0.8 is arbitrary but works well
	        W = .*(smooth_f.(abs2.(f_y) ./ u_limit^2 .+ abs2.(f_x) * λ^2, 
                             bandlimit_border[1], bandlimit_border[2]),
	        	   smooth_f.(abs2.(f_x) ./ u_limit^2 .+ abs2.(f_y) * λ^2, 
                             bandlimit_border[1], bandlimit_border[2]))
        else
            T(1)
        end
	end

	# propagate field
	field_out = fftshift(ifft(fft(ifftshift(field_new)) .* H .* W))
	
    field_out_cropped = padding ? crop_center(field_out, size(field)) : field_out
	
	# return final field and some other variables
	return field_out_cropped; (; L)
end





struct Angular_Spectrum2{A, T, P}
    HW::A
    L::T
    p::P
    padding::Bool
    pad_factor::Int
end

function Angular_Spectrum(field::Matrix{T}, z, λ, L; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1)) where T
   
        bandlimit_border = real(T).(bandlimit_border)
        L_new = padding ? pad_factor .* L : L
	    field_new = padding ? pad(field, pad_factor) : field
	    
	    # helpful propagation variables
	    (; k, f_x, f_y) = _propagation_variables(field_new, λ, L_new)
	    
	    # transfer function kernel of angular spectrum
	    H = exp.(1im .* k .* z .* sqrt.(T(1) .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)))
	    
	    # bandlimit according to Matsushima
	    # as addition we introduce a smooth bandlimit with a Hann window
	    # and fuzzy logic 
	    Δu =   1 / L_new
	    u_limit = 1 / (sqrt((2 * Δu * z)^2 + 1) * λ)
	    f_x_limit = sqrt(inv(1/u_limit^2 + λ^2))
	    
        W = let
            if bandlimit
	            # bandlimit filter
	            # smoothing at 0.8 is arbitrary but works well
	            W = .*(smooth_f.(abs2.(f_y) ./ u_limit^2 .+ abs2.(f_x) * λ^2, 
                                 bandlimit_border[1], bandlimit_border[2]),
	            	   smooth_f.(abs2.(f_x) ./ u_limit^2 .+ abs2.(f_y) * λ^2, 
                                 bandlimit_border[1], bandlimit_border[2]))
            else
                T(1)
            end
	    end
    
        p = plan_fft!(field_new)
        HW = H .* W
    
        return Angular_Spectrum2{typeof(HW), typeof(L), typeof(p)}(HW, L, p, padding, pad_factor)
    end


function (as::Angular_Spectrum2)(field; do_first_padding=true)
    field_new = as.padding && do_first_padding ? pad(field, as.pad_factor) : field
	field_out = fftshift(inv(as.p) * ((as.p * ifftshift(field_new)) .* as.HW))
    field_out_cropped = as.padding ? crop_center(field_out, size(field)) : field_out
end



const Angular_Spectrum = Angular_Spectrum2


