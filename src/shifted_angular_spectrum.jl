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
	
    
    H = exp.(1im .* k .* z .* (sqrt.(CT(1) .- abs2.(f_x .* λ .+ sxy[2]) .- abs2.(f_y .* λ .+ sxy[1]))
                              .+ λ .* (txy[2] .* f_x .+ txy[1] .* f_y)))
	
	# bandlimit according to Matsushima
    W = let
        if bandlimit
	        # bandlimit filter
            χ = max.(0, 1 / λ^2 .- abs2.(f_x .+ sxy[2] ./ λ) .- abs2.(f_y .+ sxy[1] ./ λ))

            Ωx = z * (txy[2] .- (f_x .+ sxy[2] ./ λ) ./ (sqrt.(χ)))
            Ωy = z * (txy[1] .- (f_y .+ sxy[1] ./ λ) ./ (sqrt.(χ)))

            W = ( (1 / L_new[2]) .<= abs.(1 ./ 2 ./ Ωx)) .* ( (1 / L_new[1]) .<= abs.(1 ./ 2 ./ Ωy)) 
        else
            # use an array here too, to avoid type instabilities
            W = similar(field, real(eltype(field)), size(f_y, 1), size(f_x, 2))
            W .= 1
        end
	end
   
    shift = txy .* z

    ya = similar(field_new, real(eltype(field)), (size(field_new, 1), 1))
    Zygote.@ignore ya .= (fftpos(L_new[1], size(field_new, 1), CenterFT)) .+ shift[1]
    xa = similar(field_new, real(eltype(field)), (1, size(field_new, 2)))
    Zygote.@ignore xa .= (fftpos(L_new[2], size(field_new, 2), CenterFT))' .+ shift[2]
    
    ramp_before = ifftshift(exp.(1im .* 2 .* T(π) ./ λ .* (sxy[2] .* x .+ sxy[1] .* y)), (1,2))
    ramp_after = ifftshift(exp.(1im .* 2 .* T(π) ./ λ .* (sxy[2] .* xa .+ sxy[1] .* ya)), (1,2))
    return (;field_new, H, W, fftdims, ramp_before, ramp_after)
end


function shifted_angular_spectrum(field::AbstractArray{CT, 2}, z, λ, L, α; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1.0)) where {CT<:Complex}
   
    @assert size(field, 1) == size(field, 2) "input field needs to be quadradically shaped and not $(size(field, 1)), $(size(field, 2))"

    (; field_new, H, W, fftdims, ramp_before, ramp_after) = _prepare_shifted_angular_spectrum(field, z, λ, L, real(CT).(α); padding, 
                                              pad_factor, bandlimit, bandlimit_border)

	# propagate field
    field_new_is = ifftshift(field_new, fftdims) ./ (ramp_before)
    field_out = fftshift(ramp_after .* ifft(fft(field_new_is, fftdims) .* H .* W, fftdims), fftdims)
    field_out_cropped = padding ? crop_center(field_out, size(field)) : field_out
    shift = z .* tan.(α) ./ L
	# return final field and some other variables
    return field_out_cropped, (; L, shift)
end


 # highly optimized version with pre-planning
struct ShiftedAngularSpectrum{A, T, T2, P, R}
    HW::A
    buffer::A
    buffer2::A
    L::T
    shift::T2
    p::P
    padding::Bool
    pad_factor::Int
    ramp_before::R
    ramp_after::R
end


"""
    ShiftedAngularSpectrum(field, z, λ, L, α; kwargs...)

Returns a method to propagate the electrical field with physical length `L` and wavelength `λ` with the shifted angular spectrum 
method of plane waves (AS) by the propagation distance `z`.
`α` should be a tuple containing the offset angles with respect to the optical axis.


# Arguments
* `field`: Input field
* `z`: propagation distance. Can be a single number or a vector of `z`s (Or `CuVector`). In this case the returning array has one dimension more.
* `λ`: wavelength of field
* `L`: field size (can be a scalar or a tuple) indicating field size
* `α` is the tuple of shift angles with respect to the optical axis.


# Keyword Arguments
* `pad_factor=2`: padding of 2. Larger numbers are not recommended since they don't provide better results.
* `bandlimit=true`: applies the bandlimit to avoid circular wraparound due to undersampling 
    of the complex propagation kernel [1]
* `extract_ramp=true`: divides the field by phase ramp `exp.(1im * 2π / λ * (sin(α[2]) .* x .+ sin(α[1]) .* y))` and multiplies after
                        propagation the ramp (with new real space coordinates) back to the field

# Examples
```jldoctest
julia> field = zeros(ComplexF32, (4,4)); field[3,3] = 1

julia> AS, _ = ShiftedAngularSpectrum(field, 100e-6, 633e-9, 100e-6, (deg2rad(10), 0));

julia> AS(field)
(ComplexF32[1.5269792f-5 + 1.7594219f-5im -3.996831f-5 - 7.624799f-5im -0.0047351345f0 + 0.002100923f0im -3.996831f-5 - 7.624799f-5im; -8.294997f-5 - 1.8230454f-5im 0.00028230582f0 + 8.1745195f-5im 0.0051693693f0 - 0.016958509f0im 0.00028230582f0 + 8.1745195f-5im; 0.0029884572f0 + 0.0040671355f0im -0.009686601f0 - 0.014245203f0im -0.82990384f0 + 0.5566719f0im -0.009686601f0 - 0.014245203f0im; -1.4191573f-7 + 9.41665f-5im 2.111472f-5 - 0.00031620878f0im -0.017670793f0 - 0.0014212304f0im 2.111472f-5 - 0.00031620878f0im], (L = 0.0001, shift = (0.17632698070846498, 0.0)))
```

# References
* Matsushima, Kyoji. "Shifted angular spectrum method for off-axis numerical propagation." Optics Express 18.17 (2010): 18453-18463.
"""
function ShiftedAngularSpectrum(field::AbstractArray{CT, N}, z::Number, λ, L, α; 
                          extract_ramp=true,
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1)) where {CT, N}
    
        (; field_new, H, W, fftdims, ramp_before, ramp_after) = 
            _prepare_shifted_angular_spectrum(field, z, λ, L, real(CT).(α); padding, 
                                              pad_factor, bandlimit, bandlimit_border)
    
      
        buffer2 = similar(field, complex(eltype(field_new)), (size(field_new, 1), size(field_new, 2)))
        buffer = copy(buffer2)
        
        p = plan_fft!(buffer, (1, 2))
        H .= H .* W
        HW = H
        shift = z .* tan.(α) ./ L
        if !extract_ramp 
            ramp_before = nothing
            ramp_after = nothing
        end


        return ShiftedAngularSpectrum{typeof(H), typeof(L), typeof(shift), typeof(p), typeof(ramp_before)}(HW, 
                    buffer, buffer2, L, shift, p, padding, pad_factor, ramp_before, ramp_after), (;L, shift)
    end



"""
    (shifted_as::ShiftedAngularSpectrum)(field)

Uses the struct to efficiently store some pre-calculated objects.
Propagate the field.
"""
function (as::ShiftedAngularSpectrum{A, T, T2, P, R})(field) where {A,T,T2,P,R}
    fill!(as.buffer2, 0)
    field_new = set_center!(as.buffer2, field, broadcast=true)
    ifftshift!(as.buffer, field_new, (1, 2))
    if !(R === Nothing) 
        as.buffer ./= as.ramp_before
    end
    field_imd = as.p * as.buffer
    field_imd .*= as.HW
    field_imd = inv(as.p) * field_imd
    if !(R === Nothing) 
        field_imd .*= as.ramp_after
    end
    field_out = fftshift!(as.buffer2, field_imd, (1, 2))
    field_out_cropped = as.padding ? crop_center(field_out, size(field), return_view=true) : field_out
    return field_out_cropped, (; as.L, as.shift)
end


function ChainRulesCore.rrule(as::ShiftedAngularSpectrum{A, T, T2, P, R}, field) where {A,T,T2,P,R}
    field_and_tuple = as(field) 
    function as_pullback(ȳ)
        f̄ = NoTangent()
        y2 = ȳ.backing[1] 
    
        fill!(as.buffer2, 0)
        field_new = as.padding ? set_center!(as.buffer2, y2, broadcast=true) : y2 
        ifftshift!(as.buffer, field_new, (1, 2))
        if !(R === Nothing) 
            as.buffer .*= conj.(as.ramp_after)
        end
        field_imd = as.p * as.buffer 
        field_imd .*= conj.(as.HW)
        
        field_imd = inv(as.p) * field_imd 
        if !(R === Nothing) 
            field_imd ./= conj.(as.ramp_before)
        end
        field_out = fftshift!(as.buffer2, field_imd, (1, 2))
        field_out_cropped = as.padding ? crop_center(field_out, size(field), return_view=true) : field_out
        return f̄, field_out_cropped 
    end
    return field_and_tuple, as_pullback
end
