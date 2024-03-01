export AngularSpectrum


_transform_z(::Type{T}, z::Number) where {T<:Number} = T(z)
_transform_z(::Type{T}, z::AbstractArray{T}) where T = reshape(z, 1, 1, :)
_transform_z(::Type{T}, z::AbstractArray{T2}) where {T, T2} = reshape(T.(z), 1, 1, :)

function _prepare_angular_spectrum(field::AbstractArray{CT}, z, λ, _L;
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1.0)) where {CT<:Complex}

    fftdims = (1,2)
    T = real(CT)
    λ = T(λ)
    z = _transform_z(T, z)
    L = T.(_L isa Number ? (_L, _L) : _L)

    bandlimit_border = real(CT).(bandlimit_border)
    Lp = padding ? pad_factor .* L : L

	# applies zero padding
    if ndims(field) == 2
        pad_factor2 = pad_factor
    elseif ndims(field) == 3
        pad_factor2 = (pad_factor * size(field, 1), pad_factor * size(field, 2), size(field, 3))
    end
	fieldp = padding ? pad(field, pad_factor2) : field
	
	# helpful propagation variables
	(; k, f_x, f_y, x, y) = Zygote.@ignore _propagation_variables(fieldp, λ, Lp)
	
	# transfer function kernel of angular spectrum
    H = exp.(1im .* k .* abs.(z) .* sqrt.(CT(1) .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)))

    # take complex conjugate, for negative zs
    H = real.(H) .+ sign.(z) .* 1im .* imag(H) 

	# bandlimit according to Matsushima
	# as addition we introduce a smooth bandlimit with a Hann window
	# and fuzzy logic 
	Δu =   1 ./ Lp
	u_limit = Zygote.@ignore 1 ./ (sqrt.((2 .* Δu .* z).^2 .+ 1) .* λ)

    # y and x positions in real space, use correct spacing -> fftpos
	y1 = similar(field, real(eltype(field)), (size(field, 1), 1))
	Zygote.@ignore y1 .= (fftpos(L[1], size(field, 1), CenterFT))
	x1 = similar(field, real(eltype(field)), (1, size(field, 2)))
	Zygote.@ignore x1 .= (fftpos(L[2], size(field, 2), CenterFT))'

    params = Params(y1, x1, y1, x1, L, L)

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

    return (;fieldp, H, W, fftdims, params)
end


"""
    angular_spectrum(field, z, λ, L; kwargs...)


# Examples
```jldoctest
julia> field = zeros(ComplexF32, (4,4)); field[3,3] = 1
1
julia> as, t = angular_spectrum(field, 100e-9, 632e-9, 10e-6)
(ComplexF32[6.519258f-8 - 3.5390258f-7im -2.5097302f-7 + 1.1966712f-6im 0.00041754358f0 - 0.00027736276f0im -2.5097302f-7 + 1.1966712f-6im; -2.5136978f-7 + 1.1966767f-6im 8.540863f-7 - 4.0512546f-6im -0.001423778f0 + 0.00093840604f0im 8.540863f-7 - 4.0512546f-6im; 0.00041754544f0 - 0.0002773702f0im -0.0014237723f0 + 0.0009384034f0im 0.5497784f0 + 0.8353027f0im -0.0014237723f0 + 0.0009384034f0im; -2.5136978f-7 + 1.1966767f-6im 8.540863f-7 - 4.0512546f-6im -0.001423778f0 + 0.00093840604f0im 8.540863f-7 - 4.0512546f-6im], (H = ComplexF32[0.54519475f0 + 0.8383094f0im 0.5456109f0 + 0.8380386f0im … 0.54685986f0 + 0.8372242f0im 0.5456109f0 + 0.8380386f0im; 0.5456109f0 + 0.8380386f0im 0.5460271f0 + 0.83776754f0im … 0.54727626f0 + 0.83695203f0im 0.5460271f0 + 0.83776754f0im; … ; 0.54685986f0 + 0.8372242f0im 0.54727626f0 + 0.83695203f0im … 0.548526f0 + 0.8361335f0im 0.54727626f0 + 0.83695203f0im; 0.5456109f0 + 0.8380386f0im 0.5460271f0 + 0.83776754f0im … 0.54727626f0 + 0.83695203f0im 0.5460271f0 + 0.83776754f0im], L = 1.0e-5))
```


# References
* Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673.
"""
function angular_spectrum(field::AbstractArray{CT, 2}, z, λ, L; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1.0)) where {CT<:Complex}
   
    @assert size(field, 1) == size(field, 2) "input field needs to be quadradically shaped and not $(size(field, 1)), $(size(field, 2))"

    (; fieldp, H, W, fftdims) = _prepare_angular_spectrum(field, z, λ, L; padding, 
                                              pad_factor, bandlimit, bandlimit_border)

	# propagate field
	field_out = fftshift(ifft(fft(ifftshift(fieldp, fftdims), fftdims) .* H .* W, fftdims), fftdims)
	
    field_out_cropped = padding ? crop_center(field_out, size(field)) : field_out
	
	# return final field and some other variables
    return field_out_cropped
end


angular_spectrum(field::AbstractArray, z, λ, L; kwargs...) = throw(ArgumentError("Provided field needs to have a complex elementype"))


 # highly optimized version with pre-planning
struct AngularSpectrum3{A, P, M, M2}
    HW::A
    buffer::A
    buffer2::A
    params::Params{M, M2}
    p::P
    padding::Bool
    pad_factor::Int
end

"""
    AngularSpectrum(field, z, λ, L; kwargs...)

Returns a function for efficient reuse of pre-calculated kernels.
This function returns the electrical field with physical length `L` and wavelength `λ` propagated with the angular spectrum 
method of plane waves (AS) by the propagation distance `z`.

This method is efficient but to avoid recalculating some arrays (such as the phase kernel), see [`AngularSpectrum`](@ref). 

# Arguments
* `field`: Input field
* `z`: propagation distance. Can be a single number or a vector of `z`s (Or `CuVector`). In this case the returning array has one dimension more.
* `λ`: wavelength of field
* `L`: field size (can be a scalar or a tuple) indicating field size


# Keyword Arguments
* `padding=true`: applies padding to avoid convolution wraparound
* `bandlimit=true`: applies the bandlimit to avoid circular wraparound due to undersampling 
    of the complex propagation kernel [1]
* `bandlimit_border=(0.8, 1)`: applies a smooth bandlimit cut-off instead of hard-edge. 



See [`angular_spectrum`](@ref) for the full documentation.

# Example
```jldoctest
julia> field = zeros(ComplexF32, (4,4)); field[3,3] = 1
1

julia> as = AngularSpectrum(field, 100e-9, 632e-9, 10e-6)
WaveOpticsPropagation.AngularSpectrum3{Matrix{ComplexF32}, Float64, FFTW.cFFTWPlan{ComplexF32, -1, true, 2, Tuple{Int64, Int64}}}(ComplexF32[0.54519475f0 + 0.8383094f0im 0.5456109f0 + 0.8380386f0im … 0.54685986f0 + 0.8372242f0im 0.5456109f0 + 0.8380386f0im; 0.5456109f0 + 0.8380386f0im 0.5460271f0 + 0.83776754f0im … 0.54727626f0 + 0.83695203f0im 0.5460271f0 + 0.83776754f0im; … ; 0.54685986f0 + 0.8372242f0im 0.54727626f0 + 0.83695203f0im … 0.548526f0 + 0.8361335f0im 0.54727626f0 + 0.83695203f0im; 0.5456109f0 + 0.8380386f0im 0.5460271f0 + 0.83776754f0im … 0.54727626f0 + 0.83695203f0im 0.5460271f0 + 0.83776754f0im], ComplexF32[3.0f-45 + 6.0f-45im 6.3f-44 + 7.0f-44im … 3.46f-43 + 2.42f-43im 7.1f-44 + 7.1f-44im; 8.0f-45 + 1.1f-44im 7.3f-44 + 7.6f-44im … 3.48f-43 + 2.4f-43im 3.69f-43 + 6.9f-44im; … ; 4.9f-44 + 5.9f-44im 1.11f-43 + 1.15f-43im … 7.7f-44 + 7.7f-44im -327424.25f0 + 4.579f-41im; 1.3f-44 + 6.2f-44im 1.16f-43 + 1.23f-43im … 3.66f-43 + 7.4f-44im -328013.5f0 + 4.579f-41im], ComplexF32[3.0f-45 + 6.0f-45im 6.3f-44 + 7.0f-44im … 3.46f-43 + 2.42f-43im 7.1f-44 + 7.1f-44im; 8.0f-45 + 1.1f-44im 7.3f-44 + 7.6f-44im … 3.48f-43 + 2.4f-43im 3.69f-43 + 6.9f-44im; … ; 4.9f-44 + 5.9f-44im 1.11f-43 + 1.15f-43im … 7.7f-44 + 7.7f-44im -327424.25f0 + 4.579f-41im; 1.3f-44 + 6.2f-44im 1.16f-43 + 1.23f-43im … 3.66f-43 + 7.4f-44im -328013.5f0 + 4.579f-41im], 1.0e-5, FFTW in-place forward plan for 8×8 array of ComplexF32
(dft-rank>=2/1
  (dft-direct-8-x8 "n1fv_8_avx2_128")
  (dft-direct-8-x8 "n1fv_8_avx2_128")), true, 2)

julia> as(field)
4×4 Matrix{ComplexF32}:
  6.51926f-8-3.53903f-7im  -2.50973f-7+1.19667f-6im   0.000417544-0.000277363im  -2.50973f-7+1.19667f-6im
  -2.5137f-7+1.19668f-6im   8.54086f-7-4.05125f-6im   -0.00142378+0.000938406im   8.54086f-7-4.05125f-6im
 0.000417545-0.00027737im  -0.00142377+0.000938403im     0.549778+0.835303im     -0.00142377+0.000938403im
  -2.5137f-7+1.19668f-6im   8.54086f-7-4.05125f-6im   -0.00142378+0.000938406im   8.54086f-7-4.05125f-6im
```
"""
function AngularSpectrum(field::AbstractArray{CT, N}, z, λ, L; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1)) where {CT, N}
   
        (; fieldp, H, W, fftdims, params) = _prepare_angular_spectrum(field, z, λ, L; padding, 
                                              pad_factor, bandlimit, bandlimit_border)
    
        if z isa AbstractVector
            buffer2 = similar(field, complex(eltype(fieldp)), (size(fieldp, 1), size(fieldp, 2), length(z)))
            buffer = copy(buffer2)
        else
            buffer2 = similar(field, complex(eltype(fieldp)), (size(fieldp, 1), size(fieldp, 2)))
            buffer = copy(buffer2)
        end
        
        p = plan_fft!(buffer, (1, 2))
        H .= H .* W
        HW = H
  
        return AngularSpectrum3(HW, buffer, buffer2, params, p, padding, pad_factor)
    end

"""
    (as:AngularSpectrum3)(field)

Uses the struct to efficiently store some pre-calculated objects.
Propagate the field.
"""
function (as::AngularSpectrum3)(field)
    fill!(as.buffer2, 0)
    fieldp = set_center!(as.buffer2, field, broadcast=true)
    field_imd = as.p * ifftshift!(as.buffer, fieldp, (1, 2))
    field_imd .*= as.HW
    field_out = fftshift!(as.buffer2, inv(as.p) * field_imd, (1, 2))
    field_out_cropped = as.padding ? crop_center(field_out, size(field), return_view=true) : field_out
    return field_out_cropped
end


function ChainRulesCore.rrule(as::AngularSpectrum3, field)
    field_and_tuple = as(field) 
    function as_pullback(ȳ)
        f̄ = NoTangent()
        # i tried to fix this once, but we somehow the Tangent type is missing the dimensionality
        # which we need for set_center! and crop_center
        y2 = ȳ
    
        fill!(as.buffer2, 0)
        fieldp = as.padding ? set_center!(as.buffer2, y2, broadcast=true) : y2 
        field_imd = as.p * ifftshift!(as.buffer, fieldp, (1, 2))
        field_imd .*= conj.(as.HW)
        field_out = fftshift!(as.buffer2, inv(as.p) * field_imd, (1, 2))
        # that means z is a vector and we do plane to volume propagation
        if size(as.buffer, 3) > 1 
            sum!(view(as.buffer, :, :, 1), field_out)
            field_out_cropped = as.padding ? crop_center(view(as.buffer, :, :, 1), size(field), return_view=true) : view(as.buffer, :, :, 1) 
        else
            field_out_cropped = as.padding ? crop_center(field_out, size(field), return_view=true) : field_out
        end
        return f̄, field_out_cropped 
    end
    return field_and_tuple, as_pullback
end
