export angular_spectrum
export AngularSpectrum


_transform_z(::Type{T}, z::Number) where {T<:Number} = T(z)
_transform_z(::Type{T}, z::AbstractArray{T}) where T = reshape(z, 1, 1, :)
_transform_z(::Type{T}, z::AbstractArray{T2}) where {T, T2} = reshape(T.(z), 1, 1, :)

function _prepare_angular_spectrum(field::AbstractArray{CT}, z, λ, L; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1.0)) where {CT<:Complex}

    fftdims = (1,2)
    T = real(CT)
    λ = T(λ)
    z = _transform_z(T, z)
    L = T.(L isa Number ? (L, L) : L)

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
    H = exp.(1im .* k .* z .* sqrt.(CT(1) .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)))
	
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
    angular_spectrum(field, z, λ, L; kwargs...)

Returns the electrical field with physical length `L` and wavelength `λ` propagated with the angular spectrum 
method of plane waves (AS) by the propagation distance `z`.

This method is efficient but to avoid recalculating some arrays (such as the phase kernel), see [`AngularSpectrum`](@ref). 

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
function angular_spectrum(field::AbstractArray{CT, 2}, z, λ, L; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1.0)) where {CT<:Complex}
   
    @assert size(field, 1) == size(field, 2) "input field needs to be quadradically shaped and not $(size(field, 1)), $(size(field, 2))"

    (; field_new, H, W, fftdims) = _prepare_angular_spectrum(field, z, λ, L; padding, 
                                              pad_factor, bandlimit, bandlimit_border)

	# propagate field
	field_out = fftshift(ifft(fft(ifftshift(field_new, fftdims), fftdims) .* H .* W, fftdims), fftdims)
	
    field_out_cropped = padding ? crop_center(field_out, size(field)) : field_out
	
	# return final field and some other variables
    return field_out_cropped, (; H, L)
end


angular_spectrum(field::AbstractArray, z, λ, L; kwargs...) = throw(ArgumentError("Provided field needs to have a complex elementype"))


 # highly optimized version with pre-planning
struct AngularSpectrum3{A, T, P}
    HW::A
    buffer::A
    buffer2::A
    L::T
    p::P
    padding::Bool
    pad_factor::Int
end

"""
    AngularSpectrum(field, z, λ, L; kwargs...)

Returns a function for efficient reuse of pre-calculated kernels.

See [`angular_spectrum`](@ref) for the full documentation.


# Example
```jldoctest
julia> field = zeros(ComplexF32, (4,4)); field[3,3] = 1
1

julia> as = AngularSpectrum(field, 100e-9, 632e-9, 10e-6)
WaveOpticsPropagation.AngularSpectrum3{Matrix{ComplexF64}, Float64, FFTW.cFFTWPlan{ComplexF32, -1, true, 2, UnitRange{Int64}}}(ComplexF64[0.5451947489704718 + 0.8383094212133275im 0.5456108987195186 + 0.8380386310895693im … 0.5468597886165149 + 0.8372242062878382im 0.5456108987195186 + 0.8380386310895693im; 0.5456108987195186 + 0.8380386310895693im 0.5460271219052333 + 0.8377674988586556im … 0.5472762321570075 + 0.8369520450515843im 0.5460271219052333 + 0.8377674988586556im; … ; 0.5468597886165149 + 0.8372242062878382im 0.5472762321570075 + 0.8369520450515843im … 0.5485260036074586 + 0.8361334961394803im 0.5472762321570075 + 0.8369520450515843im; 0.5456108987195186 + 0.8380386310895693im 0.5460271219052333 + 0.8377674988586556im … 0.5472762321570075 + 0.8369520450515843im 0.5460271219052333 + 0.8377674988586556im], ComplexF64[0.0 + 0.0im 0.0 + 0.0im … 2047.5009765625 + 4.571175720473986e-41im 2047.5009765625 + 4.571175720473986e-41im; 2058.2890625 + 4.571175720473986e-41im 0.0 + 0.0im … 2056.2578125 + 4.571175720473986e-41im 2058.1640625 + 4.571175720473986e-41im; … ; 0.0 + 0.0im 0.0 + 0.0im … 2047.5009765625 + 4.571175720473986e-41im 2047.5009765625 + 4.571175720473986e-41im; 0.0 + 0.0im 0.0 + 0.0im … 2058.0234375 + 4.571175720473986e-41im 0.0 + 0.0im], ComplexF64[0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; … ; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im], 1.0e-5, FFTW in-place forward plan for 8×8 array of ComplexF32
(dft-rank>=2/1
  (dft-direct-8-x8 "n1fv_8_avx2_128")
  (dft-direct-8-x8 "n1fv_8_avx2_128")), true, 2)

julia> as(field)
4×4 Matrix{ComplexF64}:
  7.07805e-8-3.53903e-7im  -2.54379e-7+1.20194e-6im   0.000417542-0.000277363im  -2.54379e-7+1.20194e-6im
 -2.52505e-7+1.20063e-6im   8.57634e-7-4.05761e-6im   -0.00142377+0.000938398im   8.57634e-7-4.05761e-6im
 0.000417545-0.00027737im  -0.00142378+0.000938403im     0.549778+0.835303im     -0.00142378+0.000938403im
 -2.52505e-7+1.20063e-6im   8.57634e-7-4.05761e-6im   -0.00142377+0.000938398im   8.57634e-7-4.05761e-6im
```
"""
function AngularSpectrum(field::AbstractArray{CT, N}, z, λ, L; 
                          padding=true, pad_factor=2,
                          bandlimit=true,
                          bandlimit_border=(0.8, 1)) where {CT, N}
   
        (; field_new, H, W, fftdims) = _prepare_angular_spectrum(field, z, λ, L; padding, 
                                              pad_factor, bandlimit, bandlimit_border)
    
      
        if z isa AbstractVector
            buffer2 = similar(field, complex(eltype(field_new)), (size(field_new, 1), size(field_new, 2), length(z)))
            buffer = copy(buffer2)
        else
            buffer2 = similar(field, complex(eltype(field_new)), (size(field_new, 1), size(field_new, 2)))
            buffer = copy(buffer2)
        end
        
        p = plan_fft!(buffer, (1, 2))
        H .= H .* W
        HW = H
  
        return AngularSpectrum3{typeof(H), typeof(L), typeof(p)}(HW, buffer, buffer2, L, p, padding, pad_factor), L
    end

"""
    (as:AngularSpectrum3)(field)

Uses the struct to efficiently store some pre-calculated objects.
Propagate the field.
"""
function (as::AngularSpectrum3)(field)
    fill!(as.buffer2, 0)
    field_new = set_center!(as.buffer2, field, broadcast=true)
    field_imd = as.p * ifftshift!(as.buffer, field_new, (1, 2))
    field_imd .*= as.HW
    field_out = fftshift!(as.buffer2, inv(as.p) * field_imd, (1, 2))
    field_out_cropped = as.padding ? crop_center(field_out, size(field), return_view=true) : field_out
    return field_out_cropped, (; as.L)
end


function ChainRulesCore.rrule(as::AngularSpectrum3, field)
    field_and_tuple = as(field) 
    function as_pullback(ȳ)
        f̄ = NoTangent()
        y2 = ȳ.backing[1] 
    
        fill!(as.buffer2, 0)
        field_new = as.padding ? set_center!(as.buffer2, y2, broadcast=true) : y2 
        field_imd = as.p * ifftshift!(as.buffer, field_new, (1, 2))
        field_imd .*= conj.(as.HW)
        field_out = fftshift!(as.buffer2, inv(as.p) * field_imd, (1, 2))
        field_out_cropped = as.padding ? crop_center(field_out, size(field), return_view=true) : field_out
        return f̄, field_out_cropped 
    end
    return field_and_tuple, as_pullback
end
