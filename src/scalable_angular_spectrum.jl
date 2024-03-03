export ScalableAngularSpectrum

 # highly optimized version with pre-planning
struct ScalableAngularSpectrumOp{AP, AP2, AB, T, P}
    ΔH::AP
    H₁::AP
    H₂::AP2 # could be nothing!
    buffer::AB
    buffer2::AB
    params::T
    FFTplan::P
end

"""
    _SAS_propagation_variables(field, z, λ, L)

Internal method to create variables we need for propagation such as frequencies in Fourier space, etc..
"""
function _SAS_propagation_variables(field::AbstractArray{T, M}, z, λ, L) where {T, M} 
	@assert size(field, 1) == size(field, 2) "Quadratic fields only working currently"
	
	# wave number
	k = T(2π) / λ
	# number of samples
	N = size(field, 1)
	# sample spacing
	dx = L / N 
	# frequency spacing
	df = 1 / L 
	# total size in frequency space
	Lf = N * df
	
	# frequencies centered around first entry 
	# 1D vectors each
	f_y = similar(field, real(eltype(field)), (N,))
	f_y .= fftfreq(N, Lf)
	f_x = f_y'
	
	# y and x positions in real space
	#y = ifftshift(range(-L/2, L/2, length=N))
	y = similar(field, real(eltype(field)), (N, 1))
	x = similar(field, real(eltype(field)), (1, N))
	y .= fftpos(L, N, CenterFT)
    y = ifftshift(y)
	x .= y'
	
	return (; k, dx, df, f_x, f_y, x, y)
end


# helper function for smooth window
function find_width_window(ineq_1D::AbstractVector, bandlimit_border)
    ineq_1D = Array(ineq_1D)
	bs = ineq_1D .≤ 1
	ind_x_first = findfirst(bs)
	ind_x_last = findlast(bs)

	if isnothing(ind_x_first) || isnothing(ind_x_last)
		return (1,1)
	end

	diff_b = round(Int, 0.5 * (1 - bandlimit_border[1]) * (Tuple(ind_x_last)[1] - Tuple(ind_x_first)[1]))
	diff_e = round(Int, 0.5 * (1 - bandlimit_border[2]) * (Tuple(ind_x_last)[1] -
	Tuple(ind_x_first)[1]))

	ineq_v_b = ineq_1D[ind_x_first + diff_b]
	ineq_v_e = ineq_1D[ind_x_first + diff_e]

	return ineq_v_b, ineq_v_e
end

"""
    ScalableAngularSpectrum(field, z, λ, L; kwargs...)

Returns the electrical field with physical length `L` and wavelength `λ` propagated with the scalable angular spectrum method of plane waves (SAS) by the propagation distance `z`.


# Arguments
* `field`: Input field
* `z`: propagation distance
* `λ`: wavelength of field
* `L`: field size (can be a scalar or a tuple) indicating field size


# Keyword Arguments 
* `skip_final_phase=true`: avoid multiplying with final phase. This phase is also undersampled.
* `bandlimit_border=(0.8, 1)`: apply soft bandlimit instead of hard cutoff


# Example
```jldoctest
julia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;

julia> sas, t = ScalableAngularSpectrum(field, 10f-3, 633f-9, 500f-6);

julia> res, t2 = sas(field);

julia> t2
(L = 0.00162048f0,)

julia> t
(L = 0.00162048f0,)

julia> t2.L / 500f-6 # calculate magnification
3.24096f0

julia> 10f-3 * 633f-9 * 256 / 500f-6^2 / 2 # equation from paper
3.2409596f0
```

# References
* [Rainer Heintzmann, Lars Loetgering, and Felix Wechsler, "Scalable angular spectrum propagation," Optica 10, 1407-1416 (2023)](https://opg.optica.org/optica/viewmedia.cfm?uri=optica-10-11-1407&html=true) 
"""
function ScalableAngularSpectrum(ψ₀::AbstractArray{T}, z, λ, L ; 
								 skip_final_phase=true, bandlimit_soft_px=20,
								  bandlimit_border=(0.8, 1)) where {T} 
	@assert bandlimit_soft_px ≥ 0 "bandlimit_soft_px must be ≥ 0"
	@assert size(ψ₀, 1) == size(ψ₀, 2) "Restricted to auadratic fields."
    
    pad_factor = 2
    set_pad_zero = false
	
	N = size(ψ₀, 1)
	z_limit = (- 4 * L * sqrt(8*L^2 / N^2 + λ^2) * sqrt(L^2 * inv(8 * L^2 + N^2 * λ^2)) / (λ * (-1+2 * sqrt(2) * sqrt(L^2 * inv(8 * L^2 + N^2 * λ^2)))))
	
	# vignetting limit
	z > z_limit &&  @warn "Propagated field might be affected by vignetting"
	L_new = pad_factor * L
	
	# applies zero padding
	ψ_p = select_region(ψ₀, new_size=size(ψ₀) .* pad_factor)
	k, dx, df, f_x, f_y, x, y = _SAS_propagation_variables(ψ_p, z, λ, L_new)  
	M = λ * z * N / L^2 / 2
	
	# calculate anti_aliasing_filter for precompensation
	cx = λ .* f_x 
	cy = λ .* f_y 
	tx = L_new / 2 / z .+ abs.(λ .* f_x)
	ty = L_new / 2 / z .+ abs.(λ .* f_y)
	
	# smooth window function
	smooth_f(x, α, β) = hann(scale(x, α, β))
	# find boundary for soft hann
	ineq_x = fftshift(cx[1, :].^2 .* (1 .+ tx[1, :].^2) ./ tx[1, :].^2 .+ cy[1, :].^2)
	limits = find_width_window(ineq_x, bandlimit_border)
	
	# bandlimit filter for precompensation
	W = .*(smooth_f.(cx.^2 .* (1 .+ tx.^2) ./ tx.^2 .+ cy.^2, limits...),
			smooth_f.(cy.^2 .* (1 .+ ty.^2) ./ ty.^2 .+ cx.^2, limits...))
	
	# ΔH is the core part of Fresnel and AS
	H_AS = sqrt.(0im .+ 1 .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)) 
	H_Fr = 1 .- abs2.(f_x .* λ) / 2 .- abs2.(f_y .* λ) / 2 
	# take the difference here, key part of the ScaledAS
	ΔH = W .* exp.(1im .* k .* z .* (H_AS .- H_Fr)) 
	
	
	# new sample coordinates
	dq = λ * z / L_new
	Q = dq * N * pad_factor
    q_y = similar(ψ_p, (pad_factor * N, 1))
    q_x = similar(ψ_p, (1, pad_factor * N))
	q_y .= fftpos(dq * pad_factor * N, pad_factor * N, CenterFT)
    q_y = ifftshift(q_y)
	q_x .= q_y'
	
    	
	# calculate phases of Fresnel
	H₁ = exp.(1im .* k ./ (2 .* z) .* (x .^ 2 .+ y .^ 2))
	
	# skip final phase because often undersampled
	if skip_final_phase
	    H₂ = nothing	
	else
		H₂ = (exp.(1im .* k .* z) .*
			exp.(1im .* k ./ (2 .* z) .* (q_x .^ 2 .+ q_y .^2)))
	end
	
    FFTplan = plan_fft!(ψ_p, (1,2))
    # output field size
    yp = similar(ψ_p, real(T), (N, 1))
    xp = similar(ψ_p, real(T), (1, N))
	yp .= fftpos(dq * N, N, CenterFT)
	xp .= yp'
 
    params = Params(y, x, yp, xp, L, Q/2)
    buffer = similar(ψ_p)
    buffer2 = similar(buffer)
    return ScalableAngularSpectrumOp(ΔH, H₁, H₂, buffer, buffer2, params, FFTplan)
end


function (sas::ScalableAngularSpectrumOp)(ψ::AbstractArray{T}; return_view=false) where T
    p = sas.FFTplan
    fill!(sas.buffer2, 0)
    ψ_p = set_center!(sas.buffer2, ψ)
    ψ_p = ifftshift!(sas.buffer, ψ_p)
    ψ_p_f = p * ψ_p 
    ψ_p_f .*= sas.ΔH
    ψ_precomp = inv(p) * ψ_p_f
    ψ_precomp .*= sas.H₁
    ψ_p_final = p * ψ_precomp
    ψ_p_final .*= 1 / (1im * sqrt(T(size(ψ_precomp, 1) * size(ψ_precomp, 2))))

    if !isnothing(sas.H₂)
        ψ_p_final .*= sas.H₂
    end
    fftshift!(sas.buffer2, ψ_p_final)

	ψ_final = crop_center(sas.buffer2, size(ψ); return_view)
    return ψ_final
end



function ChainRulesCore.rrule(sas::ScalableAngularSpectrumOp, ψ::AbstractArray{T}) where T
    field = sas(ψ) 
    function sas_pullback(ȳ)
        p = sas.FFTplan
        fill!(sas.buffer2, 0)

        set_center!(sas.buffer2, ȳ)
        ifftshift!(sas.buffer, sas.buffer2)

        if !isnothing(sas.H₂)
            sas.buffer .*= conj.(sas.H₂)
        end
        sas.buffer .*= conj.(1 / (1im * sqrt(T(size(sas.buffer2, 1) * size(sas.buffer2, 2)))))
        sas.buffer .*= (T(size(sas.buffer2, 1) * size(sas.buffer2, 2)))
        inv(p) * sas.buffer 
        sas.buffer .*= conj.(sas.H₁)
        (p) * sas.buffer
        sas.buffer .*= conj.(sas.ΔH)
        inv(p) * sas.buffer
        fftshift!(sas.buffer2, sas.buffer)

        final = crop_center(sas.buffer2, size(ψ)) 

        return NoTangent(), final
    end
    return field, sas_pullback
end
