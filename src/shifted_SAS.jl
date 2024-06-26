export ShiftedScalableAngularSpectrum

 # highly optimized version with pre-planning
struct ShiftedScalableAngularSpectrumOP{AP, AP2, AB, PP, P}
    ΔH::AP
    H₁::AP
    H₂::AP2 # could be nothing!
    buffer::AB
    buffer2::AB
    params::PP
    FFTplan::P
end

function ShiftedScalableAngularSpectrum(ψ₀::AbstractArray{CT}, z, λ, _L, _α; 
								 skip_final_phase=true) where {CT} 
	@assert size(ψ₀, 1) == size(ψ₀, 2) "Restricted to auadratic fields."
    
    pad_factor = 2
    set_pad_zero = false
	
    L = _L# _L isa Number : real(CT).((_L, _L)) : real(CT).(L)
    α = real(CT).(_α)

	N = size(ψ₀, 1)
	L_new = pad_factor .* L


	# applies zero padding
	ψ_p = select_region(ψ₀, new_size=size(ψ₀) .* pad_factor)
	k, dx, df, f_x, f_y, x, y = _SAS_propagation_variables(ψ_p, z, λ, L_new)  
	M = λ * z * N / L^2 / 2
    
    sinxy = .+ sin.(α)
    tanxy = .+ tan.(α)

    mysqrt(x) = sqrt(max(0, x))
    Δf = (1 / L_new, 1 / L_new)
    χ = 1 / λ^2 .- abs2.(f_x .+ sinxy[2] / λ) .- abs2.(f_y .+ sinxy[1] / λ)
    Ωx = z .* (tanxy[2] .- (f_x .+ sinxy[2] / λ) ./ mysqrt.(χ) .+ λ .* f_x)
    Ωy = z .* (tanxy[1] .- (f_y .+ sinxy[1] / λ) ./ mysqrt.(χ) .+ λ .* f_y)
    W = (Δf[1] .<= 1 ./ abs.(2 .* Ωx)) .* (Δf[2] .<= 1 ./ abs.(2 .* Ωy))
    

    H_AS = (sqrt.(CT(1) .- abs2.(f_x .* λ .+ sinxy[2]) .- abs2.(f_y .* λ .+ sinxy[1]))
                 .+ λ .* (tanxy[2] .* f_x .+ tanxy[1] .* f_y))
	
    H_Fr = 1 .- abs2.(f_x .* λ) / 2 .- abs2.(f_y .* λ) / 2

	# take the difference here, key part of the ScaledAS
	ΔH = W .* exp.(1im .* k .* z .* (H_AS .- H_Fr)) 
	#ΔH = W .* exp.(1im .* k .* z .* (H_AS .+ H_Fr)) 
	
    #ΔH = W .* exp.(1im .* (CT(2π) .* z .* sqrt.(CT(1) / λ.^2  .- (f_x .+ sinxy[2] / λ).^2 .- (f_y .+ sinxy[1] / λ).^2) .-
    #                       CT(2π) .* z ./ λ .* ((λ .* f_x).^2 ./ 2 .+ (λ .* f_y).^2 ./ 2) .+ CT(2π) .* z .* (tanxy[1] .* f_y .+ tanxy[2] .* f_x)))
	
	# new sample coordinates
	dq = λ * z / L_new
	Q = dq * N * pad_factor
    q_yx_shift = tanxy .* z
    
    q_y = similar(ψ_p, real(CT), (pad_factor * N, 1))
    q_x = similar(ψ_p, real(CT), (1, pad_factor * N))
	q_y .= fftpos(dq * pad_factor * N, pad_factor * N, CenterFT)
    q_y = q_yx_shift[1] .+ ifftshift(q_y)
    q_x .= q_yx_shift[2] .+ q_y'
	
	# calculate phases of Fresnel
    H₁ = exp.(1im .* k ./ (2 .* z) .* ((x) .^ 2 .+ (y) .^ 2 ))
	
	# skip final phase because often undersampled
	if skip_final_phase
	    H₂ = nothing	
	else
		H₂ = (exp.(1im .* k .* z) .*
			exp.(1im .* k ./ (2 .* z) .* (q_x .^ 2 .+ q_y .^2)))
	end
	
    FFTplan = plan_fft!(ψ_p, (1,2))

    yp = similar(ψ_p, real(CT), (pad_factor * N, 1))
    xp = similar(ψ_p, real(CT), (1, pad_factor * N))
	yp .= fftpos(dq * pad_factor * N, pad_factor * N, CenterFT)
    yp = q_yx_shift[1] .+ ifftshift(q_y)
    xp .= q_yx_shift[2] .+ q_y'

    params = Params(y, x, yp, xp, L, Q/2)
    buffer = similar(ψ_p)
    buffer2 = similar(buffer)
    return ShiftedScalableAngularSpectrumOP(ΔH, H₁, H₂, buffer, buffer2, params, FFTplan)
end




function (sas::ShiftedScalableAngularSpectrumOP)(ψ::AbstractArray{T}; return_view=false) where T
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

