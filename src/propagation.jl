"""
    _propagation_variables(field, λ, L)

Internal method to create variables we need for propagation such as frequencies in Fourier space, etc..
"""
function _propagation_variables(field::AbstractArray{T, M}, λ::Number, L::NTuple{2, T2}) where {T, T2, M} 
	# wave number
    k = real(T)(2π) / λ
	# number of samples
	Ns = size(field)[1:2]
	# sample spacing
	dx = L ./ Ns

	# frequency spacing
	df = 1 ./ L 
	
    # total size in frequency space
	Lf = Ns .* df
	
	# frequencies centered around first entry 
	# 1D vectors each
	f_y = similar(field, real(eltype(field)), (Ns[1], 1))
	f_y .= fftfreq(Ns[1], Lf[1])
	f_x = similar(field, real(eltype(field)), (1, Ns[2]))
	f_x .= fftfreq(Ns[2], Lf[2])'
	
	# y and x positions in real space, use correct spacing -> fftpos
	y = similar(field, real(eltype(field)), (Ns[1], 1))
	y .= (fftpos(L[1], Ns[1], CenterFT))
	x = similar(field, real(eltype(field)), (1, Ns[2]))
	x .= (fftpos(L[2], Ns[2], CenterFT))'
	
	return (; k, dx, df, f_x, f_y, x, y)
end


function _propagation_variables(field::AbstractArray{T, M}, λ, L::Number) where {T, M} 
    _propagation_variables(field, λ, (L, L)) 
end



function _real_space_kernel(field::AbstractArray{T, N}, z, λ, L) where {T, N}
    L = L isa Number ? (L, L) : L
	# wave number
	k = T(2π) / λ
	# number of samples
	Ns = size(field)[1:2]
	
    # y and x positions in real space, use correct spacing -> fftpos
	y = similar(field, real(eltype(field)), (Ns[1], 1))
	y .= (fftpos(L[1], Ns[1], CenterFT))
	x = similar(field, real(eltype(field)), (1, Ns[2]))
	x .= (fftpos(L[2], Ns[2], CenterFT))'

    
    #y = ifftshift(y, (1,))
    #x = ifftshift(x, (2,))

	r = sqrt.(x.^2 .+ y.^2 .+ z .^2)
    kernel =  T(2 ./ π) .* z ./ r .* (1 ./ r .- 1im * k) .* exp.(1im .* k .* r) ./ r
    kernel .*= CuArray(rr2(size(field)) .< 60 .^2)
	return kernel
end
