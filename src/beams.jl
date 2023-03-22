export gauss_beam


function gauss_beam(y::T, x::T, z::T, λ::T, w_0::T) where T
    k = T(2π) / λ
    z_R = T(π) * w_0^2 / λ
	r² = x ^ 2 + y ^ 2
	return w_0 / gauss_w(z, z_R, w_0) * exp(-r² / gauss_w(z, z_R, w_0)^2) * 
            exp(-1im * (k * z + k * r² / 2 / gauss_R(z, z_R)) - gauss_ψ(z, z_R))
end

@inline gauss_R(z, z_R) = z * (1 + (z_R /z)^2)
@inline gauss_ψ(z, z_R) = atan(z, z_R)
@inline gauss_w(z, z_R, w_0) = w_0 * sqrt(1 + (z / z_R)^2)
