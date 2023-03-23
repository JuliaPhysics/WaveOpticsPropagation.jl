export gauss_beam


gauss_beam(y, x, z, λ, w_0) = throw(ArgumentError("All arguments need to have the same datatype (Float32, Float64, ...)."))

function gauss_beam(y::T, x::T, z::T, λ::T, w_0::T) where T
    k = π / λ * 2
    z_R = π * w_0^2 / λ
	r² = x ^ 2 + y ^ 2
    # don't put exp(i * k * z) into the same exp, it causes some strange wraps
	return w_0 / gauss_w(z, z_R, w_0) * exp(-r² / gauss_w(z, z_R, w_0)^2) * 
            exp.(1im * k * z) * 
            exp(1im * (k * r² / 2 / gauss_R(z, z_R) - gauss_ψ(z, z_R)))
end

@inline gauss_R(z, z_R) = z * (1 + (z_R /z)^2)
@inline gauss_ψ(z, z_R) = atan(z, z_R)
@inline gauss_w(z, z_R, w_0) = w_0 * sqrt(1 + (z / z_R)^2)
