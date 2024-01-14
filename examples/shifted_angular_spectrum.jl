### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ b11e7be2-b315-11ee-27e7-abecfdbe64b6
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 3a5d9f20-a01d-481b-9858-b8e523ba7a20
using WaveOpticsPropagation, CUDA, IndexFunArrays, NDTools, ImageShow, FileIO, Zygote, Optim, Plots, PlutoUI, FourierTools, NDTools

# ╔═╡ dfc515b5-cfb5-4004-981f-a2262da47bab
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"   
	togoc(x) = use_CUDA[] ? CuArray(x) : x 
end

# ╔═╡ 2f6871e8-7c11-49c0-ba9a-dc498e8eb39d
N = 256

# ╔═╡ 64b448ee-5ccc-4f87-8ee0-20d2d6a41a3b
sz = (N, N)

# ╔═╡ 90286b89-aedd-4ece-b9d6-e5c26c6ad635
α = 10f0

# ╔═╡ dc01bc87-ffd7-400f-bbf2-3b00a3a84b78
L = 100f-6

# ╔═╡ fdb36c00-57e6-4e3a-a9af-ed1282cf774a
y = fftpos(L, N, CenterFT)

# ╔═╡ 89ec7708-f439-4881-9349-f46d0e75ea93
λ = 405f-9

# ╔═╡ cfeb277b-3bc0-4371-b0ab-587ed626ea6c
field = box(Float32, sz, (20,20)) .* exp.(1im .* 2f0 * π ./ λ .* y .* sind(α));

# ╔═╡ ea02bb1c-7098-4c44-bc13-f9f62fcdce48
z = 30f-6

# ╔═╡ 391ca41e-731d-4799-b09d-553c12b949d7
simshow(field);

# ╔═╡ 2a0ef89b-d9ae-4186-9ccf-15d7785ff407
AS = AngularSpectrum(field, z, λ, L)[1]

# ╔═╡ 9cbafe25-3af6-4bcf-833c-8d3d7ca428a2
res = shifted_angular_spectrum(field, z, λ, L, (deg2rad(-α), 0), bandlimit=false)

# ╔═╡ ebfe96db-a2cd-4d98-92eb-3b1ada42fa78
simshow(res[2].H .* res[2].W)

# ╔═╡ cb77a12b-f5c7-42f5-a5f9-c47051666be5
fs = fftfreq(size(img, 1), size(img, 1) / L)

# ╔═╡ cd6f41c7-532b-4681-98e9-fba8a05fb86b
[simshow(res[1][:, :]) simshow(FourierTools.shift(res[1], (-shift, 0))) simshow(AS(field)[1])]

# ╔═╡ fd16d619-9e2c-45bb-b4d9-6fafd102c771
simshow(shift)

# ╔═╡ 4efdc02b-4f69-4893-a410-6c6bbb765bab
shift = z * tand(α) / L * N

# ╔═╡ 7921f448-3d77-45dc-9fb1-411a9a64300f
# ╠═╡ disabled = true
#=╠═╡
shift = fftshift(ifft(fft(ifftshift(img)) .* exp.(1im * 2 * Float32(π) * z .* (sind(20) .* fs))))
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═b11e7be2-b315-11ee-27e7-abecfdbe64b6
# ╠═3a5d9f20-a01d-481b-9858-b8e523ba7a20
# ╠═dfc515b5-cfb5-4004-981f-a2262da47bab
# ╠═2f6871e8-7c11-49c0-ba9a-dc498e8eb39d
# ╠═64b448ee-5ccc-4f87-8ee0-20d2d6a41a3b
# ╠═fdb36c00-57e6-4e3a-a9af-ed1282cf774a
# ╠═cfeb277b-3bc0-4371-b0ab-587ed626ea6c
# ╠═90286b89-aedd-4ece-b9d6-e5c26c6ad635
# ╠═dc01bc87-ffd7-400f-bbf2-3b00a3a84b78
# ╠═89ec7708-f439-4881-9349-f46d0e75ea93
# ╠═ea02bb1c-7098-4c44-bc13-f9f62fcdce48
# ╠═391ca41e-731d-4799-b09d-553c12b949d7
# ╠═2a0ef89b-d9ae-4186-9ccf-15d7785ff407
# ╠═9cbafe25-3af6-4bcf-833c-8d3d7ca428a2
# ╠═4efdc02b-4f69-4893-a410-6c6bbb765bab
# ╠═cd6f41c7-532b-4681-98e9-fba8a05fb86b
# ╠═ebfe96db-a2cd-4d98-92eb-3b1ada42fa78
# ╠═cb77a12b-f5c7-42f5-a5f9-c47051666be5
# ╠═7921f448-3d77-45dc-9fb1-411a9a64300f
# ╠═fd16d619-9e2c-45bb-b4d9-6fafd102c771
