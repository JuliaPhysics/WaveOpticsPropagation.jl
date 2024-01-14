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

# ╔═╡ c8e01937-dee6-4c0c-b03d-4a91bf86999b
using Statistics

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
α = deg2rad(10f0)

# ╔═╡ dc01bc87-ffd7-400f-bbf2-3b00a3a84b78
L = 200f-6

# ╔═╡ fdb36c00-57e6-4e3a-a9af-ed1282cf774a
y = fftpos(L, N, CenterFT)

# ╔═╡ 89ec7708-f439-4881-9349-f46d0e75ea93
λ = 405f-9

# ╔═╡ cfeb277b-3bc0-4371-b0ab-587ed626ea6c
field = box(Float32, sz, (10,10)) .* exp.(1im .* 2f0 * π ./ λ .* y .* sin(α));

# ╔═╡ ea02bb1c-7098-4c44-bc13-f9f62fcdce48
z = 150f-6

# ╔═╡ 391ca41e-731d-4799-b09d-553c12b949d7
simshow(field)

# ╔═╡ 2a0ef89b-d9ae-4186-9ccf-15d7785ff407
AS = AngularSpectrum(field, z, λ, L)[1]

# ╔═╡ 9cbafe25-3af6-4bcf-833c-8d3d7ca428a2
res = shifted_angular_spectrum(field, z, λ, L, (α , 0))

# ╔═╡ 4efdc02b-4f69-4893-a410-6c6bbb765bab
shift = z * tan(α * 1) / L * N

# ╔═╡ 24b045a3-7828-4e25-a5bc-656f29cb8166
shift2 = z * tan(α * 1) / L

# ╔═╡ cd6f41c7-532b-4681-98e9-fba8a05fb86b
[simshow(res[1][:, :], γ=1) simshow(FourierTools.shift(res[1], (2 * shift, 0)), γ=1) simshow(AS(field)[1], γ=1)]

# ╔═╡ f3f7bc9e-a16f-49d3-920f-031326f5f1af


# ╔═╡ 62389978-8926-4112-9c2f-b3ec23f2b37d
simshow(FourierTools.shift(res[1], (2 *shift, 0)) .|> abs2, γ=0.2)

# ╔═╡ ea032be4-0384-4bd4-ad62-5eca32062af7
simshow(AS(field)[1] .|> abs2, γ=0.2)

# ╔═╡ e08adc5b-04ff-4e7c-bfd5-a33fe6acb0a1
simshow(res[1] .|> abs2, γ=0.2)

# ╔═╡ 2d56f00c-33d4-4bb7-a3fb-d50210ae24c9
Revise.errors()

# ╔═╡ add49c3a-e0a5-4d07-b7b0-f0dd73547d0f
simshow(abs2.((AS(field)[1] .|> abs2).- (FourierTools.shift(res[1], (shift, 0)) .|> abs2)), γ=1)

# ╔═╡ 19a99e3f-af5e-4902-ac06-767901a895ec
mean(AS(field)[1] .+ 1 .≈ res[1] .+ 1)

# ╔═╡ ebfe96db-a2cd-4d98-92eb-3b1ada42fa78
simshow(res[2].H .* res[2].W)

# ╔═╡ e2fabf05-5ddd-48b4-8e5d-67a036ba1bc7
simshow(res[2].W)

# ╔═╡ 508e3c5a-ab89-4ffa-8f7c-9087bc29cbce
size(res[2].W)

# ╔═╡ cb77a12b-f5c7-42f5-a5f9-c47051666be5
fs = fftfreq(size(img, 1), size(img, 1) / L)

# ╔═╡ 7921f448-3d77-45dc-9fb1-411a9a64300f
# ╠═╡ disabled = true
#=╠═╡
shift = fftshift(ifft(fft(ifftshift(img)) .* exp.(1im * 2 * Float32(π) * z .* (sind(20) .* fs))))
  ╠═╡ =#

# ╔═╡ fd16d619-9e2c-45bb-b4d9-6fafd102c771
simshow(shift)

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
# ╠═24b045a3-7828-4e25-a5bc-656f29cb8166
# ╠═cd6f41c7-532b-4681-98e9-fba8a05fb86b
# ╠═f3f7bc9e-a16f-49d3-920f-031326f5f1af
# ╠═62389978-8926-4112-9c2f-b3ec23f2b37d
# ╠═ea032be4-0384-4bd4-ad62-5eca32062af7
# ╠═e08adc5b-04ff-4e7c-bfd5-a33fe6acb0a1
# ╠═2d56f00c-33d4-4bb7-a3fb-d50210ae24c9
# ╠═add49c3a-e0a5-4d07-b7b0-f0dd73547d0f
# ╠═c8e01937-dee6-4c0c-b03d-4a91bf86999b
# ╠═19a99e3f-af5e-4902-ac06-767901a895ec
# ╠═ebfe96db-a2cd-4d98-92eb-3b1ada42fa78
# ╠═e2fabf05-5ddd-48b4-8e5d-67a036ba1bc7
# ╠═508e3c5a-ab89-4ffa-8f7c-9087bc29cbce
# ╠═cb77a12b-f5c7-42f5-a5f9-c47051666be5
# ╠═7921f448-3d77-45dc-9fb1-411a9a64300f
# ╠═fd16d619-9e2c-45bb-b4d9-6fafd102c771
