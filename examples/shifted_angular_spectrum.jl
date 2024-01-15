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

# ╔═╡ 517b00de-e25a-4688-ac2a-5ca067d7cef7
md"# Define field with a ramp"

# ╔═╡ 2f6871e8-7c11-49c0-ba9a-dc498e8eb39d
N = 256

# ╔═╡ 64b448ee-5ccc-4f87-8ee0-20d2d6a41a3b
sz = (N, N)

# ╔═╡ 90286b89-aedd-4ece-b9d6-e5c26c6ad635
α = deg2rad(10f0)

# ╔═╡ dc01bc87-ffd7-400f-bbf2-3b00a3a84b78
L = 50f-6

# ╔═╡ fdb36c00-57e6-4e3a-a9af-ed1282cf774a
y = fftpos(L[1], N, CenterFT)

# ╔═╡ 89ec7708-f439-4881-9349-f46d0e75ea93
λ = 405f-9

# ╔═╡ cfeb277b-3bc0-4371-b0ab-587ed626ea6c
field = box(Float32, sz, (20,20)) .* exp.(1im .* 2f0 * π ./ λ .* y .* sin(α));

# ╔═╡ ea02bb1c-7098-4c44-bc13-f9f62fcdce48
z = 100f-6

# ╔═╡ 391ca41e-731d-4799-b09d-553c12b949d7
simshow(field)

# ╔═╡ 83982659-dccb-4691-bac1-53abcfc9a88b
md"# Compare AS and shifted AS"

# ╔═╡ 2a0ef89b-d9ae-4186-9ccf-15d7785ff407
res_AS = AngularSpectrum(field, z, λ, L)[1](field)[1];

# ╔═╡ 9cbafe25-3af6-4bcf-833c-8d3d7ca428a2
res = ShiftedAngularSpectrum(field, z, λ, L, (α , 0), bandlimit=true)[1](field)

# ╔═╡ 4efdc02b-4f69-4893-a410-6c6bbb765bab
shift = (z .* tan.(α) ./ L .* N)[1]

# ╔═╡ 24b045a3-7828-4e25-a5bc-656f29cb8166
shift2 = z .* tan.(α) ./ L

# ╔═╡ 8afd2051-66dc-4b46-b7d8-13dd752b98da
md"
Left is shifted AS, middle is shifted AS but shifted in real space such that it roughly fits to AS. Right is AS
"

# ╔═╡ cd6f41c7-532b-4681-98e9-fba8a05fb86b
[simshow(res[1][round(Int, shift)+1:end, :], γ=1) simshow(FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :], γ=1) simshow(res_AS[round(Int, shift)+1:end, :], γ=1)]

# ╔═╡ 0c96fc1e-57ee-4f7f-8c0c-2cc8e65cc5b1
simshow(FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :], γ=1) 

# ╔═╡ 6acb594c-2f4d-44a6-b795-786b71c9657b
simshow(res_AS[round(Int, shift)+1:end, :], γ=1)

# ╔═╡ 7c9fd69b-9741-40b8-ae41-f4ce19d34594
 simshow(FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :] .- res_AS[round(Int, shift)+1:end, :], γ=1)

# ╔═╡ 1c2b9b4b-6190-476b-8e6b-fab20147f0e9
sum(abs2, res_AS[round(Int, shift)+1:end, :] .-  FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :])

# ╔═╡ 365d1605-7cb7-43d3-a428-85621e56dcd6
sum(abs2, res_AS[round(Int, shift)+1:end, :])

# ╔═╡ 154b48f2-58b8-4e4f-8648-8ea771125be5
sum(abs2, FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :])

# ╔═╡ f3f7bc9e-a16f-49d3-920f-031326f5f1af
all(.≈(1 .+ FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :], 1 .+ res_AS[round(Int, shift)+1:end, :], rtol=5f-2))

# ╔═╡ d7eac41e-c5c0-46f8-9b2f-d2f116b65d95


# ╔═╡ 4c3d7581-df26-4bec-9e0e-44279158d8b9


# ╔═╡ ed3ec3b6-4825-4cd8-9d95-13d978d64584


# ╔═╡ Cell order:
# ╠═b11e7be2-b315-11ee-27e7-abecfdbe64b6
# ╠═3a5d9f20-a01d-481b-9858-b8e523ba7a20
# ╠═dfc515b5-cfb5-4004-981f-a2262da47bab
# ╟─517b00de-e25a-4688-ac2a-5ca067d7cef7
# ╠═2f6871e8-7c11-49c0-ba9a-dc498e8eb39d
# ╠═64b448ee-5ccc-4f87-8ee0-20d2d6a41a3b
# ╠═fdb36c00-57e6-4e3a-a9af-ed1282cf774a
# ╠═cfeb277b-3bc0-4371-b0ab-587ed626ea6c
# ╠═90286b89-aedd-4ece-b9d6-e5c26c6ad635
# ╠═dc01bc87-ffd7-400f-bbf2-3b00a3a84b78
# ╠═89ec7708-f439-4881-9349-f46d0e75ea93
# ╠═ea02bb1c-7098-4c44-bc13-f9f62fcdce48
# ╠═391ca41e-731d-4799-b09d-553c12b949d7
# ╟─83982659-dccb-4691-bac1-53abcfc9a88b
# ╠═2a0ef89b-d9ae-4186-9ccf-15d7785ff407
# ╠═9cbafe25-3af6-4bcf-833c-8d3d7ca428a2
# ╠═4efdc02b-4f69-4893-a410-6c6bbb765bab
# ╠═24b045a3-7828-4e25-a5bc-656f29cb8166
# ╟─8afd2051-66dc-4b46-b7d8-13dd752b98da
# ╠═cd6f41c7-532b-4681-98e9-fba8a05fb86b
# ╠═0c96fc1e-57ee-4f7f-8c0c-2cc8e65cc5b1
# ╠═6acb594c-2f4d-44a6-b795-786b71c9657b
# ╠═7c9fd69b-9741-40b8-ae41-f4ce19d34594
# ╠═1c2b9b4b-6190-476b-8e6b-fab20147f0e9
# ╠═365d1605-7cb7-43d3-a428-85621e56dcd6
# ╠═154b48f2-58b8-4e4f-8648-8ea771125be5
# ╠═f3f7bc9e-a16f-49d3-920f-031326f5f1af
# ╠═d7eac41e-c5c0-46f8-9b2f-d2f116b65d95
# ╠═4c3d7581-df26-4bec-9e0e-44279158d8b9
# ╟─ed3ec3b6-4825-4cd8-9d95-13d978d64584
