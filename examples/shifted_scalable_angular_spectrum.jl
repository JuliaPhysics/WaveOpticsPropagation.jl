### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 59413ae0-c2ce-11ee-31f3-c52d9d263afb
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 87a307ea-506d-49d1-af79-ae1f3c52912a
using WaveOpticsPropagation, CUDA, IndexFunArrays, NDTools, ImageShow, FileIO, Zygote, Optim, Plots, PlutoUI, FourierTools, NDTools

# ╔═╡ ea14a26f-9273-4f13-8edf-fc0691a1930f
TableOfContents()

# ╔═╡ 0ee0e517-5408-42d7-9bb3-d0adef6df1e5
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"   
	togoc(x) = use_CUDA[] ? CuArray(x) : x 
end

# ╔═╡ d297ace0-deec-417e-88af-6fb7bf5d3358


# ╔═╡ 53832390-bd97-4394-8f57-a253da354030
N = 256

# ╔═╡ f1389ab1-44b9-485c-a2c1-ec8a7945c1e2
sz = (N, N)

# ╔═╡ a9ab5744-6a83-4928-99cc-19c7058724c1
α = deg2rad(10f0)

# ╔═╡ c487f254-5f32-4e8b-94d7-f18c45a2a3ca
L = 50f-6

# ╔═╡ 652adc09-d4f5-47b9-b5ac-c16e47ee03cc
y = fftpos(L[1], N, CenterFT)

# ╔═╡ 5a06d1f8-ce32-4510-adc2-862a6c48a479
λ = 405f-9

# ╔═╡ 4825d17b-eaff-4ff1-b492-b695666014f1
field = box(Float32, sz, (20,20)) .* exp.(1im .* 2f0 * π ./ λ .* y .* sin(α));

# ╔═╡ 9c9560ca-d6c4-4fe8-9ac9-4de73e6af038
z = 200f-6

# ╔═╡ 3cd421df-5f7b-4e48-b13e-35480200d4a7
simshow(field)

# ╔═╡ 8dd05266-5383-408b-a747-b50a0f4fba66
res_AS = AngularSpectrum(field, z, λ, L)[1](field)[1];

# ╔═╡ 50f75305-9107-4495-8ae9-68cd11498e16
res = ShiftedAngularSpectrum(field, z, λ, L, (α , 0), bandlimit=true)[1](field)

# ╔═╡ c95c1b1e-7645-4488-883e-e45be15717d8
res2 = WaveOpticsPropagation.ShiftedScalableAngularSpectrum(field, z, λ, L, (α , 0))[1](field)

# ╔═╡ 2e41d692-4705-47f1-81f9-8e388d265847
res3 = WaveOpticsPropagation.ScalableAngularSpectrum(field, z, λ, L)[1](field)

# ╔═╡ 1be83926-2d7a-49e9-8b2b-9055d67c87ea
z * λ / L^2 * size(field,1)

# ╔═╡ 016cde31-2cf7-4f56-b7d1-e4f709b2183e
Revise.errors()

# ╔═╡ 50055786-7728-4580-81e5-4ebb2e05626b
shift = (z .* tan.(α) ./ L .* N)[1]

# ╔═╡ 4a49a6e5-b50d-4d65-ad1a-75eda9b1a280
shift2 = z .* tan.(α) ./ L

# ╔═╡ 888cff79-9df6-4bb6-bdef-bf42ad0385cb
simshow(res2[1])

# ╔═╡ 3e012a80-466f-4ffe-ba80-8bd99e5fe230


# ╔═╡ ba3721fa-9eae-4dc4-ae36-765ce2b4bd5a
simshow(res3[1])

# ╔═╡ 0dbc516f-34f3-4047-b3db-d96147b22b13
[simshow(res[1][round(Int, shift)+1:end, :], γ=1) simshow(FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :], γ=1) simshow(res_AS[round(Int, shift)+1:end, :], γ=1)]

# ╔═╡ Cell order:
# ╠═59413ae0-c2ce-11ee-31f3-c52d9d263afb
# ╠═ea14a26f-9273-4f13-8edf-fc0691a1930f
# ╠═87a307ea-506d-49d1-af79-ae1f3c52912a
# ╠═0ee0e517-5408-42d7-9bb3-d0adef6df1e5
# ╠═d297ace0-deec-417e-88af-6fb7bf5d3358
# ╠═53832390-bd97-4394-8f57-a253da354030
# ╠═f1389ab1-44b9-485c-a2c1-ec8a7945c1e2
# ╠═652adc09-d4f5-47b9-b5ac-c16e47ee03cc
# ╠═4825d17b-eaff-4ff1-b492-b695666014f1
# ╠═a9ab5744-6a83-4928-99cc-19c7058724c1
# ╠═c487f254-5f32-4e8b-94d7-f18c45a2a3ca
# ╠═5a06d1f8-ce32-4510-adc2-862a6c48a479
# ╠═9c9560ca-d6c4-4fe8-9ac9-4de73e6af038
# ╠═3cd421df-5f7b-4e48-b13e-35480200d4a7
# ╠═8dd05266-5383-408b-a747-b50a0f4fba66
# ╠═50f75305-9107-4495-8ae9-68cd11498e16
# ╠═c95c1b1e-7645-4488-883e-e45be15717d8
# ╠═2e41d692-4705-47f1-81f9-8e388d265847
# ╠═1be83926-2d7a-49e9-8b2b-9055d67c87ea
# ╠═016cde31-2cf7-4f56-b7d1-e4f709b2183e
# ╠═50055786-7728-4580-81e5-4ebb2e05626b
# ╠═4a49a6e5-b50d-4d65-ad1a-75eda9b1a280
# ╠═888cff79-9df6-4bb6-bdef-bf42ad0385cb
# ╠═3e012a80-466f-4ffe-ba80-8bd99e5fe230
# ╠═ba3721fa-9eae-4dc4-ae36-765ce2b4bd5a
# ╠═0dbc516f-34f3-4047-b3db-d96147b22b13
