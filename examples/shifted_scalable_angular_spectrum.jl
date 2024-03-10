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
z = 50f-6

# ╔═╡ 8bd4b453-8f6c-4270-98ea-ffbb97489f29
M = λ * z / L^2 * size(field, 1) / 2

# ╔═╡ 3cd421df-5f7b-4e48-b13e-35480200d4a7
simshow(field)

# ╔═╡ 8dd05266-5383-408b-a747-b50a0f4fba66
res_AS = AngularSpectrum(field, z, λ, L)(field);

# ╔═╡ 50f75305-9107-4495-8ae9-68cd11498e16
res = ShiftedAngularSpectrum(field, z, λ, L, (α , 0), bandlimit=true)(field);

# ╔═╡ c95c1b1e-7645-4488-883e-e45be15717d8
res2 = WaveOpticsPropagation.ShiftedScalableAngularSpectrum(field, z, λ, L, (deg2rad(α) , 0))(field);

# ╔═╡ 2e41d692-4705-47f1-81f9-8e388d265847
res3 = WaveOpticsPropagation.ScalableAngularSpectrum(field, z, λ, L)(field);

# ╔═╡ 50055786-7728-4580-81e5-4ebb2e05626b
shift = (z .* tan.(α) ./ L .* N)[1]

# ╔═╡ 4a49a6e5-b50d-4d65-ad1a-75eda9b1a280
shift2 = z .* tan.(α) ./ L

# ╔═╡ 888cff79-9df6-4bb6-bdef-bf42ad0385cb
simshow(abs2.(res3))

# ╔═╡ a6f0aae0-2a90-48c2-8a67-1407576cc1ba
simshow(abs2.(res))

# ╔═╡ ba3721fa-9eae-4dc4-ae36-765ce2b4bd5a
simshow(abs2.(res2), γ=1)

# ╔═╡ ab8a3aa5-ca88-4225-b980-5ec14bfd7a86
@mytime SSAS =  WaveOpticsPropagation.ShiftedScalableAngularSpectrum(field, z, λ, L, (α , 0));

# ╔═╡ 942f76ba-9046-4c2e-9572-836d8269bde1
@mytime SAS =  WaveOpticsPropagation.ScalableAngularSpectrum(field, z, λ, L);

# ╔═╡ 54441170-9310-411e-89dd-a58351469e04
simshow(fftshift(SSAS.ΔH), γ=1)

# ╔═╡ 1f6bf5e1-39a5-4fc9-a9ad-5a988a004bf9
simshow(fftshift(SAS.ΔH), γ=1)

# ╔═╡ 0dbc516f-34f3-4047-b3db-d96147b22b13
[simshow(res[round(Int, shift)+1:end, :], γ=1) simshow(FourierTools.shift(res, (shift, 0))[round(Int, shift)+1:end, :], γ=1) simshow(res_AS[round(Int, shift)+1:end, :], γ=1)]

# ╔═╡ 1c779d41-29e3-4ea0-a2f7-4e36297679a1
 simshow(res_AS[round(Int, shift)+1:end, :], γ=1)

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
# ╠═8bd4b453-8f6c-4270-98ea-ffbb97489f29
# ╠═3cd421df-5f7b-4e48-b13e-35480200d4a7
# ╠═8dd05266-5383-408b-a747-b50a0f4fba66
# ╠═50f75305-9107-4495-8ae9-68cd11498e16
# ╠═c95c1b1e-7645-4488-883e-e45be15717d8
# ╠═2e41d692-4705-47f1-81f9-8e388d265847
# ╠═50055786-7728-4580-81e5-4ebb2e05626b
# ╠═4a49a6e5-b50d-4d65-ad1a-75eda9b1a280
# ╠═888cff79-9df6-4bb6-bdef-bf42ad0385cb
# ╠═a6f0aae0-2a90-48c2-8a67-1407576cc1ba
# ╠═ba3721fa-9eae-4dc4-ae36-765ce2b4bd5a
# ╠═ab8a3aa5-ca88-4225-b980-5ec14bfd7a86
# ╠═942f76ba-9046-4c2e-9572-836d8269bde1
# ╠═54441170-9310-411e-89dd-a58351469e04
# ╠═1f6bf5e1-39a5-4fc9-a9ad-5a988a004bf9
# ╠═0dbc516f-34f3-4047-b3db-d96147b22b13
# ╠═1c779d41-29e3-4ea0-a2f7-4e36297679a1
