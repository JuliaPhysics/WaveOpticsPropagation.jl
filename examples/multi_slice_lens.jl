### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ac905778-bd13-11ee-2efc-c156980d778e
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ db5d6db1-0e20-48cf-90f9-f79a2f123197
using WaveOpticsPropagation, CUDA, IndexFunArrays, NDTools, ImageShow, FileIO, Zygote, Optim, Plots, PlutoUI, FourierTools, NDTools

# ╔═╡ af68a467-a2cb-4562-80b1-6c7bd57fe8ed
TableOfContents()

# ╔═╡ 27266d4e-151e-4af7-becf-aecaa067e634
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"   
	togoc(x) = use_CUDA[] ? CuArray(x) : x 
end

# ╔═╡ 286ef64e-275e-4ed5-8346-682962fb8c40
ImageShow.simshow(x::CuArray) = ImageShow.simshow(Array(x))

# ╔═╡ ea0356ad-a47e-4895-adc4-ee9ce695e295
Plotsheatmap(x::CuArray) = Plots.heatmap(Array(x))

# ╔═╡ 58b1397c-56b6-4f67-911c-f8d68802c405
md"# 1. Define a ball lens"

# ╔═╡ d059c790-9a91-41cd-8b54-f3628335010f
n1 = 1.0f0

# ╔═╡ 0228a79c-1518-4362-8268-8a2c39d1fec7
n2 = 1.5f0

# ╔═╡ c2b77a7a-c359-4cf9-9459-db4a80cf1058
sz = (512, 512)

# ╔═╡ c88ba6a4-4b67-4ea8-826a-17e158158320
radius = 15f-6

# ╔═╡ 2122864f-f77b-4aac-b347-a8209e5f4cad
md"# 2. Define field"

# ╔═╡ ecf588f7-3bc4-4ccd-9837-4fef3c5e596f
L = 60f-6

# ╔═╡ 298962a7-a528-4322-ab4c-7e16a2cf2293
x = fftpos(L, sz[2], CenterFT)'

# ╔═╡ 797339e6-adbe-46df-a270-2737e550e3da
y = fftpos(L, sz[1], CenterFT)

# ╔═╡ db47abea-4e8f-4c4e-a1bd-9a9796848638
waist = 15f-6

# ╔═╡ 0f78e939-3b16-44c3-af17-ec6287ed6b23
field = togoc(0im .+ exp.(.-(x.^2 .+ y.^2) ./ waist.^2));

# ╔═╡ 3e12fb1c-1067-498f-899f-56fbea5fe2ab
simshow(field)

# ╔═╡ fe337699-a8ed-4691-be14-3d0f3f6847b8
ztotal = L

# ╔═╡ d92ecfaf-8a91-4e11-af67-0b58e3ee7a56
Nz = 512

# ╔═╡ 24961b1c-0324-448c-a4a8-addd9add9bcd
z = radius / 3 .+ reshape(fftpos(ztotal, Nz, CenterFT), (1,1, Nz));

# ╔═╡ 1b6da0f1-267c-4df6-80d0-8454fb86611c
n = togoc(Float32.(n1 .+ (n2 - n1) .* (0 .* x .+ y.^2 .+ z.^2 .< radius .^2)));

# ╔═╡ d20c628d-90fd-456e-8580-927b2bf6004a
Plots.heatmap(z[:], x[:], Array(n)[:,100,:])

# ╔═╡ 5484ba8e-cf44-4a6d-9fb4-0c787309c313
Δz = ztotal / Nz

# ╔═╡ eda88933-e0ef-4a5e-9b65-a52fe9f191c0
λ = 632f-9

# ╔═╡ f184ba3c-84b5-4ad8-b8fd-a5ae39a11a66
# for free space air propagation
AS = AngularSpectrum(field, Δz, λ / n1, L)[1]

# ╔═╡ 0f7f0d36-5c3a-452e-a788-8f8e4713bddc
# used for WPM in medium n2
AS2 = AngularSpectrum(field, Δz, λ / n2, L)[1]

# ╔═╡ 67de273f-47c7-4978-a852-7ae06e104ff8
md"# 3. Define Functions"

# ╔═╡ c406fe60-f977-4a84-861e-8e75b4cbefa5
#  propagate in air -> phase shift -> propagate in air -> phase shift -> ...
function MSAS(field, n, AS, Nz, Δz, λ)
	c = typeof(Δz)(2π) / λ * Δz
	out = similar(field, (size(field)..., Nz))

	out[:, :, 1] .= exp.(1im .* c .* (view(n, :, :, 1) .- n1)) .* view(field, :, :, 1)

	for i in 2:Nz
		out[:, :, i] .= exp.(1im .* c .* (view(n, :, :, i) .- n1)) .* AS(@view out[:, :, i-1])[1]
	end

	return out
end

# ╔═╡ b80739d0-6417-47f4-b872-538b8d2c52d8
function find_borders(n)
	first = 0
	second = 0

	for i in 2:size(n, 1)
		if n[i] > n[i-1]
			first = i
		end
		if n[i] < n[i-1]
			second = i
		end
	end
	return first, second
end

# ╔═╡ e36f7a30-3a99-4f77-9e2a-c96ef8c37f21
# similar to
# S. Schmidt, T. Tiess, S. Schröter, R. Hambach, M. Jäger, H. Bartelt, A. Tünnermann, and H. Gross, "Wave-optical modeling beyond the thin-element-approximation," Opt. Express 24, 30188-30200 (2016) 
function WPM(field, n, AS1, AS2, Nz, Δz, λ)
	c = typeof(Δz)(2π) / λ * Δz
	out = similar(field, (size(field)..., Nz))

	out[:, :, 1] .= view(field, :, :, 1)

	for i in 2:Nz
		first, second = find_borders(Array(view(n, :, 127, i)))

		if first == 0 && second == 0
			out[:, :, i] .= AS1(@view out[:, :, i-1])[1]
		else
			res_n1 = AS1(@view out[:, :, i-1])[1]
			res_n2 = AS2(@view out[:, :, i-1])[1]
	
			out[begin:first-1, :, i] .= res_n1[begin:first-1, :]
			out[first:second-1, :, i] .= res_n2[first:second-1, :]
			out[second:end, :, i] .= res_n1[second:end, :]
		end
	end

	return out
end

# ╔═╡ 1990a28d-f87e-4b90-8027-4b27fe2f939b
@mytime out_WPM = WPM(field, n, AS, AS2, Nz, Δz, λ);

# ╔═╡ 40547857-1a70-4194-a7ec-e90829d81aed
@mytime out_MAS = MSAS(field, n, AS, Nz, Δz, λ);

# ╔═╡ 655f7c43-fa15-4882-b52a-8d7e2fb0b44d
md"# 4. Analyze results"

# ╔═╡ f8825b79-85eb-4991-bf47-4ecaedd661e5
begin
	Plots.heatmap(z[:], x[:], Array(abs.(out_MAS[:,127,:])),title="Multi Slice Angular Spectrum")
end

# ╔═╡ fb0e8b4b-ef66-4d40-8b6f-2232f7878303
begin
	Plots.heatmap(z[:], x[:], Array(abs.(out_WPM[:,127,:])), title="WPM")
	#Plots.heatmap!(Array(0.0 .* n[:, 127, :] ))
end

# ╔═╡ c386d509-e7ef-49ee-9d6d-7eb18d93a1ef
[simshow(out_WPM[:,127,:]) simshow(out_MAS[:,127,:])]

# ╔═╡ 9ef44988-0a8d-44c1-b94d-155011182ba8
@bind l Slider(1:4)

# ╔═╡ 82341b70-eb05-4bca-9043-f90002ab6563
simshow(Array(cat(n[:, 127, :], out_WPM[:, 127, :],  out_MAS[:, 127, :], n[:, 127, :], dims=3)[:,:, l]), γ=0.4)

# ╔═╡ d1f42c4b-0ccc-4d11-9ca3-dce49c98f3ad
simshow(Array(1 .* cispi.(n[:, 100, :]) .+ out_MAS[:,100,:]), γ=0.1)

# ╔═╡ Cell order:
# ╠═ac905778-bd13-11ee-2efc-c156980d778e
# ╠═db5d6db1-0e20-48cf-90f9-f79a2f123197
# ╟─af68a467-a2cb-4562-80b1-6c7bd57fe8ed
# ╠═27266d4e-151e-4af7-becf-aecaa067e634
# ╠═286ef64e-275e-4ed5-8346-682962fb8c40
# ╠═ea0356ad-a47e-4895-adc4-ee9ce695e295
# ╟─58b1397c-56b6-4f67-911c-f8d68802c405
# ╠═d059c790-9a91-41cd-8b54-f3628335010f
# ╠═0228a79c-1518-4362-8268-8a2c39d1fec7
# ╠═c2b77a7a-c359-4cf9-9459-db4a80cf1058
# ╠═298962a7-a528-4322-ab4c-7e16a2cf2293
# ╠═797339e6-adbe-46df-a270-2737e550e3da
# ╠═24961b1c-0324-448c-a4a8-addd9add9bcd
# ╠═c88ba6a4-4b67-4ea8-826a-17e158158320
# ╠═1b6da0f1-267c-4df6-80d0-8454fb86611c
# ╠═d20c628d-90fd-456e-8580-927b2bf6004a
# ╟─2122864f-f77b-4aac-b347-a8209e5f4cad
# ╠═0f78e939-3b16-44c3-af17-ec6287ed6b23
# ╠═3e12fb1c-1067-498f-899f-56fbea5fe2ab
# ╠═ecf588f7-3bc4-4ccd-9837-4fef3c5e596f
# ╠═db47abea-4e8f-4c4e-a1bd-9a9796848638
# ╠═fe337699-a8ed-4691-be14-3d0f3f6847b8
# ╠═d92ecfaf-8a91-4e11-af67-0b58e3ee7a56
# ╠═5484ba8e-cf44-4a6d-9fb4-0c787309c313
# ╠═eda88933-e0ef-4a5e-9b65-a52fe9f191c0
# ╠═f184ba3c-84b5-4ad8-b8fd-a5ae39a11a66
# ╠═0f7f0d36-5c3a-452e-a788-8f8e4713bddc
# ╟─67de273f-47c7-4978-a852-7ae06e104ff8
# ╠═c406fe60-f977-4a84-861e-8e75b4cbefa5
# ╠═e36f7a30-3a99-4f77-9e2a-c96ef8c37f21
# ╠═b80739d0-6417-47f4-b872-538b8d2c52d8
# ╠═1990a28d-f87e-4b90-8027-4b27fe2f939b
# ╠═40547857-1a70-4194-a7ec-e90829d81aed
# ╟─655f7c43-fa15-4882-b52a-8d7e2fb0b44d
# ╠═f8825b79-85eb-4991-bf47-4ecaedd661e5
# ╠═fb0e8b4b-ef66-4d40-8b6f-2232f7878303
# ╠═c386d509-e7ef-49ee-9d6d-7eb18d93a1ef
# ╠═9ef44988-0a8d-44c1-b94d-155011182ba8
# ╠═82341b70-eb05-4bca-9043-f90002ab6563
# ╠═d1f42c4b-0ccc-4d11-9ca3-dce49c98f3ad
