### A Pluto.jl notebook ###
# v0.19.22

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

# ╔═╡ 6a9f04fe-ca20-11ed-1ed6-9967577ec81c
begin
	using Pkg
	Pkg.activate(".")
    Pkg.instantiate()
	using Revise
end

# ╔═╡ cba50443-4d13-4c05-b413-6a6f3d3ff1c2
using  WaveOpticsPropagation, Napari, ImageShow, FFTW, CUDA, FourierTools, NDTools, Plots, Colors, PlutoUI

# ╔═╡ f07896d0-808a-470d-8dd8-ed5770a333db
using LinearAlgebra, IndexFunArrays

# ╔═╡ 5a6fa509-7535-4188-9cb8-2ab0097d5bb0
using TestImages

# ╔═╡ ec354723-61c0-4966-a4f7-60efbc53b917
Plots.plotly()

# ╔═╡ 4d1cb1e3-327c-4fbc-a501-18efc2c86096
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	    
	togoc(x) = use_CUDA[] ? CuArray(x) : x
	toc(x) = Array(x)
	toc(x::Array) = x

	ImageShow.simshow(x::CuArray; kwargs...) = simshow(toc(x); kwargs...)
end

# ╔═╡ c3ba5b6b-33d2-4276-ab27-5009167d4d61
N = 256

# ╔═╡ 7ef68ddb-474d-49ea-9510-91c0ab4d073d
field_img = togoc(ComplexF32.((rr2(Float32, (N, N), offset=(120, 200)) .< 10^2) .+ (rr2(Float32, (N, N), offset=(100, 100)) .< 40^2) .+ (rr2(Float32, (N, N), offset=(120, 170)) .< 3^2) ));

# ╔═╡ e5fcbd81-c4f5-4bbc-8f22-dd248be8ec01
L = 2f-3

# ╔═╡ 92d96248-06e5-446b-b856-163cbe8b4f45
L / N

# ╔═╡ da4d0740-46c4-460a-83ef-d78db575a3a4
λ = 405f-9 / 1.5

# ╔═╡ 182fd5d8-857c-44d2-9451-8b6e8a9114e9
zs = togoc(fftpos(16e-3, 256, CenterFT));

# ╔═╡ 38041742-1d46-4fa3-97b4-e5614f39ac33
CUDA.@allowscalar zs[129]

# ╔═╡ 851accf9-ea26-41f7-a2a5-4f0b9e53ad9c
@mytime field_img_p, tt = angular_spectrum(togoc(field_img), zs, λ, L);

# ╔═╡ 62acd577-3402-4aee-9c62-4cc424b61758
simshow(field_img)

# ╔═╡ ddbe7f52-57dc-42d5-9edc-6377eb0bf31d
md"z=$(@bind iz PlutoUI.Slider(1:size(field_img_p, 3), show_value=true))"

# ╔═╡ f9837f4a-e134-4d92-8984-51836bb89f74
simshow((toc(abs2.(selectdim(field_img_p, 3, iz)))), γ=0.5)

# ╔═╡ 4218847d-9a0c-4e4e-a8b6-6c76e0c3564e
begin
	plot(fftpos(L, N, CenterFT), abs2.(toc(selectdim(field_img_p, 3, iz)))[120, :], title="z=$(1000 * toc(zs)[iz])mm", ticks=:native, marker=(:circle,1))
	plot!(fftpos(L, N, CenterFT), abs2.(toc(field_img)[120, :]), marker=(:circle,1))
end

# ╔═╡ 5fb47911-28b4-456c-b647-eed4af09f5e3
sum(abs2.(field_img_p[:, :, 2]))

# ╔═╡ a7325c95-d653-4e7e-ad9c-736a50ba9ea8
sum(abs2.(field_img))

# ╔═╡ 03002431-9a33-4a6b-bb1a-d30f47c3bb45


# ╔═╡ 5bb71ea0-cf7d-4be9-8df3-4d8bdab830d3
simshow(togoc(selectdim(rs_kernel, 3, iz)), γ=0.1)

# ╔═╡ 9cd1b7fd-d262-4ccc-8a2c-b82f0368ce35
extrema(abs2.(rs_kernel))

# ╔═╡ 1b1144fe-b732-40bc-8382-727381f69b85
size(field_img)

# ╔═╡ 32ab279d-8c11-4753-9028-cb1983f1ed99
function real_space_kernel(N, z, λ, L)
	L = 2 * L
	y = togoc(fftpos(L, N, CenterFT))
	x = y'

	r = sqrt.(x.^2 .+ y.^2 .+ z .^2 .+ 1f-7)
	k = π / λ * 2
	kernel =  z ./ r ./ 2 ./ π  .* (1 ./ r .- 1im * k) .* exp.(1im .* k .* r) ./ r
	return kernel
end

# ╔═╡ dac63a43-73c2-44ab-bcaf-e3802b7e9714
typeof(field_img)

# ╔═╡ 5c7993ab-53bd-4991-900d-c8c64ebb76cb
field = CUDA.zeros(ComplexF32, 128, 128, 128); nothing

# ╔═╡ 6ea1e637-112b-49f6-8526-b2a2ae20ef0c
sizeof(field) / 2^30

# ╔═╡ 31e0393f-9b1f-4dab-9a30-55b97d81dbac


# ╔═╡ 751c276f-6916-4db1-9a2f-5ec7379e45e3
p = plan_fft!(field); nothing

# ╔═╡ e61443e0-e3c6-4eff-98f7-e94d55f9f8fa
begin
	GC.gc()
	@mytime p * field; ; nothing
end

# ╔═╡ 5ec4dbe5-9d88-45d4-9a47-3f2e888bb4c4
begin
	y = togoc(range(-10f0, 10f0, 100))
	x = togoc(range(-10f0, 10f0, 100)')

	z = togoc(reshape((range(0f0, 10f0, 100)), 1, 1, :))
end

# ╔═╡ 663ceff3-15be-4a61-8495-263af975de3c
@mytime exp.(1im .* z .* (x.^2 .+ y.^2));

# ╔═╡ d04ce01c-557f-441e-9968-c47df096349d
@mytime exp.(1im .* (x.^2 .+ y.^2)) .^ z;

# ╔═╡ 81295271-bac4-44d7-8990-0741f0b8bb6e
NDTools.expand_dims

# ╔═╡ 7feaec85-9604-4d3b-8f1d-c1da3df20a75
@code_warntype WaveOpticsPropagation._prepare_angular_spectrum(field[:, :, 1], 1f0, 1f0, 1f0)

# ╔═╡ Cell order:
# ╠═6a9f04fe-ca20-11ed-1ed6-9967577ec81c
# ╠═cba50443-4d13-4c05-b413-6a6f3d3ff1c2
# ╠═f07896d0-808a-470d-8dd8-ed5770a333db
# ╠═ec354723-61c0-4966-a4f7-60efbc53b917
# ╠═5a6fa509-7535-4188-9cb8-2ab0097d5bb0
# ╠═4d1cb1e3-327c-4fbc-a501-18efc2c86096
# ╠═c3ba5b6b-33d2-4276-ab27-5009167d4d61
# ╠═7ef68ddb-474d-49ea-9510-91c0ab4d073d
# ╠═e5fcbd81-c4f5-4bbc-8f22-dd248be8ec01
# ╠═92d96248-06e5-446b-b856-163cbe8b4f45
# ╠═da4d0740-46c4-460a-83ef-d78db575a3a4
# ╠═38041742-1d46-4fa3-97b4-e5614f39ac33
# ╠═182fd5d8-857c-44d2-9451-8b6e8a9114e9
# ╠═851accf9-ea26-41f7-a2a5-4f0b9e53ad9c
# ╠═62acd577-3402-4aee-9c62-4cc424b61758
# ╠═f9837f4a-e134-4d92-8984-51836bb89f74
# ╠═ddbe7f52-57dc-42d5-9edc-6377eb0bf31d
# ╠═4218847d-9a0c-4e4e-a8b6-6c76e0c3564e
# ╠═0d275418-1393-41c4-b29a-6b56cb25f728
# ╠═5fb47911-28b4-456c-b647-eed4af09f5e3
# ╠═a7325c95-d653-4e7e-ad9c-736a50ba9ea8
# ╠═a8fb4cfa-61c1-4477-a4e8-af6f8e631a7e
# ╠═4f585a19-fd50-40d3-8e60-27ebf080bce6
# ╠═03002431-9a33-4a6b-bb1a-d30f47c3bb45
# ╠═94e1e172-a807-44cf-8f3f-b98262f3f895
# ╠═6dc5b732-f121-465a-9dd9-fdf27fb73e90
# ╠═09954458-5659-46e2-9711-cae7dfac8e36
# ╠═5bb71ea0-cf7d-4be9-8df3-4d8bdab830d3
# ╠═9cd1b7fd-d262-4ccc-8a2c-b82f0368ce35
# ╠═1b1144fe-b732-40bc-8382-727381f69b85
# ╠═32ab279d-8c11-4753-9028-cb1983f1ed99
# ╠═dac63a43-73c2-44ab-bcaf-e3802b7e9714
# ╠═6ea1e637-112b-49f6-8526-b2a2ae20ef0c
# ╠═5c7993ab-53bd-4991-900d-c8c64ebb76cb
# ╠═31e0393f-9b1f-4dab-9a30-55b97d81dbac
# ╠═751c276f-6916-4db1-9a2f-5ec7379e45e3
# ╠═e61443e0-e3c6-4eff-98f7-e94d55f9f8fa
# ╠═5ec4dbe5-9d88-45d4-9a47-3f2e888bb4c4
# ╠═663ceff3-15be-4a61-8495-263af975de3c
# ╠═d04ce01c-557f-441e-9968-c47df096349d
# ╠═81295271-bac4-44d7-8990-0741f0b8bb6e
# ╠═7feaec85-9604-4d3b-8f1d-c1da3df20a75
