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

# ╔═╡ 5ad8803f-4df4-4d8b-8ca5-a417f15cee58
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 1e3670db-7a12-4187-a4b9-aff613748f84
# this is our main package
using WaveOpticsPropagation

# ╔═╡ 752bacd0-5915-4983-86c7-5ef945f96130
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Plots, NDTools

# ╔═╡ cd1b749d-8f50-428a-9d07-f4f0b797fb8d
using CUDA

# ╔═╡ eb37d998-ef91-4f6e-8ccd-498693c6fd26
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ fae7df2a-d9ab-4400-b1b5-da4ec9f42bd5
TableOfContents()

# ╔═╡ 997e13f0-5892-4dc5-9948-0bb8cecaa041
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ 27603905-781f-47f9-92d4-ebe71cff55ee
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ dd520fa8-f149-478a-87a5-948d8e02da4d
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 812f7288-3815-4e87-8a5d-57f20a41d3ba
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 326d9c32-53db-4e61-84c9-1568beb37590
md"# 1. Generature Aperture"

# ╔═╡ c8dcc43d-9274-4619-a820-dee0a11552f6
L = 1f-3

# ╔═╡ 7c3812e8-24d1-4408-86cc-2332c04dfc79
λ = 633f-9

# ╔═╡ 7c95e8e2-0dc6-4d3e-b57a-3e8443847ba7
N = 256

# ╔═╡ 1feb58bc-74cd-4dc6-93bd-caf0f318746c
x = WaveOpticsPropagation.fftpos(L, N, CenterFT)

# ╔═╡ 85dbabf2-b9d5-4413-ac21-090625e63532
@bind ΔS Slider(0:N÷4, show_value=true, default=12)

# ╔═╡ cd0a4108-5d97-4baa-b3d7-38b6d7078240
S = x[1 + ΔS * 2] - x[1]

# ╔═╡ d9b13717-e79b-40ce-8fda-6d9d2cfdba8d
@bind ΔW Slider(0:N÷2, show_value=true, default=2)

# ╔═╡ d6cbc0f5-8572-4654-a270-f4aa4c1e5866
W = x[1 + ΔW * 2 + 1] - x[1]

# ╔═╡ e624fa70-e85b-4107-841d-021e4a699235
begin
	aperture = zeros(ComplexF32, (N, N))

	mid = N ÷ 2+ 1
	aperture[:, mid-ΔS-ΔW:mid-ΔS+ΔW] .= 1
	aperture[:, mid+ΔS-ΔW:mid+ΔS+ΔW] .= 1
end;

# ╔═╡ cad80e46-1699-4472-aae7-1a8f186f5d04
simshow(aperture)

# ╔═╡ 966b102f-5559-4914-a9ed-85515c249ca9
md"# 2. Propagate with Angular Spectrum"

# ╔═╡ 6d2941d4-810a-4edf-b909-159a8d7fb3c5
z = 30f-3

# ╔═╡ 229bbcb4-b4e6-4dbb-a53d-d6c2b8b6b1e5
AS  = AngularSpectrum(aperture, z, λ, (L, L))

# ╔═╡ 0b2891c4-a0cd-41cc-ad1f-e650f5629c6d
@mytime I = abs2.(AS(aperture));

# ╔═╡ c7876f3a-12d5-4775-820e-61dc47adf189
heatmap(AS.params.xp[:], AS.params.yp[:], I)

# ╔═╡ 4db95ee0-eaf1-4e18-b47a-78e1309d0263
simshow(I)

# ╔═╡ 4514eec2-f2f9-4700-9ba5-7bccd36dd0ea
md"# 3. Propagate with Fraunhofer"

# ╔═╡ d5ea29f8-d643-459b-85af-16a6155d4d46
FR =  Fraunhofer(aperture, z, λ, (L, L))

# ╔═╡ d02afd96-beb7-4693-97b6-1b08f610cbea
Revise.retry()

# ╔═╡ 4cee7136-7f37-48a5-966c-aed9d674ae48
xout = FR.params.xp

# ╔═╡ 3fc36591-730e-43ad-8bad-f3c23e3c40b9
yout = FR.params.yp;

# ╔═╡ 28cf3957-8a7b-45e4-88f8-e5bcd5d56e3a
I_fr = abs2.(FR(aperture));

# ╔═╡ 4761a635-0fed-4407-be3f-cfdbf4fc827e
heatmap(xout[:], yout[:], I_fr)

# ╔═╡ cabbdb5d-1f93-4eeb-8325-a0348a616024
FR.params.Lp

# ╔═╡ d0f4a46e-4245-4c59-9180-9571bf0cdebf
md"# 4. Propagate with Scalable Angular Spectrum"

# ╔═╡ ca57e236-f4b9-401a-b996-d5aaa46d3d71
SAS =  ScalableAngularSpectrum(aperture, z, λ, L)

# ╔═╡ a22b1ecb-894b-494c-8888-e1b3f1a48f97
xsas = SAS.params.xp

# ╔═╡ fedcd534-120c-41aa-bd60-19734d32597a
ysas = SAS.params.yp;

# ╔═╡ b59ebf87-4b86-4678-baad-791eaa9c28d6
I_sas = abs2.(SAS(aperture));

# ╔═╡ e4ef052b-8e17-49b1-913a-33e5c32e7436
M = SAS.params.Lp / SAS.params.L

# ╔═╡ fd4d69e4-828e-4c95-898b-c70fa003adf4
xsas

# ╔═╡ cecb84d0-fbcf-41ed-898e-c2b603ee0f04
heatmap(xsas[:], ysas[:], I_sas)

# ╔═╡ 88200bb4-06e4-49aa-8a31-390344f54da6


# ╔═╡ a8ea0c36-9fa9-4933-be4d-551565758f59


# ╔═╡ Cell order:
# ╠═5ad8803f-4df4-4d8b-8ca5-a417f15cee58
# ╠═1e3670db-7a12-4187-a4b9-aff613748f84
# ╠═752bacd0-5915-4983-86c7-5ef945f96130
# ╠═cd1b749d-8f50-428a-9d07-f4f0b797fb8d
# ╠═eb37d998-ef91-4f6e-8ccd-498693c6fd26
# ╠═fae7df2a-d9ab-4400-b1b5-da4ec9f42bd5
# ╠═997e13f0-5892-4dc5-9948-0bb8cecaa041
# ╠═27603905-781f-47f9-92d4-ebe71cff55ee
# ╠═dd520fa8-f149-478a-87a5-948d8e02da4d
# ╠═812f7288-3815-4e87-8a5d-57f20a41d3ba
# ╠═326d9c32-53db-4e61-84c9-1568beb37590
# ╠═c8dcc43d-9274-4619-a820-dee0a11552f6
# ╠═7c3812e8-24d1-4408-86cc-2332c04dfc79
# ╠═7c95e8e2-0dc6-4d3e-b57a-3e8443847ba7
# ╠═d6cbc0f5-8572-4654-a270-f4aa4c1e5866
# ╠═cd0a4108-5d97-4baa-b3d7-38b6d7078240
# ╠═1feb58bc-74cd-4dc6-93bd-caf0f318746c
# ╟─85dbabf2-b9d5-4413-ac21-090625e63532
# ╟─d9b13717-e79b-40ce-8fda-6d9d2cfdba8d
# ╠═e624fa70-e85b-4107-841d-021e4a699235
# ╠═cad80e46-1699-4472-aae7-1a8f186f5d04
# ╟─966b102f-5559-4914-a9ed-85515c249ca9
# ╠═6d2941d4-810a-4edf-b909-159a8d7fb3c5
# ╠═229bbcb4-b4e6-4dbb-a53d-d6c2b8b6b1e5
# ╠═0b2891c4-a0cd-41cc-ad1f-e650f5629c6d
# ╠═c7876f3a-12d5-4775-820e-61dc47adf189
# ╠═4db95ee0-eaf1-4e18-b47a-78e1309d0263
# ╟─4514eec2-f2f9-4700-9ba5-7bccd36dd0ea
# ╠═d5ea29f8-d643-459b-85af-16a6155d4d46
# ╠═d02afd96-beb7-4693-97b6-1b08f610cbea
# ╠═4cee7136-7f37-48a5-966c-aed9d674ae48
# ╠═3fc36591-730e-43ad-8bad-f3c23e3c40b9
# ╠═28cf3957-8a7b-45e4-88f8-e5bcd5d56e3a
# ╠═4761a635-0fed-4407-be3f-cfdbf4fc827e
# ╠═cabbdb5d-1f93-4eeb-8325-a0348a616024
# ╟─d0f4a46e-4245-4c59-9180-9571bf0cdebf
# ╠═ca57e236-f4b9-401a-b996-d5aaa46d3d71
# ╠═a22b1ecb-894b-494c-8888-e1b3f1a48f97
# ╠═fedcd534-120c-41aa-bd60-19734d32597a
# ╠═b59ebf87-4b86-4678-baad-791eaa9c28d6
# ╠═e4ef052b-8e17-49b1-913a-33e5c32e7436
# ╠═fd4d69e4-828e-4c95-898b-c70fa003adf4
# ╠═cecb84d0-fbcf-41ed-898e-c2b603ee0f04
# ╠═88200bb4-06e4-49aa-8a31-390344f54da6
# ╠═a8ea0c36-9fa9-4933-be4d-551565758f59
