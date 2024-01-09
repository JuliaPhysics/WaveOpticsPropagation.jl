### A Pluto.jl notebook ###
# v0.19.30

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

# ╔═╡ 1ea0729c-ad94-11ee-3b52-bb21d59e9509
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ aafab549-c977-48f4-9dba-9d693e439b44
using WaveOpticsPropagation, CUDA, IndexFunArrays, NDTools, ImageShow, FileIO, Zygote, Optim, Plots, PlutoUI, FourierTools, NDTools

# ╔═╡ 4aabd650-792c-407c-9590-e4960ac776d6
md"# Setup Environment and packages"

# ╔═╡ 3ed16b3f-08a8-4fc3-97cf-a884c452dff0
TableOfContents()

# ╔═╡ 20734e1c-faf7-4117-8652-528eca92c6a2
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"   
	togoc(x) = use_CUDA[] ? CuArray(x) : x 
end

# ╔═╡ 7028f15f-7ec7-48b3-9b4d-66bd0c34bae6
λ = 500f-9

# ╔═╡ f9869ff6-5b21-40a3-98cf-33e7192dfa8e
L = 128f-6 / 2

# ╔═╡ 1666eaa6-7857-415e-a43e-00849571db2a
N = 512

# ╔═╡ d418e87f-675f-4bc2-8eaf-70cf669aeb8d
y = fftpos(L, N, NDTools.CenterFT)

# ╔═╡ 945ac45e-2773-4d34-878e-4f66a253afa1
D_circ = N / 8

# ╔═╡ 4f7aca22-00d9-406c-90e1-f73394a718ee
U_circ = togoc(ComplexF32.(rr((N, N)) .< D_circ / 2) .* exp.(1im .* 2f0 * π ./ λ .* y .* sind(45f0)) .+ ComplexF32.(rr((N, N)) .< D_circ / 2) .* exp.(1im .* 2f0 * π ./ λ .* y' .* sind(-45f0)));

# ╔═╡ 34c326ee-6663-432e-b884-a5419ce64827
@bind M Slider(1:0.1:20, show_value=true, default=4)

# ╔═╡ 4202033d-62e6-451f-a064-e61d420da5ff
z_circ = Float32(M / N / λ * L^2 * 2)

# ╔═╡ 4dc0d972-88b7-4cb4-9ef3-925b154e45c0
md"# First Example: Circular

In the first example, one straight beam and one oblique beam are passing through a round aperture.

The Fresnel number is $(round((D_circ / 2 * L / N)^2 / z_circ / λ, digits=3))
"

# ╔═╡ e5c50250-e3c0-4561-842a-d6c671999741
simshow(Array(U_circ))

# ╔═╡ 484704ce-4497-4567-b387-aad06dee6a62
@mytime SAS, _ = ScalableAngularSpectrum(U_circ, z_circ, λ, L)

# ╔═╡ 9e3f7e53-3fdb-4638-89bc-27bced937193
@time SAS_cpu, _ = ScalableAngularSpectrum(Array(U_circ), z_circ, λ, L)

# ╔═╡ 9ec9d456-a33b-4605-9025-c82676eca7e2
U_circ_cpu = Array(U_circ);

# ╔═╡ cf8d99c6-120d-45a6-9fcc-9d1cb6e40a9b
@time SAS_cpu(U_circ_cpu);

# ╔═╡ 342b000d-8865-4cd2-bb6b-2de0f6376e9c
@mytime U_p, t = SAS(U_circ)

# ╔═╡ fc7dd2c9-7c7b-45a7-9fed-a319a62375cc
simshow(Array(abs2.(U_p)), γ=0.13, cmap=:inferno)

# ╔═╡ 505bd6fa-2717-4297-b8b5-bf9452300655
sum(abs2, U_p)

# ╔═╡ bb1c90d3-62c0-4941-b8ac-00761856a3cd
sum(abs2, U_circ)

# ╔═╡ 70afe821-c31c-4626-8b80-20695a46544f
L_box = 128f-6;

# ╔═╡ cd2bdc4a-7abe-4184-a9c3-83b9820cf46d
N_box = 512;

# ╔═╡ e227181f-017a-4601-914a-f63583755105
y_box = fftpos(L_box, N_box, NDTools.CenterFT);

# ╔═╡ f7926732-1e17-428b-9660-2bf3653a3f4a
D_box = L_box / 16f0

# ╔═╡ 4a30f342-cad6-46d5-a787-2144909c8765
x_box = y_box';

# ╔═╡ f0cf4b3d-13fd-48d0-9890-1a8e21686a7e
U_box = togoc((x_box.^2 .<= (D_box / 2).^2) .* (y_box.^2 .<= (D_box / 2).^2) .* (exp.(1im .* 2f0 * π ./ λ .* y_box' .* sind(20f0))));

# ╔═╡ 11947eea-b2eb-4168-bc96-8e1fc7b74599
@bind M_box Slider(1:0.5:20, show_value=true, default=8)

# ╔═╡ aacc5f18-05b5-4a5e-9217-fa62966830e5
z_box = M_box / N_box / λ * L_box^2 * 2

# ╔═╡ d325fa62-7895-44a9-87de-718e9d61f9bc
md"# Second Example: Quadratic


The Fresnel number is $(round((D_box)^2 / z_box / λ, digits=3))
"

# ╔═╡ 38bc91ff-3188-4129-aa6b-2af458cf59b1
@mytime SAS2, _ = ScalableAngularSpectrum(U_box, z_box, λ, L_box, skip_final_phase=true)

# ╔═╡ 2b11c87d-4746-4be6-81f6-f004d30beae4
@mytime U_box_p, t_box = SAS2(U_box)

# ╔═╡ 9e762dfc-4873-4fe3-b785-f6c4808417a6
simshow(Array(abs2.(U_box_p)), γ=0.13, cmap=:inferno)

# ╔═╡ Cell order:
# ╟─4aabd650-792c-407c-9590-e4960ac776d6
# ╠═1ea0729c-ad94-11ee-3b52-bb21d59e9509
# ╠═3ed16b3f-08a8-4fc3-97cf-a884c452dff0
# ╠═aafab549-c977-48f4-9dba-9d693e439b44
# ╠═20734e1c-faf7-4117-8652-528eca92c6a2
# ╠═4dc0d972-88b7-4cb4-9ef3-925b154e45c0
# ╠═7028f15f-7ec7-48b3-9b4d-66bd0c34bae6
# ╠═f9869ff6-5b21-40a3-98cf-33e7192dfa8e
# ╠═1666eaa6-7857-415e-a43e-00849571db2a
# ╠═d418e87f-675f-4bc2-8eaf-70cf669aeb8d
# ╠═945ac45e-2773-4d34-878e-4f66a253afa1
# ╠═4f7aca22-00d9-406c-90e1-f73394a718ee
# ╠═34c326ee-6663-432e-b884-a5419ce64827
# ╠═4202033d-62e6-451f-a064-e61d420da5ff
# ╠═e5c50250-e3c0-4561-842a-d6c671999741
# ╠═484704ce-4497-4567-b387-aad06dee6a62
# ╠═9e3f7e53-3fdb-4638-89bc-27bced937193
# ╠═9ec9d456-a33b-4605-9025-c82676eca7e2
# ╠═cf8d99c6-120d-45a6-9fcc-9d1cb6e40a9b
# ╠═342b000d-8865-4cd2-bb6b-2de0f6376e9c
# ╠═fc7dd2c9-7c7b-45a7-9fed-a319a62375cc
# ╠═505bd6fa-2717-4297-b8b5-bf9452300655
# ╠═bb1c90d3-62c0-4941-b8ac-00761856a3cd
# ╟─d325fa62-7895-44a9-87de-718e9d61f9bc
# ╠═70afe821-c31c-4626-8b80-20695a46544f
# ╠═cd2bdc4a-7abe-4184-a9c3-83b9820cf46d
# ╠═e227181f-017a-4601-914a-f63583755105
# ╠═f7926732-1e17-428b-9660-2bf3653a3f4a
# ╠═4a30f342-cad6-46d5-a787-2144909c8765
# ╠═f0cf4b3d-13fd-48d0-9890-1a8e21686a7e
# ╠═11947eea-b2eb-4168-bc96-8e1fc7b74599
# ╠═aacc5f18-05b5-4a5e-9217-fa62966830e5
# ╠═38bc91ff-3188-4129-aa6b-2af458cf59b1
# ╠═2b11c87d-4746-4be6-81f6-f004d30beae4
# ╠═9e762dfc-4873-4fe3-b785-f6c4808417a6
