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

# ╔═╡ 75796f02-ac0d-11ee-3913-11fa1f818dea
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 9896f57c-0540-4be7-91fb-43f08f9e084c
using WaveOpticsPropagation, CUDA, IndexFunArrays, NDTools, ImageShow, FileIO, Zygote, Optim, Plots, PlutoUI

# ╔═╡ 5a88b44b-ecce-4002-a4ba-87e8b9d55c81
md"# Load and initialize packages"

# ╔═╡ 0c95865d-7998-4191-a637-457c8d04d6d4
TableOfContents()

# ╔═╡ 6dc592f5-7fd3-4782-a72d-23b5dbfa80e2
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"   
	togoc(x) = use_CUDA[] ? CuArray(x) : x 
end

# ╔═╡ aa9e7ca4-945a-4b8c-bc8b-29ab192350e3
md"# Initialize physical parameters"

# ╔═╡ a777cc7a-c79c-494c-9a70-83a90b2ee08d
L = 100f-3

# ╔═╡ 42ae161e-26b7-4d34-9e92-4c4af27fc447
λ = 633f-9

# ╔═╡ 9facb2bf-f279-41de-8858-81dac7a032b5
z = 1

# ╔═╡ 3cde3fbe-9641-4ffd-9630-9c7bf167ec34
N = 256

# ╔═╡ 771642e4-c03e-41a2-a4de-07fdfd4bd535
x = WaveOpticsPropagation.fftpos(L, N)

# ╔═╡ 7f79a274-8f9c-43b5-bb7e-96bb495fb044
L / N

# ╔═╡ ac7c756a-b182-4458-b5ea-017e23fc1009
2 / N * L

# ╔═╡ 6acb2991-cba5-4cf5-8d82-099f9ad2335a
@bind ΔS Slider(0:N÷4, show_value=true, default=12)

# ╔═╡ e5fc1e39-9300-4c96-b747-d532245cfd97
S = x[1 + ΔS * 2] - x[1]

# ╔═╡ 4f9c514a-5f24-4d49-8a92-88f250d9ef92
@bind ΔW Slider(0:N÷2, show_value=true, default=2)

# ╔═╡ d1734ddf-ec04-4134-82a8-187e697c4810
begin
	slit = zeros(ComplexF32, (N, N))

	mid = N ÷ 2+ 1
	slit[:, mid-ΔS-ΔW:mid-ΔS+ΔW] .= 1
	slit[:, mid+ΔS-ΔW:mid+ΔS+ΔW] .= 1
end;

# ╔═╡ 3ebcb7d8-0ed6-497a-95e3-6e40bfadd8d0
simshow(slit)

# ╔═╡ ac72f69a-935e-4ba4-b8bd-e16f27367acd
W = x[1 + ΔW * 2 + 1] - x[1]

# ╔═╡ 3330fef8-e6ed-412a-9321-9dc705ce41fe
md"# Propagate with Fraunhofer Diffraction"

# ╔═╡ 7e797b20-77ee-4888-9b23-74bb2713912f
@time output = fraunhofer(slit, z, λ, L);

# ╔═╡ 0c74a49c-e4de-4172-ac75-4c2692c505fb
# creating this function is more efficient!
efficient_fraunhofer = Fraunhofer(slit, z, λ, L);

# ╔═╡ 0e6f74f1-8723-4596-b754-973b253443a4
@time output2 = efficient_fraunhofer(slit);

# ╔═╡ 5c0486d2-5fbd-4cd7-9fc5-583a8e188b2e
begin
	intensity = abs2.(output)
	intensity ./= maximum(intensity)
end;

# ╔═╡ cd54d590-ea6f-4d3b-84ec-6f983a97c81c
simshow(intensity, γ=1)

# ╔═╡ 5f3c7de8-a64e-42aa-b8a1-91f5869c9610
md"# Compare to analytical solution"

# ╔═╡ 4d836453-7e64-4f3b-9405-d3cdafc264dc
I_analytical(x) = sinc(W * x / λ / z)^2 * cos(π * S * x / λ / z)^2

# ╔═╡ 42e1a449-0e5c-4d67-a90c-10a54f479c8a
xpos_out = WaveOpticsPropagation.fftpos(t.L, N, NDTools.CenterFT)

# ╔═╡ 5d36ba4d-e9fe-4b1b-b45d-c82b598dfe7a
begin
	plot(xpos_out, intensity[(begin+end)÷2+1, :])
	plot!(xpos_out, I_analytical.(xpos_out))
end

# ╔═╡ ba6965bf-c7f1-4dfd-8381-e330582b3462
all(≈(I_analytical.(xpos_out)[110:150] .+ 1, 1 .+  intensity[129, 110:150, :], rtol=1f-2))

# ╔═╡ 8193fd81-1580-4047-ac04-7c3837975c2d
md"# Optimization
We try to find the initial field from one diffraction pattern.

We start with a initial guess of a double slit where distance and width are wrong. By using Fraunhofer as forward model, we are able to optimize for correct distance and width!
"

# ╔═╡ 2c28b256-f60c-48b6-a34a-d2908c979e30
intensity_cu = togoc(intensity);

# ╔═╡ 3a19cbfe-6efc-45f7-9aa3-f319f34a534e
begin
	rec0 = togoc(zeros(Float32, (N, N)));
	
	rec0[:, mid-40:mid-10] .= 1
	rec0[:, mid+10:mid+60] .= 1
end;

# ╔═╡ b19225e0-5536-4af1-bbbc-c6124e91536b
simshow(abs2.(Array(rec0)))

# ╔═╡ e0aa0714-08e0-46f4-b6d4-8c5df044ddd9
md"## Define Forward model and gradient
We use both methods: `fraunhofer` and `Fraunhofer`.
In principle `Fraunhofer` generates some buffers and stores the FFT plan and should be slightly faster.
"

# ╔═╡ c8a3d8b0-f68d-4c38-ae1a-8081846bda23
fwd = let z=z, L=L, λ=λ
	x -> fraunhofer(x .+ 0im, z, λ, L)
end

# ╔═╡ c9651b37-d911-4230-a2f5-f94e8f726fbc
fwd2 = let z=z, L=L, λ=λ, fr=Fraunhofer(rec0 .+ 0im, z, λ, L)
	x -> fr(x .+ 0im)
end

# ╔═╡ e5c3f3dc-5eb3-421e-90ab-a16ea73e5621
f = let intensity_cu=intensity_cu, fwd=fwd
	f(x) = sum(abs2, intensity_cu .- abs2.(fwd(abs2.(x) .+ 0im)))
end

# ╔═╡ cb0b276f-d479-4024-8ca4-116a210fe2ed
f2 = let intensity_cu=intensity_cu, fwd2=fwd2
	f(x) = sum(abs2, intensity_cu .- abs2.(fwd2(abs2.(x) .+ 0im)))
end

# ╔═╡ 30555d35-04a8-44b7-afd7-345730d97a61
@mytime f(rec0);

# ╔═╡ c6c30873-9419-4a54-a9f5-6bf883856102
@mytime f2(rec0);

# ╔═╡ 8ab5cfd9-15e0-4be9-9936-cd499555fbec
@mytime gradient(f, rec0)

# ╔═╡ dd8a5b39-b28a-4b79-857f-8c0a8923f2b3
@mytime gradient(f2, rec0);

# ╔═╡ c9a37a37-eef0-498e-8105-dd29284bfa4e
g!(G, x) = G .= Zygote.gradient(f, x)[1]

# ╔═╡ 6795095b-bbbe-4dec-a79b-12d604873474
g2!(G, x) = G .= Zygote.gradient(f2, x)[1]

# ╔═╡ b75deb02-31e2-4b3d-adad-2f380b107e1b
md"## Optimize"

# ╔═╡ c539d936-acda-4970-9f5d-92538fda2bc5
res = optimize(f, g!, rec0, LBFGS(), Optim.Options(iterations=400))

# ╔═╡ 9563f911-b861-4cb1-87bc-fbbb3a3b77d2
md"## Results
Up to a shift we obtain the correct result! The shift is not possible to find out since we only measure the intensity
"

# ╔═╡ 8c53353c-7339-4060-8d2e-caf8f3025b48
simshow(abs2.(Array(res.minimizer)))

# ╔═╡ ae232c00-b4b7-41ff-8c49-a55382743956
simshow(abs2.(Array(slit)))

# ╔═╡ Cell order:
# ╟─5a88b44b-ecce-4002-a4ba-87e8b9d55c81
# ╠═75796f02-ac0d-11ee-3913-11fa1f818dea
# ╟─0c95865d-7998-4191-a637-457c8d04d6d4
# ╠═9896f57c-0540-4be7-91fb-43f08f9e084c
# ╠═6dc592f5-7fd3-4782-a72d-23b5dbfa80e2
# ╟─aa9e7ca4-945a-4b8c-bc8b-29ab192350e3
# ╠═a777cc7a-c79c-494c-9a70-83a90b2ee08d
# ╠═42ae161e-26b7-4d34-9e92-4c4af27fc447
# ╠═9facb2bf-f279-41de-8858-81dac7a032b5
# ╠═3cde3fbe-9641-4ffd-9630-9c7bf167ec34
# ╠═d1734ddf-ec04-4134-82a8-187e697c4810
# ╠═3ebcb7d8-0ed6-497a-95e3-6e40bfadd8d0
# ╠═771642e4-c03e-41a2-a4de-07fdfd4bd535
# ╠═ac72f69a-935e-4ba4-b8bd-e16f27367acd
# ╠═e5fc1e39-9300-4c96-b747-d532245cfd97
# ╠═7f79a274-8f9c-43b5-bb7e-96bb495fb044
# ╠═ac7c756a-b182-4458-b5ea-017e23fc1009
# ╠═6acb2991-cba5-4cf5-8d82-099f9ad2335a
# ╠═4f9c514a-5f24-4d49-8a92-88f250d9ef92
# ╟─3330fef8-e6ed-412a-9321-9dc705ce41fe
# ╠═7e797b20-77ee-4888-9b23-74bb2713912f
# ╠═0c74a49c-e4de-4172-ac75-4c2692c505fb
# ╠═0e6f74f1-8723-4596-b754-973b253443a4
# ╠═5c0486d2-5fbd-4cd7-9fc5-583a8e188b2e
# ╠═cd54d590-ea6f-4d3b-84ec-6f983a97c81c
# ╟─5f3c7de8-a64e-42aa-b8a1-91f5869c9610
# ╠═4d836453-7e64-4f3b-9405-d3cdafc264dc
# ╠═42e1a449-0e5c-4d67-a90c-10a54f479c8a
# ╠═5d36ba4d-e9fe-4b1b-b45d-c82b598dfe7a
# ╠═ba6965bf-c7f1-4dfd-8381-e330582b3462
# ╟─8193fd81-1580-4047-ac04-7c3837975c2d
# ╠═2c28b256-f60c-48b6-a34a-d2908c979e30
# ╠═3a19cbfe-6efc-45f7-9aa3-f319f34a534e
# ╠═b19225e0-5536-4af1-bbbc-c6124e91536b
# ╟─e0aa0714-08e0-46f4-b6d4-8c5df044ddd9
# ╠═c8a3d8b0-f68d-4c38-ae1a-8081846bda23
# ╠═c9651b37-d911-4230-a2f5-f94e8f726fbc
# ╠═e5c3f3dc-5eb3-421e-90ab-a16ea73e5621
# ╠═cb0b276f-d479-4024-8ca4-116a210fe2ed
# ╠═30555d35-04a8-44b7-afd7-345730d97a61
# ╠═c6c30873-9419-4a54-a9f5-6bf883856102
# ╠═8ab5cfd9-15e0-4be9-9936-cd499555fbec
# ╠═dd8a5b39-b28a-4b79-857f-8c0a8923f2b3
# ╠═c9a37a37-eef0-498e-8105-dd29284bfa4e
# ╠═6795095b-bbbe-4dec-a79b-12d604873474
# ╟─b75deb02-31e2-4b3d-adad-2f380b107e1b
# ╠═c539d936-acda-4970-9f5d-92538fda2bc5
# ╟─9563f911-b861-4cb1-87bc-fbbb3a3b77d2
# ╠═8c53353c-7339-4060-8d2e-caf8f3025b48
# ╠═ae232c00-b4b7-41ff-8c49-a55382743956
