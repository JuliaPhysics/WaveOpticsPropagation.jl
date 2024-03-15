### A Pluto.jl notebook ###
# v0.19.38

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

# ╔═╡ fe97a788-9f64-4099-9740-bf8179b0847b
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 910abe5b-1795-486b-90b0-2d7daabe5240
# this is our main package
using WaveOpticsPropagation

# ╔═╡ 4969f583-5bb4-4023-87b6-50a93706cec9
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Plots, NDTools, NNlib, FFTW

# ╔═╡ 82772fcc-ea3e-4a3b-a2df-207c8e036ed4
using FourierTools

# ╔═╡ 0cb2dda9-1e11-4c9f-893c-a80d744159a6
using Optim, Zygote

# ╔═╡ 2e69f067-2ae9-4591-a28f-360a4ff0ba2e
FFTW.forget_wisdom()

# ╔═╡ 7a9ee3d2-54bd-40ed-bf2f-f176fb73026a
FFTW.set_num_threads(12)

# ╔═╡ 58db1105-99a8-4c0d-854b-2b097bde8fca
TableOfContents()

# ╔═╡ 3d6550a9-502d-4b3f-b7cb-56c41a4d966e
normf(x) = x ./ maximum(x)

# ╔═╡ 8e66d7cb-13be-4b11-ad3d-fbbd087c0139
md"# Defining Physical Parameters"

# ╔═╡ af609bd1-c7e4-4310-af31-c2511726d0e1
sz = (256,256)

# ╔═╡ 449c9203-d1ce-4914-8b55-9d60c464636f
field = Float32.(gaussian(sz, sigma=20)) .+ 0im;

# ╔═╡ bfdd9979-d0e6-4094-8808-a5dc3bdb5b60
simshow(field)

# ╔═╡ bea6cd49-83e4-4620-929c-4893d2baf502
f = 50f-3

# ╔═╡ 59a68666-0ce5-4f62-86c6-17d2f84954d9
L = 2.5e-3

# ╔═╡ 40d10497-ce23-4cbf-b453-db0b82788c87
dx = L / sz[1]

# ╔═╡ 48ebe008-e49f-45d6-9600-6ec0ecb9252d
λ = 500f-9

# ╔═╡ f4bcde25-6130-468f-8564-174bcca22297
md"## Ideal Phase"

# ╔═╡ 66a3b4e3-5b10-496f-9e8d-546359090622
phase = cis.(-2f0 * π ./ λ / 2 / f .* rr2(Float32, sz, scale=dx));

# ╔═╡ b57e0b2e-eaad-4b94-9b0b-8f8ca9318ce6
simshow(phase)

# ╔═╡ b2f503c9-42a9-4caa-bf96-1fbe0532ce21
md"## Wrong phase"

# ╔═╡ 08c2453b-eea8-411d-8611-93bd8714a493
md"## Propagate by the focal length free space"

# ╔═╡ ea5bc70f-952e-4cfd-b406-6a0f67e2f1cc
z = f

# ╔═╡ e375da67-e028-4525-8360-a7c2d9d34c1a
AS = AngularSpectrum(field, f, λ, L)

# ╔═╡ af27ad72-79fc-4ec5-b4d7-e1a354cd644a
lens(x) = AS(x)

# ╔═╡ c25e5f7f-24fa-4617-ba6d-3275cae5631a
@bind fraction Slider(range(0f0, 1f0, 50), show_value=true, default=0.75)

# ╔═╡ e772977f-bc47-412c-8038-001af4d5ef01
phase_clipped = cis.(angle.(phase) .* fraction);

# ╔═╡ d9fa0122-e0cd-4d59-8c94-e29a1edb75a9
simshow(phase_clipped)

# ╔═╡ f0087e5f-8c51-40de-8708-4218216b36b7
I1 = abs2.(lens(field .* phase));

# ╔═╡ c4c37737-927d-464a-868c-b53e7c1fa3b6
I2 = abs2.(lens(field .* phase_clipped));

# ╔═╡ c337b832-46e5-4333-9ec7-009bb59cbfc9
[simshow(I1) simshow(I2)]

# ╔═╡ a6edb05d-ae9c-4e21-84aa-f63d3ae35e74
begin
	plot(normf(I1[sz[1]÷2+1,:]).^0.2, label="Ideal")
	plot!(normf(I2[sz[1]÷2+1,:]).^0.2, label="Clipped Phase")
end

# ╔═╡ 7ae7a052-d991-47ab-872a-3b72c132d524
begin
	plot(normf(FourierTools.ft(I1[sz[1]÷2+1,:]) .|> abs))
	plot!(normf(FourierTools.ft(I2[sz[1]÷2+1,:]) .|> abs))
end

# ╔═╡ 26efcaba-a9d6-497a-854f-addb82b9988c
sum(I1)

# ╔═╡ d0d98b0b-b262-43c6-8f53-18c73e12910e
sum(abs2, field)

# ╔═╡ c9241344-565c-4e1e-85f4-0659ff465423
sum(I2)

# ╔═╡ 48c63470-d520-432a-9b69-0ae6d598879d
rr_arr = rr(I1);

# ╔═╡ 2834ecb4-4abd-4d4c-991e-8eee657e658d
sum(I1 .* rr_arr)

# ╔═╡ e65744b4-07c5-4483-871f-1b0d49eed626
sum(I2 .* rr_arr)

# ╔═╡ 0411b34b-99b2-4142-8992-ef838dee2ad1
md"# Optimization
We introduce a function such that the phase can only be optimized in that range
"

# ╔═╡ cdb6c70d-23ae-400c-a404-89887fd33e48
hh = range(-5, 5, 100)

# ╔═╡ 85342824-d0db-43f5-9ba3-f25e7fdd72d9
map_f = let fraction=fraction
	function map_f(x::T) where T
		return max.(min.(x,T(π * fraction)), T(-π * fraction))
	end
end

# ╔═╡ d14d4d64-1661-4896-9c43-57ebada7c859
plot(hh, map_f.(hh))

# ╔═╡ 83802d1f-ed4a-47fb-864c-07ce623ba180
md"## build Forward model and boilerplate"

# ╔═╡ 4b52f53c-3f41-4265-bd67-b5ef37c3629b
function make_fwd(f, λ, L)
	phase = cis.(-2f0 * π ./ λ / 2 / f .* rr2(Float32, sz, scale=dx))
	AS2 = AngularSpectrum(phase, f, λ, L)
	gb = Float32.(gaussian(sz, sigma=20)) .+ 0im
	
	function fwd(x)
		I = abs2.(AS2(cis.(map_f.(x)) .* gb))
		return I
	end

	return fwd
end

# ╔═╡ b9ceab04-ca9b-4283-8619-18dffd0a7114
fwd = make_fwd(f, λ, L)

# ╔═╡ 1fb087b1-4abc-44d3-b1ab-e9588c40d6a5
md"## Run OptimizatioN"

# ╔═╡ 23236a3b-ba23-41e0-84a4-058425d284cc
init0 = map_f.(angle.(phase))#0.1f0 .* randn(real(eltype(phase)), size(phase)) .+ map_f.(angle.(phase))

# ╔═╡ f85dcffc-502e-40b3-aaf8-b3cdf1473bd7
simshow(select_region(init0, M=1), set_zero=true)

# ╔═╡ 1b8de892-f564-4b06-a1c2-ade26562dfae
@time fwd(init0);

# ╔═╡ 827c9db5-91cd-4b4e-87b0-4b01cef6639f
sum(I1 .* rr_arr)

# ╔═╡ 6aa542f9-e96c-4be3-b101-b52b77040f8d
sum(I2 .* rr_arr)

# ╔═╡ 2c0d8704-08fe-4074-87b9-00a392cae2a8


# ╔═╡ 65357531-77e3-4c76-a5de-1c0c36d3cc6c
md"## Look at different norms"

# ╔═╡ 65e9381d-f14d-4184-89c1-006e1578f040
sum(I1 .* rr_arr) / sum(I1)

# ╔═╡ def82b67-9c0e-4f8e-8fdb-9260f4151fbd
rr_arr2 = rr((10000,))

# ╔═╡ 19d3ff40-4826-4935-92ca-004f29791f7e
sigma = 1

# ╔═╡ 27804c15-7c85-4305-9908-c23d69c8eded
sum((gaussian((10000, ); sigma) .* rr_arr2)) / sum(gaussian((10000,);sigma))

# ╔═╡ d38c7f83-b7e8-453a-8149-5d751ef01081
sqrt(10)

# ╔═╡ aa62237c-3eb4-46d5-ab6f-b6d9dd89de6a
sum(gaussian(sz, sigma=10))

# ╔═╡ cc8e5830-449e-4fd3-bfc6-09bb19b56ac7
f3(x) = begin
	f = abs.(fftshift(fft(ifftshift(x))))
	-sum(f .* rr_arr) / sum(f)
end

# ╔═╡ 7c0e2c54-d187-40fd-a8df-922521c7bac2
function make_fg!(fwd)
	l(x) = begin
		I = fwd(x)
		f3(I) + sum(rr_arr .* I) / sum(I)
	end
    # some Optim boilerplate to get gradient and loss as fast as possible
    fg! = let f=fwd
        function fg!(F, G, x) 
            # Zygote calculates both derivative and loss, therefore do everything in one step
            if G !== nothing
                y, back = Zygote.withgradient(l, x)
                # calculate gradient
                G .= back[1]
                if F !== nothing
                    return y
                end
            end
            if F !== nothing
                return l(x)
            end
        end
    end
    
    return fg!
end

# ╔═╡ c70e0adf-4ac3-41e3-bdd5-d97eaa69342a
fg! = make_fg!(fwd)

# ╔═╡ f4ed9b62-1685-42e9-bb8e-0d0b80a74baa
res = Optim.optimize(Optim.only_fg!(fg!), init0, LBFGS(), 
	Optim.Options(iterations=10, store_trace=true, allow_f_increases=true, g_abstol=1f-20)
)

# ╔═╡ 8a98dbac-b8f2-400a-99fc-0c9ac0685a86
plot([t.value for t in res.trace], ylabel="loss")

# ╔═╡ 40c2a912-f806-4bc0-b19f-833c15abf0bb
[simshow(cis.(map_f.(res.minimizer))) simshow(phase_clipped) simshow(phase)]

# ╔═╡ a378802f-12cf-47b4-8e7c-5596dbe21573
res.minimizer

# ╔═╡ ee91f97a-3a2e-467c-9bdb-64586090422e
I3 = fwd(res.minimizer);

# ╔═╡ 6335090d-9607-4c7f-8666-0ae44a4add1d
sum(I3[:, :, end] .* rr_arr)

# ╔═╡ 8168381d-8551-48dd-b50f-951713fb19e0
[simshow(I3) simshow(I2) simshow(I1)]

# ╔═╡ 3a1caaea-c0fa-472d-8dce-574c4b4fc6c4
begin
	plot(I1[(begin+end)÷2+1, :, end].^0.3 |> normf, label="ground truth")
	#plot!(I1[(begin+end)÷2+1, :, end-3].^1)
	plot!(I2[(begin+end)÷2+1, :].^0.3 |> normf, label="clipped phase")
	plot!(I3[(begin+end)÷2+1,:, end].^0.3 |> normf, label="optimized")
end

# ╔═╡ 60ac719c-15eb-4125-a632-8d564e2a01b7
begin
	plot(abs.(ft(I1[(begin+end)÷2+1, :, end])) |> normf)
	plot!(abs.(ft(I2[(begin+end)÷2+1, :])) |> normf)
	plot!(abs.(ft(I3[(begin+end)÷2+1, :, end])) |> normf)
end

# ╔═╡ 9b167483-6649-4223-b67f-5aca801a896a
f3(I1)

# ╔═╡ d0434fb2-d621-46d6-906a-26d80dd950b6
f3(I2)

# ╔═╡ ed9cc155-b82e-4976-b0ef-b52a29fe6f3f
f3(I3)

# ╔═╡ ef342a2e-3370-4764-8b05-e0b38ebf145b
sum(I1 .* rr_arr) / sum(I2)

# ╔═╡ cf78c850-52d0-4001-ad06-201ded97fa72
sum(I2 .* rr_arr) / sum(I2)

# ╔═╡ 3c8a3065-4b17-47f4-9048-dc01fce3e5db
sum(I3 .* rr_arr) / sum(I3)

# ╔═╡ Cell order:
# ╠═fe97a788-9f64-4099-9740-bf8179b0847b
# ╠═910abe5b-1795-486b-90b0-2d7daabe5240
# ╠═4969f583-5bb4-4023-87b6-50a93706cec9
# ╠═82772fcc-ea3e-4a3b-a2df-207c8e036ed4
# ╠═2e69f067-2ae9-4591-a28f-360a4ff0ba2e
# ╠═7a9ee3d2-54bd-40ed-bf2f-f176fb73026a
# ╟─58db1105-99a8-4c0d-854b-2b097bde8fca
# ╠═3d6550a9-502d-4b3f-b7cb-56c41a4d966e
# ╟─8e66d7cb-13be-4b11-ad3d-fbbd087c0139
# ╠═af609bd1-c7e4-4310-af31-c2511726d0e1
# ╠═449c9203-d1ce-4914-8b55-9d60c464636f
# ╠═bfdd9979-d0e6-4094-8808-a5dc3bdb5b60
# ╠═bea6cd49-83e4-4620-929c-4893d2baf502
# ╠═59a68666-0ce5-4f62-86c6-17d2f84954d9
# ╠═40d10497-ce23-4cbf-b453-db0b82788c87
# ╠═48ebe008-e49f-45d6-9600-6ec0ecb9252d
# ╟─f4bcde25-6130-468f-8564-174bcca22297
# ╠═66a3b4e3-5b10-496f-9e8d-546359090622
# ╠═b57e0b2e-eaad-4b94-9b0b-8f8ca9318ce6
# ╟─b2f503c9-42a9-4caa-bf96-1fbe0532ce21
# ╠═e772977f-bc47-412c-8038-001af4d5ef01
# ╠═d9fa0122-e0cd-4d59-8c94-e29a1edb75a9
# ╟─08c2453b-eea8-411d-8611-93bd8714a493
# ╠═ea5bc70f-952e-4cfd-b406-6a0f67e2f1cc
# ╠═e375da67-e028-4525-8360-a7c2d9d34c1a
# ╠═af27ad72-79fc-4ec5-b4d7-e1a354cd644a
# ╠═c337b832-46e5-4333-9ec7-009bb59cbfc9
# ╠═a6edb05d-ae9c-4e21-84aa-f63d3ae35e74
# ╠═c25e5f7f-24fa-4617-ba6d-3275cae5631a
# ╠═7ae7a052-d991-47ab-872a-3b72c132d524
# ╠═f0087e5f-8c51-40de-8708-4218216b36b7
# ╠═c4c37737-927d-464a-868c-b53e7c1fa3b6
# ╠═26efcaba-a9d6-497a-854f-addb82b9988c
# ╠═d0d98b0b-b262-43c6-8f53-18c73e12910e
# ╠═c9241344-565c-4e1e-85f4-0659ff465423
# ╠═48c63470-d520-432a-9b69-0ae6d598879d
# ╠═2834ecb4-4abd-4d4c-991e-8eee657e658d
# ╠═e65744b4-07c5-4483-871f-1b0d49eed626
# ╟─0411b34b-99b2-4142-8992-ef838dee2ad1
# ╠═cdb6c70d-23ae-400c-a404-89887fd33e48
# ╠═d14d4d64-1661-4896-9c43-57ebada7c859
# ╠═85342824-d0db-43f5-9ba3-f25e7fdd72d9
# ╟─83802d1f-ed4a-47fb-864c-07ce623ba180
# ╠═0cb2dda9-1e11-4c9f-893c-a80d744159a6
# ╠═4b52f53c-3f41-4265-bd67-b5ef37c3629b
# ╠═b9ceab04-ca9b-4283-8619-18dffd0a7114
# ╠═7c0e2c54-d187-40fd-a8df-922521c7bac2
# ╟─1fb087b1-4abc-44d3-b1ab-e9588c40d6a5
# ╠═23236a3b-ba23-41e0-84a4-058425d284cc
# ╠═f85dcffc-502e-40b3-aaf8-b3cdf1473bd7
# ╠═1b8de892-f564-4b06-a1c2-ade26562dfae
# ╠═c70e0adf-4ac3-41e3-bdd5-d97eaa69342a
# ╠═f4ed9b62-1685-42e9-bb8e-0d0b80a74baa
# ╠═827c9db5-91cd-4b4e-87b0-4b01cef6639f
# ╠═6aa542f9-e96c-4be3-b101-b52b77040f8d
# ╠═6335090d-9607-4c7f-8666-0ae44a4add1d
# ╠═8a98dbac-b8f2-400a-99fc-0c9ac0685a86
# ╠═2c0d8704-08fe-4074-87b9-00a392cae2a8
# ╠═40c2a912-f806-4bc0-b19f-833c15abf0bb
# ╠═8168381d-8551-48dd-b50f-951713fb19e0
# ╠═a378802f-12cf-47b4-8e7c-5596dbe21573
# ╠═ee91f97a-3a2e-467c-9bdb-64586090422e
# ╟─65357531-77e3-4c76-a5de-1c0c36d3cc6c
# ╠═3a1caaea-c0fa-472d-8dce-574c4b4fc6c4
# ╠═60ac719c-15eb-4125-a632-8d564e2a01b7
# ╠═65e9381d-f14d-4184-89c1-006e1578f040
# ╠═27804c15-7c85-4305-9908-c23d69c8eded
# ╠═def82b67-9c0e-4f8e-8fdb-9260f4151fbd
# ╠═19d3ff40-4826-4935-92ca-004f29791f7e
# ╠═d38c7f83-b7e8-453a-8149-5d751ef01081
# ╠═aa62237c-3eb4-46d5-ab6f-b6d9dd89de6a
# ╠═cc8e5830-449e-4fd3-bfc6-09bb19b56ac7
# ╠═9b167483-6649-4223-b67f-5aca801a896a
# ╠═d0434fb2-d621-46d6-906a-26d80dd950b6
# ╠═ed9cc155-b82e-4976-b0ef-b52a29fe6f3f
# ╠═ef342a2e-3370-4764-8b05-e0b38ebf145b
# ╠═cf78c850-52d0-4001-ad06-201ded97fa72
# ╠═3c8a3065-4b17-47f4-9048-dc01fce3e5db
