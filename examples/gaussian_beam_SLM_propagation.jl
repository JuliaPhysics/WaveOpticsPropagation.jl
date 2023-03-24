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

# ╔═╡ 97891d2b-facd-4f51-9774-5afde6a874e9
begin
	using Pkg, Revise
	Pkg.activate(".")
end

# ╔═╡ 034e96b2-c88a-11ed-35f3-9fbfcefa62df
using WaveOpticsPropagation, Napari, ImageShow, FFTW, CUDA, FourierTools, NDTools, Plots, Colors

# ╔═╡ 8082f7f3-3f47-4479-b3fa-20b2eb43ee78
using LinearAlgebra

# ╔═╡ b53bc181-42e6-410f-99fb-6fa28b9c4f09
using IndexFunArrays

# ╔═╡ f6d1a0dd-431f-4cbd-ae3c-3f031dc2beb5
using PlutoUI

# ╔═╡ febd1f55-b327-4e31-975b-afb635a82376
using TestImages

# ╔═╡ dfc85871-6697-44e1-a28a-86e7314a7d29
FFTW.set_num_threads(4)

# ╔═╡ 6e4cb613-6cec-4eb0-93a9-bc93b83bf755
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	    
	togoc(x) = use_CUDA[] ? CuArray(x) : x
	toc(x) = x
	toc(x::CuArray) = Array(x)
	toc(x::LinearAlgebra.Adjoint{T, <:CuArray}) where T = Array(x)

end

# ╔═╡ 3dd7717c-0cbb-45f9-b03b-9316d37fd635
ImageShow.simshow(x::CuArray; kwargs...) = simshow(toc(x); kwargs...)

# ╔═╡ 15f89bfc-34c3-408e-9577-b79e88aea179


# ╔═╡ 7e73e1e3-e881-4025-b88a-bb7b12f85e02
N = 2000

# ╔═╡ e362673e-dbe9-4ad6-ba7a-d1f22da3f191
field = togoc(zeros(ComplexF32, (N, N)))

# ╔═╡ d6e11fcc-18ef-4c64-82b8-20abbb26d339
function dmd_pattern()
	dmd = togoc(zeros(ComplexF32, 768, 1024));

	dmd[300:400, 300:400] .= 1
	dmd[390:410, 400:500] .= 1

	dmd[450:451, 450:451] .= 1
	dmd[7 .+ (450:451), 450:451] .= 1

	return dmd
end

# ╔═╡ f0cd85d6-b8a4-4be5-8e99-0eebadc0d27b
simshow(dmd_pattern())

# ╔═╡ 5b586b1e-e785-4980-815f-0534f997ea41
λ = 405f-9 / 1.5

# ╔═╡ 3ee91f4c-83a5-4e33-948f-88b36cf1828c
L = 14.0f-3 * 2000 / 1024 * 2

# ╔═╡ 25916008-4a44-4140-9558-6f055d01b101
L / 2000

# ╔═╡ 49f045dd-b764-4e37-abc4-271a87db4677
y = togoc(fftpos(L, N, CenterFT))

# ╔═╡ 2c301c7f-bc79-4141-8ae1-3022bfc54c66
x = y';

# ╔═╡ b083214a-645f-499f-8aaa-e5511cbe64d8
L ./ size(dmd)

# ╔═╡ a11d219e-f6d2-4137-b0b7-be2ffa589893
field_d = set_center!(copy(field), dmd_pattern()) .* cispi.(togoc(0.0000f0 .* rr2(field)));

# ╔═╡ 5fc8ecab-6a7c-467f-a67d-d4ba7cd0b2e2
simshow(field_d)

# ╔═╡ 641e4caa-3d4a-46b1-af8f-a6ef8d246723
@mytime field_p = angular_spectrum(field_d, 16f-3, λ, L)[1]

# ╔═╡ b6db3c81-e960-4a29-b955-43721b3af769
@bind iz PlutoUI.Slider(1:2, show_value=true)

# ╔═╡ 3df00e4a-89f1-4ef3-a101-e02113fa471d
simshow(abs2.([toc(field_d);;; toc(field_p)][:, :, iz]))

# ╔═╡ 6f31315a-f545-4989-91c9-b58db0e4ba18
@view_image abs2.([toc(field_d);;; toc(field_p)])

# ╔═╡ 8e4c91eb-ed58-4d1f-9a43-48d2c262e4ea
begin
	plot(abs.(resample(toc(field_d)[1000:1100, 938], (3000,))))
	plot!(abs2.(resample(toc(field_p)[1000:1100, 938], (3000,))))
end;

# ╔═╡ 661541a4-84f4-4f74-bbe7-3aab62fb52ce
begin
	plot(abs2.(toc(field_d)[1000:1100, 938]))
	plot!(abs2.(toc(field_p)[1000:1100, 938]))
end

# ╔═╡ 3b90e7f5-daef-4181-81b7-db3d0a9a922b
z = range(100f-6, 10f-3, 5)

# ╔═╡ 09ca6d0b-fbe0-4ff4-a900-2dd2179ab6bd
L / 1024

# ╔═╡ 8397198e-3b49-49db-af87-a932d4967a63
heatmap(toc(x), toc(y), Float32.(simshow(abs2.(toc(field_p)), γ=0.4)))

# ╔═╡ 3ba2a0a2-6535-464e-ae83-cddd68bff413
toc(x)

# ╔═╡ 0f232dcc-faf8-46d6-87bb-4fe318964968
toc(x)

# ╔═╡ 57cfdcf2-99c0-4db2-a740-ed74994116c9
typeof(x)

# ╔═╡ Cell order:
# ╠═97891d2b-facd-4f51-9774-5afde6a874e9
# ╠═034e96b2-c88a-11ed-35f3-9fbfcefa62df
# ╠═8082f7f3-3f47-4479-b3fa-20b2eb43ee78
# ╠═b53bc181-42e6-410f-99fb-6fa28b9c4f09
# ╠═f6d1a0dd-431f-4cbd-ae3c-3f031dc2beb5
# ╠═febd1f55-b327-4e31-975b-afb635a82376
# ╠═3dd7717c-0cbb-45f9-b03b-9316d37fd635
# ╠═dfc85871-6697-44e1-a28a-86e7314a7d29
# ╠═6e4cb613-6cec-4eb0-93a9-bc93b83bf755
# ╠═15f89bfc-34c3-408e-9577-b79e88aea179
# ╠═7e73e1e3-e881-4025-b88a-bb7b12f85e02
# ╠═e362673e-dbe9-4ad6-ba7a-d1f22da3f191
# ╠═d6e11fcc-18ef-4c64-82b8-20abbb26d339
# ╠═f0cd85d6-b8a4-4be5-8e99-0eebadc0d27b
# ╠═5ed910a9-b79f-46a4-a78f-fce572a08fe2
# ╠═5b586b1e-e785-4980-815f-0534f997ea41
# ╠═25916008-4a44-4140-9558-6f055d01b101
# ╠═3ee91f4c-83a5-4e33-948f-88b36cf1828c
# ╠═49f045dd-b764-4e37-abc4-271a87db4677
# ╠═2c301c7f-bc79-4141-8ae1-3022bfc54c66
# ╠═b083214a-645f-499f-8aaa-e5511cbe64d8
# ╠═a11d219e-f6d2-4137-b0b7-be2ffa589893
# ╠═5fc8ecab-6a7c-467f-a67d-d4ba7cd0b2e2
# ╠═641e4caa-3d4a-46b1-af8f-a6ef8d246723
# ╠═b6db3c81-e960-4a29-b955-43721b3af769
# ╠═3df00e4a-89f1-4ef3-a101-e02113fa471d
# ╠═b7684276-2099-417d-aa57-573d98c61f2d
# ╠═6f31315a-f545-4989-91c9-b58db0e4ba18
# ╠═8e4c91eb-ed58-4d1f-9a43-48d2c262e4ea
# ╠═661541a4-84f4-4f74-bbe7-3aab62fb52ce
# ╠═2341052a-c5fc-4ef1-9623-138c2781fde8
# ╠═3b90e7f5-daef-4181-81b7-db3d0a9a922b
# ╠═09ca6d0b-fbe0-4ff4-a900-2dd2179ab6bd
# ╠═8397198e-3b49-49db-af87-a932d4967a63
# ╠═3ba2a0a2-6535-464e-ae83-cddd68bff413
# ╠═0f232dcc-faf8-46d6-87bb-4fe318964968
# ╠═57cfdcf2-99c0-4db2-a740-ed74994116c9
