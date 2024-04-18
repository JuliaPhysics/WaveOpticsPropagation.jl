### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ db7305de-e790-11ee-0a0b-9123095cc122
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 4a13d17e-3646-45e4-b25b-a6522130b619
using WaveOpticsPropagation

# ╔═╡ 45d6e0b9-2dc9-459c-967c-c91c5ad254d5
using FourierTools

# ╔═╡ 244917c8-3ddb-4d14-a81c-2d998443329a
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Plots, NDTools, FFTW

# ╔═╡ 06d4e42e-4152-4fb0-9807-6bcdc36c9b77
md"# Define field"

# ╔═╡ 08fee313-2b3e-4e9c-a158-62d96a43f4fb
sz = (128, 128)

# ╔═╡ 5da908cc-68ef-4a56-a764-a45e2aff6bfb
field = box(sz) .+ 0im;

# ╔═╡ 1f14927a-4892-42ca-81cf-880af671b8cf
λ = 600e-9

# ╔═╡ 757059e4-3fe1-4543-99ea-46ddd5ffe983
dx = λ / 2

# ╔═╡ e9fdf460-f5cc-49b2-a0fc-f2fe69c80a6d
L = sz[1] * dx

# ╔═╡ c7b682a5-4fd2-4431-9262-ffff0e3c8f6f
z = range(0, (2 * sz[1] - 1) * dx, 2 * sz[1])

# ╔═╡ ce5b7eb3-443d-47b7-9988-11d6f67fd243
dx

# ╔═╡ 2aca36cf-e6eb-4c33-a641-004e3fe03025
z.step

# ╔═╡ 0b47b5d3-3bac-413d-9c70-fc36e9a239fa
AS = AngularSpectrum(field, z, λ, L)

# ╔═╡ 665c64cb-3266-4225-9a37-321d4e966fd5
AS2 = AngularSpectrum(field, dx, λ, L, bandlimit_border=(0.999, 1))

# ╔═╡ 7ecedc34-971c-413f-ba2f-136b28c9a196
@time out = AS(field);

# ╔═╡ dfbd89bd-eaca-44aa-a40d-0bf0516a48a3
simshow(AS(field, crop=false)[:, :, 50], γ=0.1)

# ╔═╡ a5194aca-751e-4baf-b529-c7f4e8135434


# ╔═╡ 981dd9c3-f10b-416e-b1c8-2aeb998ea5de
@bind iz Slider(1:size(out, 3))

# ╔═╡ b4af444f-8504-46c2-a3f8-be93bf5652d0
[simshow(AS.HW[:, :, iz]) simshow(AS2.HW[:, :])  simshow(ffts(AS2.HW[:, :]), γ=0.1)]

# ╔═╡ f219a850-f066-49d4-b4f9-d391797022af
simshow(fft(ifftshift(fftshift(ifft(AS.HW[:, :, 1])) .* (rr(size(AS.HW)[1:2]) .< 50))))

# ╔═╡ 21bc4ed2-5490-485c-b645-c045e643acd1
function multi_slice(field, z, λ, L)
	AS = AngularSpectrum(field, z[2]-z[1], λ, L)

	AS.HW .= fft(ifftshift(fftshift(ifft(AS.HW)) .* (rr(size(AS.HW)) .< 50)))
	out = similar(field, (size(field)..., length(z)))

	f = field
	for i in 1:length(z)
		out[:,:, i] .= f
		f = AS(f)
	end
	return out
end

# ╔═╡ eb36720f-7a6e-45e1-bae3-1de6abfbf24a
@time out2 = multi_slice(field, z, λ, L);

# ╔═╡ 6ec49c02-8af4-4c58-b18f-66fa3361d1a5
[simshow(abs2.(out[:, :, iz]), γ=0.2) simshow(abs2.(out2[:, :, iz]), γ=0.2)]

# ╔═╡ Cell order:
# ╠═db7305de-e790-11ee-0a0b-9123095cc122
# ╠═4a13d17e-3646-45e4-b25b-a6522130b619
# ╠═45d6e0b9-2dc9-459c-967c-c91c5ad254d5
# ╠═244917c8-3ddb-4d14-a81c-2d998443329a
# ╠═06d4e42e-4152-4fb0-9807-6bcdc36c9b77
# ╠═08fee313-2b3e-4e9c-a158-62d96a43f4fb
# ╠═5da908cc-68ef-4a56-a764-a45e2aff6bfb
# ╠═1f14927a-4892-42ca-81cf-880af671b8cf
# ╠═757059e4-3fe1-4543-99ea-46ddd5ffe983
# ╠═e9fdf460-f5cc-49b2-a0fc-f2fe69c80a6d
# ╠═c7b682a5-4fd2-4431-9262-ffff0e3c8f6f
# ╠═ce5b7eb3-443d-47b7-9988-11d6f67fd243
# ╠═2aca36cf-e6eb-4c33-a641-004e3fe03025
# ╠═0b47b5d3-3bac-413d-9c70-fc36e9a239fa
# ╠═665c64cb-3266-4225-9a37-321d4e966fd5
# ╠═7ecedc34-971c-413f-ba2f-136b28c9a196
# ╠═dfbd89bd-eaca-44aa-a40d-0bf0516a48a3
# ╠═b4af444f-8504-46c2-a3f8-be93bf5652d0
# ╠═a5194aca-751e-4baf-b529-c7f4e8135434
# ╠═981dd9c3-f10b-416e-b1c8-2aeb998ea5de
# ╠═6ec49c02-8af4-4c58-b18f-66fa3361d1a5
# ╠═eb36720f-7a6e-45e1-bae3-1de6abfbf24a
# ╠═f219a850-f066-49d4-b4f9-d391797022af
# ╠═21bc4ed2-5490-485c-b645-c045e643acd1
