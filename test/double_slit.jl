### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# ╔═╡ 9745c33c-c8d4-11ed-216d-cb75e0aaec28
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using Pkg, Revise
	Pkg.activate("../examples/.")
end
  ╠═╡ =#

# ╔═╡ 4acbbe98-f033-42c5-b6d0-60a5ddf2570e
using Test, WaveOpticsPropagation, IndexFunArrays, NDTools, FourierTools

# ╔═╡ 2b5650e9-db71-459d-ae13-7b52ca8d2a85
# ╠═╡ skip_as_script = true
#=╠═╡
using Plots, ImageShow
  ╠═╡ =#

# ╔═╡ 83144a55-ac66-4b92-b9c7-36daacbab791
"""
	double_slit

`d` is distance between slits in pixel
`b` is the width of the slits in pixel
"""
function double_slit(N, d, b, offset, z, λ, L)

	slit_init = (box(ComplexF32, (N, N), (N, b), offset=(11, offset + N÷2 + 1 - d ÷2 )) .+  
					   box(ComplexF32, (N, N), (N, b), offset=(11, offset + N÷2 + 1 + d ÷2 )))

	xpos = fftpos(L, N, NDTools.CenterFT) .- offset .* L ./ N
	sinθ = sin.(atan.(xpos, z))
	d_m = d * L / N
	b_m = b * L / N
	slit_ana = cos.(π .* d_m .* sinθ ./ λ).^2 .* sinc.(b_m .* sinθ ./ λ).^2

	slit_prop = (angular_spectrum(slit_init, z, λ, L))
	slit_prop2 = (AngularSpectrum(slit_init, z, λ, L)(slit_init))
	@test slit_prop2 ≈ slit_prop
	
	slit_prop = abs2.(slit_prop)
	slit_prop2 = abs2.(slit_prop2)
	
	slit_prop ./= maximum(slit_prop[11, :]) 
	slit_prop2 ./= maximum(slit_prop2[11, :]) 

	@test all(.≈(0.01f0 .+ slit_ana, 0.01f0 .+ slit_prop[11, :], rtol=0.3))
	@test findmax(slit_ana) == findmax(slit_prop[11, :])

	return slit_init, slit_prop, slit_prop2, slit_ana
end

# ╔═╡ c12b2aac-5e48-47b2-a0f8-e626a0020b20
slit_init, slit_prop, slit_prop2, slit_ana = double_slit(100, 10, 3, 0, 25e-3, 1000e-9, 1e-3)

# ╔═╡ 33a6b4e4-ee04-45fe-96b4-6fa8614b5c6c
#=╠═╡
begin
	#plot(abs2.(slit_init[11, :]))
	plot(slit_prop[11, :])
	plot!(slit_ana)
end
  ╠═╡ =#

# ╔═╡ 6b2554ac-613a-4b06-b9a5-fb96a81816ee
double_slit(512, 10, 3, -30, 5e-3, 632e-9, 1e-3)

# ╔═╡ 05587986-fe48-4ac8-91eb-5fedfbf6634e
double_slit(512, 10, 3, 30, 5e-3, 1000e-9, 1e-3)

# ╔═╡ 8c9133cd-ae62-4a3d-81b8-4e30c795a04f
 double_slit(100, 10, 3, 0, 25e-3, 1000e-9, 1e-3)

# ╔═╡ Cell order:
# ╠═9745c33c-c8d4-11ed-216d-cb75e0aaec28
# ╠═4acbbe98-f033-42c5-b6d0-60a5ddf2570e
# ╠═2b5650e9-db71-459d-ae13-7b52ca8d2a85
# ╠═83144a55-ac66-4b92-b9c7-36daacbab791
# ╠═c12b2aac-5e48-47b2-a0f8-e626a0020b20
# ╠═33a6b4e4-ee04-45fe-96b4-6fa8614b5c6c
# ╠═6b2554ac-613a-4b06-b9a5-fb96a81816ee
# ╠═05587986-fe48-4ac8-91eb-5fedfbf6634e
# ╠═8c9133cd-ae62-4a3d-81b8-4e30c795a04f
