### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 9902bc0e-c8dc-11ed-05fe-a9f16c85ba7b
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using Pkg, Revise
	Pkg.activate("../examples/.")
end
  ╠═╡ =#

# ╔═╡ 748c98db-f175-4b8d-87d4-eae2ec6f26ab
# ╠═╡ skip_as_script = true
#=╠═╡
using Plots, ImageShow, PlutoUI
  ╠═╡ =#

# ╔═╡ 38f7c686-48c2-4051-9e2e-8f2aad6e95ee
using Test, WaveOpticsPropagation, IndexFunArrays, NDTools, FourierTools 

# ╔═╡ 0299632f-e805-4e27-82e8-d67a756409f2


# ╔═╡ 96986249-3edb-4fae-b4a5-ec7ae18fdb59
function test_gauss_consistency(λ, L, N, z, z_init, w_0; do_test=true)
	y = reshape(Float32.(fftpos(L[1], N[1], CenterFT)) ,:, 1)
	x = reshape(Float32.(fftpos(L[2], N[2], CenterFT)), 1, :)
	field = gauss_beam.(y, x, z_init, λ, w_0);
	field_as = angular_spectrum(field, z, λ, L)[1];
	field_z = gauss_beam.(y, x, z_init + z, λ, w_0);


	if do_test
		@test ≈(field_as .+ maximum(abs.(field_as)) ,  maximum(abs.(field_as)) .+ field_z, 
		rtol=0.004)
		@test all(≈(field_as .+ maximum(abs.(field_as)) ,  maximum(abs.(field_as)) .+ field_z, 
		rtol=0.1))
		@test sum(abs2.(field_z)) ≈ sum(abs2.(field_as)) ≈ sum(abs2.(field))
	end
		
	return field, field_as, field_z
end

# ╔═╡ c732c1d7-6ae8-41ab-ae2e-d635fa9d66a3
res1 = test_gauss_consistency(405f-9, (10f-4, 20.0f-4), (512, 511), 1f-2, 1f-10, 0.01f-3)

# ╔═╡ a5e7878f-4ea1-4f75-87f8-fafd09f5ad82
res2 = test_gauss_consistency(10f-9, (1f-4, 0.5f-4), (400, 401), 0.02f-2, 0.05f-2, 0.003f-3, do_test = true)

# ╔═╡ 22e070fd-b27f-4cb2-8ad7-71f409e881bf
# ╠═╡ skip_as_script = true
#=╠═╡
simshow([res1[2] res1[3]], γ=0.1)
  ╠═╡ =#

# ╔═╡ 1f026add-c513-42fa-a26c-5da40badf8a8
# ╠═╡ skip_as_script = true
#=╠═╡
simshow([res2[1] res2[2] res2[3]], γ=0.2)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═9902bc0e-c8dc-11ed-05fe-a9f16c85ba7b
# ╠═748c98db-f175-4b8d-87d4-eae2ec6f26ab
# ╠═38f7c686-48c2-4051-9e2e-8f2aad6e95ee
# ╠═0299632f-e805-4e27-82e8-d67a756409f2
# ╠═96986249-3edb-4fae-b4a5-ec7ae18fdb59
# ╠═c732c1d7-6ae8-41ab-ae2e-d635fa9d66a3
# ╠═a5e7878f-4ea1-4f75-87f8-fafd09f5ad82
# ╠═22e070fd-b27f-4cb2-8ad7-71f409e881bf
# ╠═1f026add-c513-42fa-a26c-5da40badf8a8
