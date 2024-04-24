### A Pluto.jl notebook ###
# v0.19.41

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

# ╔═╡ d551f366-0213-11ef-176b-7fcf5b47315e
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 9e6d909f-06de-46e7-929a-61bfd67bea26
using Zygote, WaveOpticsPropagation, FFTW, ImageShow, FiniteDifferences, CUDA, Optim, IndexFunArrays, FourierTools, TestImages

# ╔═╡ ef79854f-0ae9-41aa-8f92-aae8d7023d71
using Plots

# ╔═╡ f9a13d72-f904-467c-8346-a9ef272693ff
using PlutoUI

# ╔═╡ 20c5faa9-5694-42fd-b1bb-0c5093f6c93c
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	    
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ b8cf117e-d084-4d42-8022-03fd6ac4c417
TableOfContents()

# ╔═╡ 94ea2891-db37-477e-b535-1a6d402d4af2


# ╔═╡ a971769f-1215-4422-a209-fad710be16dd
sz = (1024, 1024)

# ╔═╡ bdf3b801-28e0-41cf-9dc4-92fe8fdaca0e
beam = togoc(0im .+ gaussian(Float32, sz, sigma=100)) .* togoc(cispi.(conv((normal(Float32, (sz), sigma=5)), 3 .* rand(Float32, (sz)))));

# ╔═╡ 36b798c9-feee-43b7-b9ec-4f0558a44d2c
simshow(Array(beam))

# ╔═╡ ad4ef631-841d-412e-a09a-4db60fa7e4a9
z = 100f-3

# ╔═╡ 15011619-f26a-4858-b0a0-d5dbcad49824
λ = 633f-9

# ╔═╡ 48048a82-5bbf-4f16-a7d9-15993c83f905
L = 10f-3

# ╔═╡ 3f791571-2a36-4479-a853-c6ed4e8a270d
md"# Forward model"

# ╔═╡ e52cfcd8-eec8-4a8b-b953-f1c6ef2be959
N = 3

# ╔═╡ 42c00ce2-d1e4-4f84-84be-a7e4cd6551ce
begin
	diffuser = cispi.(conv((normal(Float32, (sz), sigma=1)), 10 .* rand(Float32, (sz..., N))))
	diffuser_c = togoc(diffuser)# .* box(sz, (120, 150))

	diffuser_c[:, :, 1] .= cis.(0.7 .* rr2(sz, scale=0.05)) .+ 0.1f0 .* diffuser_c[:, :, 1]
	diffuser_c[:, :, 2] .= cis.(0.7 .*rr2(sz, scale=0.05)) .+ 0.1f0 .* diffuser_c[:, :, 2]
	diffuser_c[:, :, 3] .=  cis.(0.7 .*rr2(sz, scale=0.05)) .+ 0.1f0 .* diffuser_c[:, :, 3]

	#diffuser_c[:, :, 3] .= 1

#	diffuser_c[:, :, 4] .*= cis.(0.3 .* xx((sz)))
#	diffuser_c[:, :, 5] .*= cis.(.- 0.3 .* xx((sz)))
#	diffuser_c[:, :, 6] .*= cis.(0.3 .* yy((sz)))
#	diffuser_c[:, :, 7] .*= cis.(.- 0.3 .* yy((sz)))
end

# ╔═╡ d058c9ba-82ec-4f1e-8938-eb9e1e95b713
fraunhofer = Fraunhofer(beam .* diffuser_c, z, λ, L)

# ╔═╡ 5d21e0bc-ea6e-4b2c-8977-d1761d1b3c65
fraunhofer(beam .* diffuser_c) ./ 0.00195312 ≈ fftshift(fft(ifftshift(beam .* diffuser_c, (1,2)), (1,2)), (1,2)) 

# ╔═╡ df72f725-8464-46b0-a291-8503ea4e563f
size(diffuser_c)

# ╔═╡ 90d78713-0d4d-48b2-b590-a79d4e7c12d5
simshow(Array(diffuser_c[:, :, 1]))

# ╔═╡ f9c4d5ca-a35a-462a-b5f4-4120012f5a1b
fwd(x) = begin
	return abs2.(fraunhofer(x .* diffuser_c))
end

# ╔═╡ d84457d3-30ae-4711-b47f-7d2a7376a237
md"# Optimizer"

# ╔═╡ d219be8b-2d22-4fbe-b022-217edfeec5dc
function make_fg!(fwd, measured, loss=:L2)
    L2 = let measured=measured, fwd=fwd
        function L2(x)
            return sum(abs2, fwd(x) .- measured)
		end
    end

    f = let 
        if loss == :L2
            L2
        end
    end

    g! = let f=f
        function g!(G, rec)
            if !isnothing(G)
                return G .= Zygote.gradient(f, rec)[1]
            end
        end
    end
    return f, g!
end

# ╔═╡ 692e0263-8273-45a6-b3e6-c30bbeaeb649
measurement_c = fwd(beam);

# ╔═╡ 5c550775-a614-44dc-a81d-638448d23067
@bind iz Slider(1:N)

# ╔═╡ 56a03baf-3873-43d5-b8c1-8569de6026b0
simshow(Array(measurement_c)[:, :, iz], γ=1)

# ╔═╡ fc9bb298-5b46-4585-9ea5-4df59405d18b
fraunhofer.params.Lp

# ╔═╡ de16936c-7774-4aab-9702-f0cf4fa660fd
md"# Optimize"

# ╔═╡ ee5260f3-d3a4-4bf2-8b54-31723c3c516f
rec0 = togoc(ones(ComplexF32, sz));

# ╔═╡ d90bff7e-150d-407d-a510-385fd5a32df3
f, g! = make_fg!(fwd, measurement_c)

# ╔═╡ eadee3cb-1302-4f1a-a21c-17265d9a2e6b


# ╔═╡ 05b63595-61dd-416f-8610-0f6ee1919a6f
@mytime res = Optim.optimize(f, g!, rec0, LBFGS(),
                                 Optim.Options(iterations = 80,  
                                               store_trace=true))

# ╔═╡ 9ed4c43f-1ad5-43b1-bef6-40faf027207d
plot([t.iteration for t in res.trace], [t.value for t in res.trace], label="LBFGS", yaxis=:log)

# ╔═╡ 5372b296-75d6-497d-b6c9-e2f104e76522
[simshow(Array(res.minimizer) .* cispi(1.5), γ=1) simshow(Array(Array(beam)), γ=1)]

# ╔═╡ 82273df3-554e-43f9-a881-63609c239648
cat(simshow(Array(res.minimizer) .* cispi(1.5), γ=1), simshow(Array(Array(beam)), γ=1), dims=3)[:, :, 1]

# ╔═╡ 02feb5a2-d43d-4191-b9cd-8d3946c05332
simshow(Array(beam))

# ╔═╡ 980e4c15-7a04-4aad-89f1-119c33693288


# ╔═╡ Cell order:
# ╠═d551f366-0213-11ef-176b-7fcf5b47315e
# ╠═9e6d909f-06de-46e7-929a-61bfd67bea26
# ╠═20c5faa9-5694-42fd-b1bb-0c5093f6c93c
# ╠═b8cf117e-d084-4d42-8022-03fd6ac4c417
# ╠═ef79854f-0ae9-41aa-8f92-aae8d7023d71
# ╠═f9a13d72-f904-467c-8346-a9ef272693ff
# ╠═94ea2891-db37-477e-b535-1a6d402d4af2
# ╠═a971769f-1215-4422-a209-fad710be16dd
# ╠═bdf3b801-28e0-41cf-9dc4-92fe8fdaca0e
# ╠═36b798c9-feee-43b7-b9ec-4f0558a44d2c
# ╠═ad4ef631-841d-412e-a09a-4db60fa7e4a9
# ╠═15011619-f26a-4858-b0a0-d5dbcad49824
# ╠═48048a82-5bbf-4f16-a7d9-15993c83f905
# ╠═d058c9ba-82ec-4f1e-8938-eb9e1e95b713
# ╠═5d21e0bc-ea6e-4b2c-8977-d1761d1b3c65
# ╟─3f791571-2a36-4479-a853-c6ed4e8a270d
# ╠═e52cfcd8-eec8-4a8b-b953-f1c6ef2be959
# ╠═df72f725-8464-46b0-a291-8503ea4e563f
# ╠═42c00ce2-d1e4-4f84-84be-a7e4cd6551ce
# ╠═90d78713-0d4d-48b2-b590-a79d4e7c12d5
# ╠═f9c4d5ca-a35a-462a-b5f4-4120012f5a1b
# ╠═d84457d3-30ae-4711-b47f-7d2a7376a237
# ╠═d219be8b-2d22-4fbe-b022-217edfeec5dc
# ╠═692e0263-8273-45a6-b3e6-c30bbeaeb649
# ╠═5c550775-a614-44dc-a81d-638448d23067
# ╠═56a03baf-3873-43d5-b8c1-8569de6026b0
# ╠═fc9bb298-5b46-4585-9ea5-4df59405d18b
# ╠═de16936c-7774-4aab-9702-f0cf4fa660fd
# ╠═ee5260f3-d3a4-4bf2-8b54-31723c3c516f
# ╠═d90bff7e-150d-407d-a510-385fd5a32df3
# ╠═eadee3cb-1302-4f1a-a21c-17265d9a2e6b
# ╠═05b63595-61dd-416f-8610-0f6ee1919a6f
# ╠═9ed4c43f-1ad5-43b1-bef6-40faf027207d
# ╠═5372b296-75d6-497d-b6c9-e2f104e76522
# ╠═82273df3-554e-43f9-a881-63609c239648
# ╠═02feb5a2-d43d-4191-b9cd-8d3946c05332
# ╠═980e4c15-7a04-4aad-89f1-119c33693288
