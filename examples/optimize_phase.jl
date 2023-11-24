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

# ╔═╡ 41c6fe0a-854c-11ee-0c0d-49cafb7df0d1
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 22b6b431-8bc1-4630-9a5e-998874831b84
using Zygote, WaveOpticsPropagation, FFTW, ImageShow, FiniteDifferences, CUDA, Optim, IndexFunArrays, FourierTools, TestImages

# ╔═╡ a715d3ef-b3ef-496f-b50f-7ef688df5efc
using PlutoUI

# ╔═╡ dfacac83-565f-431b-a42c-1072c419a4be
using Plots

# ╔═╡ d18059d4-9ff2-4afa-b40f-5c37db59b57f
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	    
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ a1c76c16-1eff-4cde-adaa-b544f6be45a7
sz = (512, 512)

# ╔═╡ 74c17e61-15d7-48cd-9fce-689d1c5a3148
#field = zeros(ComplexF32, sz);

# ╔═╡ fdd08423-020d-4dbd-9198-b7b03264a169
#gield .= cis.(rr2(sz, scale=0.05)) .* gaussian(sz, sigma=50);

field = (rr2(sz) .< 150 .^2)# .* conv((normal(Float32, (sz), sigma=2)), rand(Float32, sz) .* cispi.(2 .* ( 2 .* rand(Float32, (sz...)))))

# ╔═╡ b2f8e5bf-fc0e-4dbc-9816-3cb0072132c2
begin
	field2 = box(Float32, sz, (100, 100)) + 1im .* box(Float32, sz, (100, 100), offset=(200, 300))
	field2 .= (0.1f0 .+ Float32.(Gray.(testimage("cameraman")))) .* cispi.(Float32.(Gray.(testimage("barbara_gray_512"))))
end

# ╔═╡ c34da860-e131-483f-a2d7-ce2dcb1ba1e6
simshow(field2)

# ╔═╡ e661ac7e-63f9-40d7-ba9f-1784bce61e16
field_c = togoc(cat(field2, field, dims=3));

# ╔═╡ a1ba22f5-8015-4e19-ab9c-bc82a4d38e44
sum(abs2, field_c)

# ╔═╡ 680c0d66-97af-4d21-b90c-97cb6d8dfbd9
simshow(field)

# ╔═╡ e67607af-d100-42f9-a830-4c5889584b3d
#sum(abs2, AS_c2(field_c)[1][:, :, 1])

# ╔═╡ 82a210f8-910b-42b8-bdd2-34b11011d34a
field_c

# ╔═╡ 1beb527b-19f0-4352-839a-9ac7e91a3f15
λ = 633f-9

# ╔═╡ c11c6881-3d50-49f6-8799-8e16f172ac9a
L = 5f-3

# ╔═╡ 3a6eb1f4-0763-4aa6-92c1-0abe491dd558
z = 30f-3

# ╔═╡ 7be3d272-cf76-4950-b80b-f743ea200d4a
AS_c, _ = Angular_Spectrum(field_c[:, :, 2], z, λ, L)

# ╔═╡ f796abe0-00cc-4178-a945-96ae5a245227
simshow(Array(AS_c(field_c[:, :, 2])[1]), γ=0.3)

# ╔═╡ 9b03154d-bf00-4e95-a5a1-5ae5e7cb48a4
simshow(Array(abs2.(AS_c(field_c)[1])))

# ╔═╡ 0180ea0b-87a8-4233-86fe-09308a70738d
N = 10

# ╔═╡ 1418e664-6207-4df4-aa79-f18633b63b2e
begin
	diffuser = conv((normal(Float32, (sz), sigma=1)), cispi.(2 .* rand(Float32, (sz..., N))))
	diffuser_c = togoc(diffuser)# .* box(sz, (120, 150))

	diffuser_c[:, :, 1] .= cis.(rr2(sz, scale=0.05)) .* gaussian(sz, sigma=100);
	diffuser_c[:, :, 2] .= cis.(.-1 .* rr2(sz, scale=0.05)) .* gaussian(sz, sigma=100);
	diffuser_c[:, :, 3] .= 1

	diffuser_c[:, :, 4] .*= cis.(0.3 .* xx((sz)))
	diffuser_c[:, :, 5] .*= cis.(.- 0.3 .* xx((sz)))
	diffuser_c[:, :, 6] .*= cis.(0.3 .* yy((sz)))
	diffuser_c[:, :, 7] .*= cis.(.- 0.3 .* yy((sz)))

end

# ╔═╡ 52f7af37-1a44-4a7c-8a9e-31df9bbcda9a
simshow(Array(diffuser_c[:, :, 5]))

# ╔═╡ b436fdc0-ae8d-4dbe-94e6-8b7c26339a70
#AS_c, _ = Angular_Spectrum(field_c[:, :, 1], z, λ, L)

# ╔═╡ c97f6332-82d9-4441-80ad-8a49c7739119
AS_c2, _ = Angular_Spectrum(togoc(zeros(ComplexF32, (sz..., N))), togoc(repeat([z], N)), λ, L)

# ╔═╡ 0bd359cc-0bd4-4748-8905-361969811248
p = plan_fft(diffuser_c, (1,2))

# ╔═╡ 23e64d50-8a6c-418f-86eb-1b4f9917fb32
fwd(x) = begin
	#first = AS_c(x)[1]
	first = x[:, :, 2]
	second = first .* diffuser_c
	third = x[:, :, 1] .* AS_c2(second)[1]
	#fourth = abs2.(fftshift(p * ifftshift(third, (1,2)), (1,2))) ./ size(x, 1).^2
	fourth = abs2.(AS_c2(third)[1])
	return fourth#, second, third
end

# ╔═╡ a5593e55-5b77-4c99-b9a9-5e060748d6af
simshow(Array(fwd(field_c)[2][:, :, 8]))

# ╔═╡ d00e3c52-a25d-4f13-9f76-f9fb0e255737
simshow(Array(AS_c(fwd(field_c)[1][:, :, 8])[1]))

# ╔═╡ 86c47180-3afe-4545-912e-12911155a02a
simshow(Array(diffuser_c)[:, :, 8])

# ╔═╡ 2c6ec01e-df45-4482-9d52-ce04ed343c5d
simshow(Array(AS_c2(diffuser_c)[1][:, :, 8]))

# ╔═╡ df218193-d0f0-4c85-9439-9b70e2e55cf8
simshow(Array(diffuser_c)[:, :, 9])

# ╔═╡ 2531c673-ab41-4374-85fe-e4e256f05c98
field_c |> size

# ╔═╡ 8516d9be-440c-45c5-8f20-cce74a9783e9
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

# ╔═╡ bdeb034b-8c34-4134-943c-eed63314cf20
measurement_c = fwd(field_c);

# ╔═╡ 525a1be2-286b-4626-b122-377084c6d456
simshow(Array(measurement_c)[:, :, 1], γ=1)

# ╔═╡ 18eb702e-ecad-4648-9e3a-b85cef60deb4
begin
	rec0 = togoc(ones(ComplexF32, (sz..., 2))) .* (rr2(sz) .< 150 .^2)
	#rec0[:, :, 1] .= field_c[:, :, 1]
	#rec0[:, :, 2] .= conv(CuArray(normal(sz, sigma=20)), field_c[:, :, 2])
	#rec0 .= cis.(rr2(sz, scale=0.04, offset=(200, 200))) .* box(sz, (80, 120));
end

# ╔═╡ c9776bdb-04ab-44e7-9c8e-9a7b8eb080f0
simshow(Array(rec0))

# ╔═╡ 504074c0-9ff8-4b5e-8c6a-d196c8111a81
f, g! = make_fg!(fwd, measurement_c)

# ╔═╡ ad585dda-69c5-4bb5-a1bf-1e6001447669
sum(abs2, measurement_c .* diffuser_c)

# ╔═╡ 2d6dadd6-57a8-4c98-bf12-56276cb4cffe
CUDA.@time res = Optim.optimize(f, g!, rec0, ConjugateGradient(),
                                 Optim.Options(iterations = 200,  
                                               store_trace=true, 
								 f_abstol=1e-10,
								 g_abstol=1e-10, ))

# ╔═╡ 870e717a-577f-4097-a650-422e99fde221
@time AS_c2(diffuser_c)

# ╔═╡ a1bb8bb6-7b00-4478-84f3-1e350d530dbe
plot([t.iteration for t in res.trace], [t.value for t in res.trace], label="LBFGS", yaxis=:log)

# ╔═╡ f252bf81-98d7-4383-9c29-fe4c2a034dc6
simshow([Array(field_c[:, :, 1]) cispi(0.42) .* Array(res.minimizer)[:, :, 1]], γ=0.4)

# ╔═╡ df93cc2e-4031-4011-a1d6-d76c62fd3000
simshow(Array(field_c[:, :, 2]))

# ╔═╡ 96c3bbfd-1b94-465e-b852-fadf7649ba57
simshow(abs.([Array(field2) Array(res.minimizer)[:, :, 1]]), γ=1)

# ╔═╡ 633dafe5-0f74-49a5-93bc-aaeec49d0254
simshow(angle.([Array(field2) Array(res.minimizer)[:, :, 1]]))

# ╔═╡ 13d98c90-296f-4fd4-bfc4-93ee81ae3484
simshow(([Array(field2) Array(res.minimizer)[:, :, 2]]))

# ╔═╡ afc890a3-25ae-4c4d-b886-a5433f317e73
simshow(Array(res.minimizer)[:, :, 2])

# ╔═╡ 956c58a2-b182-48a9-b694-b8f6f9ba8077
simshow(Array(field))

# ╔═╡ a9860e96-7888-4e68-9b9b-9adc0e260eef


# ╔═╡ 40c7f76c-2f40-403e-b160-d98ee9c54071
@bind z_i Slider(1:size(measurement_c, 3))

# ╔═╡ 4823c477-38eb-47f0-89fa-6f02a0940d35
#simshow(Array(fwd(res.minimizer)[:, :, z_i]) .- Array(measurement_c[:, :, z_i]))

# ╔═╡ 00aaa61e-48a1-45dd-bdfe-5bd4879cd2d3
simshow(Array(fwd(res.minimizer)[:, :, z_i]), γ=0.8)

# ╔═╡ 6f57680b-7f7a-4d69-a62c-ccfce645151c
simshow(Array(measurement_c[:, :, z_i]), γ=0.8)

# ╔═╡ 88239ba1-5528-46c3-9914-3ea62dfc0cfb


# ╔═╡ a3987c52-33d8-48f4-b373-4b6df1d6c17e
fwd(res.minimizer)

# ╔═╡ Cell order:
# ╠═41c6fe0a-854c-11ee-0c0d-49cafb7df0d1
# ╠═d18059d4-9ff2-4afa-b40f-5c37db59b57f
# ╠═22b6b431-8bc1-4630-9a5e-998874831b84
# ╠═a715d3ef-b3ef-496f-b50f-7ef688df5efc
# ╠═dfacac83-565f-431b-a42c-1072c419a4be
# ╠═a1c76c16-1eff-4cde-adaa-b544f6be45a7
# ╠═74c17e61-15d7-48cd-9fce-689d1c5a3148
# ╠═fdd08423-020d-4dbd-9198-b7b03264a169
# ╠═b2f8e5bf-fc0e-4dbc-9816-3cb0072132c2
# ╠═c34da860-e131-483f-a2d7-ce2dcb1ba1e6
# ╠═e661ac7e-63f9-40d7-ba9f-1784bce61e16
# ╠═a1ba22f5-8015-4e19-ab9c-bc82a4d38e44
# ╠═680c0d66-97af-4d21-b90c-97cb6d8dfbd9
# ╠═7be3d272-cf76-4950-b80b-f743ea200d4a
# ╠═f796abe0-00cc-4178-a945-96ae5a245227
# ╠═e67607af-d100-42f9-a830-4c5889584b3d
# ╠═82a210f8-910b-42b8-bdd2-34b11011d34a
# ╟─9b03154d-bf00-4e95-a5a1-5ae5e7cb48a4
# ╠═1beb527b-19f0-4352-839a-9ac7e91a3f15
# ╠═c11c6881-3d50-49f6-8799-8e16f172ac9a
# ╠═3a6eb1f4-0763-4aa6-92c1-0abe491dd558
# ╠═0180ea0b-87a8-4233-86fe-09308a70738d
# ╠═1418e664-6207-4df4-aa79-f18633b63b2e
# ╠═52f7af37-1a44-4a7c-8a9e-31df9bbcda9a
# ╠═b436fdc0-ae8d-4dbe-94e6-8b7c26339a70
# ╠═c97f6332-82d9-4441-80ad-8a49c7739119
# ╠═0bd359cc-0bd4-4748-8905-361969811248
# ╠═23e64d50-8a6c-418f-86eb-1b4f9917fb32
# ╠═a5593e55-5b77-4c99-b9a9-5e060748d6af
# ╠═d00e3c52-a25d-4f13-9f76-f9fb0e255737
# ╠═86c47180-3afe-4545-912e-12911155a02a
# ╠═2c6ec01e-df45-4482-9d52-ce04ed343c5d
# ╠═df218193-d0f0-4c85-9439-9b70e2e55cf8
# ╠═2531c673-ab41-4374-85fe-e4e256f05c98
# ╠═8516d9be-440c-45c5-8f20-cce74a9783e9
# ╠═bdeb034b-8c34-4134-943c-eed63314cf20
# ╠═525a1be2-286b-4626-b122-377084c6d456
# ╠═18eb702e-ecad-4648-9e3a-b85cef60deb4
# ╠═c9776bdb-04ab-44e7-9c8e-9a7b8eb080f0
# ╠═504074c0-9ff8-4b5e-8c6a-d196c8111a81
# ╠═ad585dda-69c5-4bb5-a1bf-1e6001447669
# ╠═2d6dadd6-57a8-4c98-bf12-56276cb4cffe
# ╠═870e717a-577f-4097-a650-422e99fde221
# ╠═a1bb8bb6-7b00-4478-84f3-1e350d530dbe
# ╠═f252bf81-98d7-4383-9c29-fe4c2a034dc6
# ╠═df93cc2e-4031-4011-a1d6-d76c62fd3000
# ╠═96c3bbfd-1b94-465e-b852-fadf7649ba57
# ╠═633dafe5-0f74-49a5-93bc-aaeec49d0254
# ╠═13d98c90-296f-4fd4-bfc4-93ee81ae3484
# ╠═afc890a3-25ae-4c4d-b886-a5433f317e73
# ╠═956c58a2-b182-48a9-b694-b8f6f9ba8077
# ╠═a9860e96-7888-4e68-9b9b-9adc0e260eef
# ╠═40c7f76c-2f40-403e-b160-d98ee9c54071
# ╠═4823c477-38eb-47f0-89fa-6f02a0940d35
# ╠═00aaa61e-48a1-45dd-bdfe-5bd4879cd2d3
# ╠═6f57680b-7f7a-4d69-a62c-ccfce645151c
# ╠═88239ba1-5528-46c3-9914-3ea62dfc0cfb
# ╠═a3987c52-33d8-48f4-b373-4b6df1d6c17e
