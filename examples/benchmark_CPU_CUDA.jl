### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# ╔═╡ 2fd337ac-8ad0-11ee-3739-459b5825a8c5
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 958b4e13-bb6c-4d4d-83f9-a922bbbfb842
using WaveOpticsPropagation, FFTW, CUDA, Zygote

# ╔═╡ 62c0385d-e712-43eb-adae-6731159f9f92
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"   
	togoc(x) = use_CUDA[] ? CuArray(x) : x 
end

# ╔═╡ 8ee34fde-40a7-4cc1-a8ea-056a457901b0
md"# Compared with FFT"

# ╔═╡ a43a5d7e-5052-4ee1-a112-c8881e90b6a6
sz = (2048, 2048)

# ╔═╡ c33ed1de-feb3-4bab-80c5-cd422be96bb6
array = randn(ComplexF32, sz);

# ╔═╡ 5d7cf31a-7e08-45ea-80e4-a0e27a022258
array_c = togoc(array);

# ╔═╡ 4e651fc1-b6db-4c44-b830-f65c6f68f4a9
p_cpu = plan_fft!(array, flags=FFTW.ESTIMATE);

# ╔═╡ 5f5113ed-d335-4038-a230-b739242aafbc
p_cuda = plan_fft!(array_c);

# ╔═╡ f897cd86-ddc1-4b84-a11c-d99161c1df3d
@mytime p_cpu * array;

# ╔═╡ fcf4d2ab-512b-4a2c-a131-9d2519da1747
@mytime p_cpu * array;

# ╔═╡ f7633864-230a-4a06-a48e-8acf4adb58f1
@mytime CUDA.@sync p_cuda * array_c;

# ╔═╡ 0a7b6bc6-9a23-484b-ac26-6effb901ea3b
@mytime CUDA.@sync p_cuda * array_c;

# ╔═╡ 181eb167-f944-480d-907a-e54a2ba113df
md"# Propagation"

# ╔═╡ 864c37c5-6907-4b31-8a4e-8c3c9a684606
AS = AngularSpectrum(array, 1f0, 1f0, 1f0)[1];

# ╔═╡ d9339c77-35cf-4575-b935-79d580badba5
AS_c = AngularSpectrum(array_c, 1f0, 1f0, 1f0)[1];

# ╔═╡ ad07d942-4985-40dc-9371-d6579e5b92a3
@mytime angular_spectrum(array, 1f0, 1f0, 1f0);

# ╔═╡ 127e50d1-4538-4010-b7c3-716bffd804ed
@mytime AS(array);

# ╔═╡ b0d00f6f-2a6c-42c0-b878-1791b2073353
@mytime CUDA.@sync angular_spectrum(array_c, 1f0, 1f0, 1f0);

# ╔═╡ 20935643-b699-46c0-8a44-d948551d9c44
@mytime CUDA.@sync AS_c(array_c);

# ╔═╡ a0637882-140c-466f-ab7d-af9cdb979d1f
md"# Gradient"

# ╔═╡ ca2386b8-0882-4b31-90fa-7ab1df839457
f(x) = sum(abs2, AS_c(x)[1])

# ╔═╡ 17679b22-61ac-42c2-959c-5e28fe880d15
@mytime CUDA.@sync f(array_c)

# ╔═╡ 02c3e595-b978-4ad4-961d-5d7398d3d320
@mytime CUDA.@sync Zygote.gradient(f, array_c)

# ╔═╡ 819f3128-3a42-425b-871d-a4913b17e94c
@mytime CUDA.@sync Zygote.withgradient(f, array_c)

# ╔═╡ Cell order:
# ╠═2fd337ac-8ad0-11ee-3739-459b5825a8c5
# ╠═958b4e13-bb6c-4d4d-83f9-a922bbbfb842
# ╠═62c0385d-e712-43eb-adae-6731159f9f92
# ╟─8ee34fde-40a7-4cc1-a8ea-056a457901b0
# ╠═a43a5d7e-5052-4ee1-a112-c8881e90b6a6
# ╠═c33ed1de-feb3-4bab-80c5-cd422be96bb6
# ╠═5d7cf31a-7e08-45ea-80e4-a0e27a022258
# ╠═4e651fc1-b6db-4c44-b830-f65c6f68f4a9
# ╠═5f5113ed-d335-4038-a230-b739242aafbc
# ╠═f897cd86-ddc1-4b84-a11c-d99161c1df3d
# ╠═fcf4d2ab-512b-4a2c-a131-9d2519da1747
# ╠═f7633864-230a-4a06-a48e-8acf4adb58f1
# ╠═0a7b6bc6-9a23-484b-ac26-6effb901ea3b
# ╟─181eb167-f944-480d-907a-e54a2ba113df
# ╠═864c37c5-6907-4b31-8a4e-8c3c9a684606
# ╠═d9339c77-35cf-4575-b935-79d580badba5
# ╠═ad07d942-4985-40dc-9371-d6579e5b92a3
# ╠═127e50d1-4538-4010-b7c3-716bffd804ed
# ╠═b0d00f6f-2a6c-42c0-b878-1791b2073353
# ╠═20935643-b699-46c0-8a44-d948551d9c44
# ╠═a0637882-140c-466f-ab7d-af9cdb979d1f
# ╠═ca2386b8-0882-4b31-90fa-7ab1df839457
# ╠═17679b22-61ac-42c2-959c-5e28fe880d15
# ╠═02c3e595-b978-4ad4-961d-5d7398d3d320
# ╠═819f3128-3a42-425b-871d-a4913b17e94c
