@testset "Angular Spectrum" begin
    L = 100f-3
    λ = 633f-9
    z = 1
    N = 256
    ΔS = 18
    ΔW = 2
	slit = zeros(ComplexF32, (N, N))
    
    x = WaveOpticsPropagation.fftpos(L, N)
    W = x[1 + ΔW * 2 + 1] - x[1]
    S = x[1 + ΔS * 2] - x[1]

    L_new_ref = λ * z / L * N
	mid = N ÷ 2+ 1
	slit[:, mid-ΔS-ΔW:mid-ΔS+ΔW] .= 1
	slit[:, mid+ΔS-ΔW:mid+ΔS+ΔW] .= 1
    output, t = fraunhofer(slit, z, λ, L)
    L_new = t.L    

    efficient_fraunhofer, t = Fraunhofer(slit, z, λ, L);
    output2, L_new2 = efficient_fraunhofer(slit)
    

    @test L_new ≈ L_new_ref
    @test L_new2.L ≈ L_new_ref

    @test output2 ≈ output

    I_analytical(x) = sinc(W * x / λ / z)^2 * cos(π * S * x / λ / z)^2
    intensity = abs2.(output)
	intensity ./= maximum(intensity)
    xpos_out = WaveOpticsPropagation.fftpos(L_new, N, NDTools.CenterFT)

    @test all(≈(I_analytical.(xpos_out)[110:150] .+ 1, 1 .+  intensity[129, 110:150, :], rtol=1f-2))

    arr = randn(ComplexF32, (N, N))
    fr, t = Fraunhofer(arr, z, λ, L)
    f(x) = sum(abs2, arr .- fr(x)[1]) 
    f2(x) = sum(abs2, arr .- fraunhofer(x, z, λ, L)[1])
    @test Zygote.gradient(f, arr)[1] ≈ Zygote.gradient(f2, arr)[1]

    arr = randn(ComplexF32, (15, 15))
    fr, t = Fraunhofer(arr, z, λ, L, skip_final_phase=false)
    f(x) = sum(abs2, arr .- fr(x)[1]) 
    f2(x) = sum(abs2, arr .- fraunhofer(x, z, λ, L, skip_final_phase=false)[1])
    @test Zygote.gradient(f, arr)[1] ≈ Zygote.gradient(f2, arr)[1]

end
