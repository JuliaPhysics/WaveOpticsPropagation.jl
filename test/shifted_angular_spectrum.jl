@testset "Shifted Angular Spectrum" begin
    L = 50f-6
    N = 128 
    α = deg2rad(10f0)
    λ = 405f-9
    z = 100f-6
    sz = (N, N)
    y = fftpos(L, N, CenterFT)
    field = box(Float32, sz, (20,20)) .* exp.(1im .* 2f0 * π ./ λ .* y .* sin(α));
    res_AS = AngularSpectrum(field, z, λ, L)[1](field)[1];
    res = shifted_angular_spectrum(field, z, λ, L, (α , 0), bandlimit=true)
    shift = z * tan(α) / L * N
    shift2 = z * tan(α) / L
    @test all(.≈(1 .+ FourierTools.shift(res[1], (shift, 0))[round(Int, shift)+1:end, :], 1 .+ res_AS[round(Int, shift)+1:end, :], rtol=5f-2))


    field = box(Float32, sz, (20,20)) .* exp.(1im .* 2f0 * π ./ λ .* y' .* sin(α));
    res_AS = AngularSpectrum(field, z, λ, L)[1](field)[1];
    res = shifted_angular_spectrum(field, z, λ, L, (0, α), bandlimit=true)
    shift = z * tan(α) / L * N
    shift2 = z * tan(α) / L
    @test all(.≈(1 .+ FourierTools.shift(res[1], (0, shift))[:, round(Int, shift)+1:end], 1 .+ res_AS[:, round(Int, shift)+1:end], rtol=5f-2))


    res_AS = AngularSpectrum(field, z, λ, L)[1](field)[1];
    res = shifted_angular_spectrum(field, z, λ, L, (0, 0), bandlimit=true)[1]
    @test all(.≈(1 .+ res, 1 .+ res_AS, rtol=1f-1)) 
    @test ≈(1 .+ res, 1 .+ res_AS, rtol=1f-2) 

end
