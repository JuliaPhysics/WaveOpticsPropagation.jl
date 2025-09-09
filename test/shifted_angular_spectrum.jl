@testset "Shifted Angular Spectrum" begin
    L = 50f-6
    N = 128 
    α = deg2rad(10f0)
    λ = 405f-9
    z = 100f-6
    sz = (N, N)
    y = fftpos(L, N, CenterFT)
    field = box(Float32, sz, (20,20)) .* exp.(1im .* 2f0 * π ./ λ .* y .* sin(α));
    res_AS = AngularSpectrum(field, z, λ, L)(field);
    res = WaveOpticsPropagation.shifted_angular_spectrum(field, z, λ, L, (α , 0), bandlimit=true)
    shift = z * tan(α) / L * N
    shift2 = z * tan(α) / L
    res2 = ShiftedAngularSpectrum(field, z, λ, L, (α, 0), bandlimit=true)(field)
    @test all(.≈(1 .+ FourierTools.shift(res, (shift, 0))[round(Int, shift)+1:end, :], 1 .+ res_AS[round(Int, shift)+1:end, :], rtol=5f-2))
    @test all(.≈(1 .+ FourierTools.shift(res2, (shift, 0))[round(Int, shift)+1:end, :], 1 .+ res_AS[round(Int, shift)+1:end, :], rtol=5f-2))


    field = box(Float32, sz, (20,20)) .* exp.(1im .* 2f0 * π ./ λ .* y' .* sin(α));
    res_AS = AngularSpectrum(field, z, λ, L)(field);
    res = WaveOpticsPropagation.shifted_angular_spectrum(field, z, λ, L, (0, α), bandlimit=true)
    shift = z * tan(α) / L * N
    shift2 = z * tan(α) / L
    res2 = ShiftedAngularSpectrum(field, z, λ, L, (0, α), bandlimit=true)(field)
    @test all(.≈(1 .+ FourierTools.shift(res, (0, shift))[:, round(Int, shift)+1:end], 1 .+ res_AS[:, round(Int, shift)+1:end], rtol=5f-2))
    @test all(.≈(1 .+ FourierTools.shift(res2, (0, shift))[:, round(Int, shift)+1:end], 1 .+ res_AS[:, round(Int, shift)+1:end], rtol=5f-2))


    res_AS = AngularSpectrum(field, z, λ, L)(field);
    res = WaveOpticsPropagation.shifted_angular_spectrum(field, z, λ, L, (0, 0), bandlimit=true)
    res2 = ShiftedAngularSpectrum(field, z, λ, L, (0, 0), bandlimit=true)(field)
    @test all(.≈(1 .+ res, 1 .+ res_AS, rtol=1f-1)) 
    @test ≈(1 .+ res, 1 .+ res_AS, rtol=1f-2) 
    @test all(.≈(1 .+ res2, 1 .+ res_AS, rtol=1f-1)) 
    @test ≈(1 .+ res2, 1 .+ res_AS, rtol=1f-2) 




    @testset "Test gradient with Finite Differences" begin
        field = zeros(ComplexF64, (24, 24))
        field[14:16, 14:16] .= 1

        α = deg2rad(10)
        gg(x) = sum(abs2.(x .- WaveOpticsPropagation.shifted_angular_spectrum(cis.(x), 100e-6, 633e-9, 100e-6, (α, 0))))

        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg, field)[1]

        # out1 = gradient(gg, field)[1]
        # @test out1 .+ cis(1) ≈ out2  .+ cis(1)
        AS = ShiftedAngularSpectrum(field, 100e-6, 633e-9, 100e-6, (α, 0))

        f_AS(x) = sum(abs2.(x .- AS(cis.(x))))

        out3 = gradient(f_AS, field)[1]

        @test out3 ≈ out2


        field = zeros(ComplexF64, (15, 15))
        field[5:6, 3:8] .= 1
        gg(x) = sum(abs2.(x .- WaveOpticsPropagation.shifted_angular_spectrum(cis.(x), 100e-6, 633e-9, 100e-6, (α, 0))))
        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg, field)[1]
        # out1 = gradient(gg, field)[1]
        # @test out1 .+ cis(1) ≈ out2  .+ cis(1)
        AS = ShiftedAngularSpectrum(field, 100e-6, 633e-9, 100e-6, (α, 0))
        f_AS(x) = sum(abs2.(x .- AS(cis.(x))))
        out3 = gradient(f_AS, field)[1]
        @test out3 ≈ out2

        
        field = zeros(ComplexF64, (15, 15))
        field[5:6, 3:8] .= 1
        AS = ShiftedAngularSpectrum(field, 100e-6, 633e-9, 100e-6, (α, 0); extract_ramp=false)
        f_AS(x) = sum(abs2.(x .- AS(cis.(x))))
        out2 = FiniteDifferences.grad(central_fdm(5, 1), f_AS, field)[1]
        out3 = gradient(f_AS, field)[1]
        @test out3 ≈ out2


    end

end
