@testset "Angular Spectrum" begin

    @testset "Test gradient with Finite Differences" begin
        field = zeros(ComplexF64, (24, 24))
        field[14:16, 14:16] .= 1

        gg(x) = sum(abs2.(x .- angular_spectrum(cis.(x), 100e-6, 633e-9, 100e-6)[1]))

        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg, field)[1]

        out1 = gradient(gg, field)[1]
        @test out1 .+ cis(1) ≈ out2  .+ cis(1)
        AS, _ = Angular_Spectrum(field, 100e-6, 633e-9, 100e-6)

        f_AS(x) = sum(abs2.(x .- AS(cis.(x))[1]))

        out3 = gradient(f_AS, field)[1]

        @test out3 ≈ out1


        field = zeros(ComplexF64, (15, 15))
        field[5:6, 3:8] .= 1

        gg(x) = sum(abs2.(x .- angular_spectrum(cis.(x), 100e-6, 633e-9, 100e-6)[1]))

        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg, field)[1]

        out1 = gradient(gg, field)[1]
        @test out1 .+ cis(1) ≈ out2  .+ cis(1)
        AS, _ = Angular_Spectrum(field, 100e-6, 633e-9, 100e-6)

        f_AS(x) = sum(abs2.(x .- AS(cis.(x))[1]))

        out3 = gradient(f_AS, field)[1]

        @test out3 ≈ out1

    end

    @testset "double slit" begin
        include("double_slit.jl")  
    end

    @testset "Gaussian beam" begin
        include("gaussian_beam.jl")
    end
end
