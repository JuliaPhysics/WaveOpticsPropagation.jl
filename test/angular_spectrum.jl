@testset "Angular Spectrum" begin

    @testset "Test gradient with Finite Differences" begin
        field = zeros(ComplexF64, (14, 14))
        field[8:12, 8:12] .= 1

        gg(x) = sum(abs2.(x .- WaveOpticsPropagation.angular_spectrum(cis.(x), 100e-6, 633e-9, 100e-6)))

        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg, field)[1]

        AS = AngularSpectrum(field, 100e-6, 633e-9, 100e-6)

        f_AS(x) = sum(abs2.(x .- AS(cis.(x))))

        out3 = gradient(f_AS, field)[1]

        @test out3 ≈ out2


        field = zeros(ComplexF64, (15, 15))
        field[5:6, 3:8] .= 1
        gg3(x) = sum(abs2.(x .- WaveOpticsPropagation.angular_spectrum(cis.(x), 100e-6, 633e-9, 100e-6)))
        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg3, field)[1]
        AS = AngularSpectrum(field, 100e-6, 633e-9, 100e-6)
        f_AS3(x) = sum(abs2.(x .- AS(cis.(x))))
        out3 = gradient(f_AS3, field)[1]
        @test out2 ≈ out3


        field = zeros(ComplexF64, (15, 15))
        field[5:6, 3:8] .= 1
        gg2(x) = sum(abs2.(x .- WaveOpticsPropagation.angular_spectrum(cis.(x), [100e-6, 200e-6], 633e-9, 100e-6)))
        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg2, field)[1]
        AS = AngularSpectrum(field, [100e-6, 200e-6], 633e-9, 100e-6)
        f_AS2(x) = sum(abs2.(x .- AS(cis.(x))))
        out3 = gradient(f_AS2, field)[1]
        @test out2 ≈ out3

    end

    @testset "Test symmetry" begin
        arr = randn(ComplexF32, (4,4))
        arr_ = permutedims(arr, (2,1))
        @test AngularSpectrum(arr, 100e-6, 633e-9, (100e-6, 10e-6))(arr)[:] ≈ permutedims(AngularSpectrum(arr_, 100e-6, 633e-9, (10e-6, 100e-6))(arr_), (2,1))[:]
        
        arr = randn(ComplexF32, (4,2))
        arr_ = permutedims(arr, (2,1))
        @test AngularSpectrum(arr, 100e-6, 633e-9, (100e-6, 10e-6))(arr)[:] ≈ permutedims(AngularSpectrum(arr_, 100e-6, 633e-9, (10e-6, 100e-6))(arr_), (2,1))[:]
    end


    @testset "double slit" begin
        include("double_slit.jl")  
    end

    @testset "Gaussian beam" begin
        include("gaussian_beam.jl")
    end
end
