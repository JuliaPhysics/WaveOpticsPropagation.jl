@testset "Scalable Angular Spectrum" begin

    @testset "Test gradient with Finite Differences" begin
        field = zeros(ComplexF64, (32, 32))
        field[14:16, 14:16] .= 1


        ss = ScalableAngularSpectrum(cis.(field), 100e-6, 633e-9, 100e-6) 
        gg(x) = sum(abs2.(x .- ss(cis.(x))[1]))

        out2 = FiniteDifferences.grad(central_fdm(5, 1), gg, field)[1]

        out1 = gradient(gg, field)[1]
        @test out1 .+ cis(1) ≈ out2  .+ cis(1)
    end


end
