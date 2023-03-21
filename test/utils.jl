
@testset "Pad" begin
    @test pad(ones((2, 2)), 2) == [0.0 0.0 0.0 0.0; 0.0 1.0 1.0 0.0; 0.0 1.0 1.0 0.0; 0.0 0.0 0.0 0.0]
    @test pad(ones((1,)), 2) == [0.0, 1.0]
    @test pad(ones((1,)), 3) == [0.0, 1.0, 0.0]
    @test pad(ones((1,)), 4) == [0.0, 0.0, 1.0, 0.0]
    @test pad(ones((1,)), 5) == [0.0, 0.0, 1.0, 0.0, 0.0]
    @test pad(ones((2, 2)), 3) == [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0]
    @test pad(ones((3, 3)), 3) == [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]

    @test pad(ones((2, 2)), (2, 2, 2)) == [0.0 0.0; 0.0 0.0;;; 1.0 1.0; 1.0 1.0]
    @test pad(ones((1, 1)), (1, 3, 4)) == [0.0 0.0 0.0;;; 0.0 0.0 0.0;;; 0.0 1.0 0.0;;; 0.0 0.0 0.0]
    @test pad(ones((1, 1)), (1, 3, 4), value = 10) == [10.0 10.0 10.0;;; 10.0 10.0 10.0;;; 10.0 1.0 10.0;;; 10.0 10.0 10.0]
    @test pad(ones((2, 2)), 2.5) == [0.0 0.0 0.0 0.0 0.0; 0.0 1.0 1.0 0.0 0.0; 0.0 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
end

@testset "set_center!" begin
    @test set_center!([-1, 1, 1, 13, 1, 4], [5, 5, 5]) == [-1, 1, 5, 5, 5, 4]
    @test set_center!(ones((3, 3)), [5]) == [1.0 1.0 1.0; 1.0 5.0 1.0; 1.0 1.0 1.0]
    @test set_center!(ones((3, 3)), [5], broadcast = true) == [1.0 1.0 1.0; 5.0 5.0 5.0; 1.0 1.0 1.0]
end

@testset "crop center" begin
    @test crop_center([1 2; 3 4], (1,)) == [3 4]
    @test crop_center([1 2 3; 3 4 5], (1,)) == [3 4 5]
    @test crop_center([1 2 3; 3 4 5], (1, 2)) == [3 4]
    @test crop_center([1 2 3; 3.3 4 5], (1, 2)) == [3.3 4.0]
    @test crop_center([1 2 3; 3.3 4 5], (2, 2)) == [1.0 2.0; 3.3 4.0]
end


@testset "test rrule for pad and crop_center" begin
    test_rrule(pad, randn((4,2)), 2, check_thunked_output_tangent=false)
    test_rrule(pad, randn((2,8)), 2.5, check_thunked_output_tangent=false)
    test_rrule(pad, randn((1,9)), 1, check_thunked_output_tangent=false)
    test_rrule(pad, randn((1,9)), (2,20), check_thunked_output_tangent=false)
    test_rrule(pad, randn((1,9)), (1,9), check_thunked_output_tangent=false)
    
    test_rrule(crop_center, randn((6,6)), (1,1), check_thunked_output_tangent=false)
    test_rrule(crop_center, randn((6,4)), (2,3), check_thunked_output_tangent=false)
    test_rrule(crop_center, randn((5,5)), (3,3), check_thunked_output_tangent=false)
    test_rrule(crop_center, randn((5,5)), (5,5), check_thunked_output_tangent=false)
end
