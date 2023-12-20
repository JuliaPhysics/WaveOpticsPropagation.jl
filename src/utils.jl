export crop_center
export set_center!
export pad


hann(x) = sin(π * x / 2)^2
scale(x, α, β) = clamp(1 - (x - α) / (β - α), 0, 1)
smooth_f(x, α, β) = hann(scale(x, α, β))

"""
    get_indices_around_center(i_in, i_out)

A function which provides two output indices `i1` and `i2`
where `i2 - i1 = i_out`

The indices are chosen in a way that the range `i1:i2`
cuts the interval `1:i_in` in a way that the center frequency
stays at the center position.

Works for both odd and even indices.
"""
function get_indices_around_center(i_in, i_out)
    if (mod(i_in, 2) == 0 && mod(i_out, 2) == 0 
     || mod(i_in, 2) == 1 && mod(i_out, 2) == 1) 
        x = (i_in - i_out) ÷ 2
        return 1 + x, i_in - x
    elseif mod(i_in, 2) == 1 && mod(i_out, 2) == 0
        x = (i_in - 1 - i_out) ÷ 2
        return 1 + x, i_in - x - 1 
    elseif mod(i_in, 2) == 0 && mod(i_out, 2) == 1
        x = (i_in - (i_out - 1)) ÷ 2
        return 1 + x, i_in - (x - 1)
    end
end

"""
    pad(arr::AbstractArray{T, N}, new_size; value=zero(T))

Pads `arr` with `values` around such that the resulting array size is `new_size`.

See also [`crop_center`](@ref), [`set_center!`](@ref).

```jldoctest
julia> pad(ones((2,2)), 2)
4×4 Matrix{Float64}:
 0.0  0.0  0.0  0.0
 0.0  1.0  1.0  0.0
 0.0  1.0  1.0  0.0
 0.0  0.0  0.0  0.0
```
"""
function pad(arr::AbstractArray{T, N}, new_size::NTuple; value=zero(T)) where {T, N}
    n_arr = similar(arr, new_size)
    fill!(n_arr, value)
    # don't do broadcast. Does not fit with the meaning of pad
    set_center!(n_arr, arr, broadcast=false)
end


"""
    pad(arr::AbstractArray{T, N}, M; value=zero(T))

Pads `arr` with `values` around such that the resulting array size is `round(Int, M .* size(arr))`.
If `M isa Integer`, no rounding is needed.


!!! warn "Automatic Differentation"
    If `M isa AbstractFloat`, automatic differentation might fail because of size failures


See also [`crop_center`](@ref), [`set_center!`](@ref).

```jldoctest
julia> pad(ones((2,2)), 2)
4×4 Matrix{Float64}:
 0.0  0.0  0.0  0.0
 0.0  1.0  1.0  0.0
 0.0  1.0  1.0  0.0
 0.0  0.0  0.0  0.0
```
"""
function pad(arr::AbstractArray{T, N}, M::Number; value=zero(T)) where {T, N}
    pad(arr, round.(Int, size(arr) .* M), value=value)
end


"""
    crop_center(arr, new_size)

Extracts a center of an array. 
`new_size_array` must be list of sizes indicating the output
size of each dimension. Centered means that a center frequency
stays at the center position. Works for even and uneven.
If `length(new_size_array) < length(ndims(arr))` the remaining dimensions
are untouched and copied.


See also [`pad`](@ref), [`set_center!`](@ref).

# Examples
```jldoctest
julia> crop_center([1 2; 3 4], (1,))
1×2 Matrix{Int64}:
 3  4

julia> crop_center([1 2; 3 4], (1,1))
1×1 Matrix{Int64}:
 4

julia> crop_center([1 2; 3 4], (2,2))
2×2 Matrix{Int64}:
 1  2
 3  4
```
"""
function crop_center(arr, new_size::NTuple{N}; return_view=true) where {N}
    M = ndims(arr)
    @assert N ≤ M "Can't specify more dimensions than the array has."
    @assert all(new_size .≤ size(arr)[1:N]) "You can't extract a larger array than the input array."
    @assert Base.require_one_based_indexing(arr) "Require one based indexing arrays"

    out_indices = ntuple(i ->   let  
                                    if i ≤ N
                                        inds = get_indices_around_center(size(arr, i),
                                                                         new_size[i])
                                        inds[1]:inds[2]
                                    else
                                        1:size(arr, i)
                                    end
                                end,
                          Val(M))
    

    # if return_view
        # return @inbounds view(arr, out_indices...)
    # else
        return @inbounds arr[out_indices...]
    # end
end


"""
    set_center!(arr_large, arr_small; broadcast=false)

Puts the `arr_small` central into `arr_large`.

The convention, where the center is, is the same as the definition
as for FFT based centered.

Function works both for even and uneven arrays.
See also [`crop_center`](@ref), [`pad`](@ref), [`set_center!`](@ref).

# Keyword
* If `broadcast==false` then a lower dimensional `arr_small` will not be broadcasted
along the higher dimensions.
* If `broadcast==true` it will be broadcasted along higher dims.


See also [`crop_center`](@ref), [`pad`](@ref).


# Examples
```jldoctest
julia> set_center!([1, 1, 1, 1, 1, 1], [5, 5, 5])
6-element Vector{Int64}:
 1
 1
 5
 5
 5
 1

julia> set_center!([1, 1, 1, 1, 1, 1], [5, 5, 5, 5])
6-element Vector{Int64}:
 1
 5
 5
 5
 5
 1

julia> set_center!(ones((3,3)), [5])
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  5.0  1.0
 1.0  1.0  1.0

julia> set_center!(ones((3,3)), [5], broadcast=true)
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 5.0  5.0  5.0
 1.0  1.0  1.0
```
"""
function set_center!(arr_large::AbstractArray{T, N}, arr_small::AbstractArray{T1, M};
                     broadcast=false) where {T, T1, M, N}
    @assert N ≥ M "Can't put a higher dimensional array in a lower dimensional one."

    if broadcast == false
        inds = ntuple(i -> begin
                        a, b = get_indices_around_center(size(arr_large, i), size(arr_small, i))
                        a:b
                      end,
                      Val(N)) 
        arr_large[inds..., ..] .= arr_small
    else
        inds = ntuple(i -> begin
                        a, b = get_indices_around_center(size(arr_large, i), size(arr_small, i))
                        a:b
                      end,
                      Val(M)) 
        arr_large[inds..., ..] .= arr_small
    end

    
    return arr_large
end

function ChainRulesCore.rrule(::typeof(crop_center), arr, new_size::NTuple{N, <:Int}) where N
    y = crop_center(arr, new_size) 
    function crop_center_pullback(ȳ)
        c̄ = pad(ȳ, size(arr), value=zero(eltype(arr)))
        return NoTangent(), c̄, NoTangent()
    end
    return y, crop_center_pullback
end


function ChainRulesCore.rrule(::typeof(pad), arr, M::Union{NTuple, Number}; value=zero(eltype(arr)))
    y = pad(arr, M; value=value) 
    function pad_pullback(ȳ)
        @assert size(y) == size(ȳ)
        p̄ = crop_center(ȳ, size(arr))
        return NoTangent(), p̄, NoTangent()
    end
    return y, pad_pullback
end
