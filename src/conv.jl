

function conv(u_in, v, dims=(1,2))
    u = pad(u_in, size(v)[1:2])

    return crop_center(ifft(fft(u, dims) .* fft(v, dims), dims), size(u_in))
end
