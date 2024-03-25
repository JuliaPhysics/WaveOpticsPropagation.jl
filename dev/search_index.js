var documenterSearchIndex = {"docs":
[{"location":"functions/#Efficient-Propagations","page":"Function Docstrings","title":"Efficient Propagations","text":"","category":"section"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"They create efficient functions which avoid recalculating some parts.","category":"page"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"AngularSpectrum\nFraunhofer\nScalableAngularSpectrum\nShiftedAngularSpectrum\nParams","category":"page"},{"location":"functions/#WaveOpticsPropagation.AngularSpectrum","page":"Function Docstrings","title":"WaveOpticsPropagation.AngularSpectrum","text":"AngularSpectrum(field, z, λ, L; kwargs...)\n\nReturns a function for efficient reuse of pre-calculated kernels. This function returns the electrical field with physical length L and wavelength λ propagated with the angular spectrum  method of plane waves (AS) by the propagation distance z.\n\nThis method is efficient but to avoid recalculating some arrays (such as the phase kernel), see AngularSpectrum. \n\nArguments\n\nfield: Input field\nz: propagation distance. Can be a single number or a vector of zs (Or CuVector). In this case the returning array has one dimension more.\nλ: wavelength of field\nL: field size (can be a scalar or a tuple) indicating field size\n\nKeyword Arguments\n\npadding=true: applies padding to avoid convolution wraparound\nbandlimit=true: applies the bandlimit to avoid circular wraparound due to undersampling    of the complex propagation kernel [1]\nbandlimit_border=(0.8, 1): applies a smooth bandlimit cut-off instead of hard-edge. \n\nSee angular_spectrum for the full documentation.\n\nExample\n\njulia> field = zeros(ComplexF32, (4,4)); field[3,3] = 1\n1\n\njulia> as = AngularSpectrum(field, 100e-9, 632e-9, 10e-6);\n\njulia> as(field);\n\n\n\n\n\n","category":"function"},{"location":"functions/#WaveOpticsPropagation.Fraunhofer","page":"Function Docstrings","title":"WaveOpticsPropagation.Fraunhofer","text":"Fraunhofer(U, z, λ, L; skip_final_phase=true)\n\nThis returns a function for efficient reuse of pre-calculated kernels. This function then returns the electrical field with physical length L and wavelength λ  propagated with the Fraunhofer propagation (a single FFT) by the propagation distance z. This is based on a far field approximation.\n\nArguments\n\nU: Input field\nz: propagation distance\nλ: wavelength of field\nL: field size indicating field size\n\nKeyword Arguments\n\nskip_final_phase=true skip the final phase which is multiplied to the propagated field at the end \n\nSee fraunhofer for the full documentation.\n\nExample\n\njulia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;\n\njulia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;\n\njulia> f = Fraunhofer(field, 4f-3, 632f-9, 100f-6);\n\njulia> res = f(field);\n\njulia> f.params.Lp[1] / 100f-6\n64.71681f0\n\njulia> 4f-3 * 632f-9 * 256 / (100f-6)^2\n64.71681f0\n\n\n\n\n\n","category":"function"},{"location":"functions/#WaveOpticsPropagation.ScalableAngularSpectrum","page":"Function Docstrings","title":"WaveOpticsPropagation.ScalableAngularSpectrum","text":"ScalableAngularSpectrum(field, z, λ, L; kwargs...)\n\nReturns the electrical field with physical length L and wavelength λ propagated with the scalable angular spectrum method of plane waves (SAS) by the propagation distance z.\n\nArguments\n\nfield: Input field\nz: propagation distance\nλ: wavelength of field\nL: field size (can be a scalar or a tuple) indicating field size\n\nKeyword Arguments\n\nskip_final_phase=true: avoid multiplying with final phase. This phase is also undersampled.\n\nExample\n\njulia> field = zeros(ComplexF32, (256,256)); field[130,130] = 1;\n\njulia> f = ScalableAngularSpectrum(field, 10f-3, 633f-9, 500f-6);\n\njulia> f.params.Lp\n0.00162048f0\n\njulia> f.params.Lp / 500f-6 # calculate magnification\n3.24096f0\n\njulia> 10f-3 * 633f-9 * 256 / 500f-6^2 / 2 # equation from paper\n3.2409596f0\n\nReferences\n\nRainer Heintzmann, Lars Loetgering, and Felix Wechsler, \"Scalable angular spectrum propagation,\" Optica 10, 1407-1416 (2023) \n\n\n\n\n\n","category":"function"},{"location":"functions/#WaveOpticsPropagation.ShiftedAngularSpectrum","page":"Function Docstrings","title":"WaveOpticsPropagation.ShiftedAngularSpectrum","text":"ShiftedAngularSpectrum(field, z, λ, L, α; kwargs...)\n\nReturns a method to propagate the electrical field with physical length L and wavelength λ with the shifted angular spectrum  method of plane waves (AS) by the propagation distance z. α should be a tuple containing the offset angles with respect to the optical axis.\n\nArguments\n\nfield: Input field\nz: propagation distance. Can be a single number or a vector of zs (Or CuVector). In this case the returning array has one dimension more.\nλ: wavelength of field\nL: field size (can be a scalar or a tuple) indicating field size\nα is the tuple of shift angles with respect to the optical axis.\n\nKeyword Arguments\n\npad_factor=2: padding of 2. Larger numbers are not recommended since they don't provide better results.\nbandlimit=true: applies the bandlimit to avoid circular wraparound due to undersampling    of the complex propagation kernel [1]\nextract_ramp=true: divides the field by phase ramp exp.(1im * 2π / λ * (sin(α[2]) .* x .+ sin(α[1]) .* y)) and multiplies after                       propagation the ramp (with new real space coordinates) back to the field\n\nExamples\n\njulia> field = zeros(ComplexF32, (4,4)); field[3,3] = 1\n1\n\njulia> shifted_AS = ShiftedAngularSpectrum(field, 100e-6, 633e-9, 100e-6, (deg2rad(10), 0));\n\njulia> shifted_AS(field);\n\nReferences\n\nMatsushima, Kyoji. \"Shifted angular spectrum method for off-axis numerical propagation.\" Optics Express 18.17 (2010): 18453-18463.\n\n\n\n\n\n(shifted_as::ShiftedAngularSpectrum)(field)\n\nUses the struct to efficiently store some pre-calculated objects. Propagate the field.\n\n\n\n\n\n","category":"type"},{"location":"functions/#Propagation","page":"Function Docstrings","title":"Propagation","text":"","category":"section"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"Those were merely for testing purposes.","category":"page"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"fraunhofer\nangular_spectrum","category":"page"},{"location":"functions/#Utilities","page":"Function Docstrings","title":"Utilities","text":"","category":"section"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"pad\ncrop_center\nset_center!","category":"page"},{"location":"functions/#WaveOpticsPropagation.pad","page":"Function Docstrings","title":"WaveOpticsPropagation.pad","text":"pad(arr::AbstractArray{T, N}, new_size; value=zero(T))\n\nPads arr with values around such that the resulting array size is new_size.\n\nSee also crop_center, set_center!.\n\njulia> pad(ones((2,2)), 2)\n4×4 Matrix{Float64}:\n 0.0  0.0  0.0  0.0\n 0.0  1.0  1.0  0.0\n 0.0  1.0  1.0  0.0\n 0.0  0.0  0.0  0.0\n\n\n\n\n\npad(arr::AbstractArray{T, N}, M; value=zero(T))\n\nPads arr with values around such that the resulting array size is round(Int, M .* size(arr)). If M isa Integer, no rounding is needed.\n\nwarn: Automatic Differentation\nIf M isa AbstractFloat, automatic differentation might fail because of size failures\n\nSee also crop_center, set_center!.\n\njulia> pad(ones((2,2)), 2)\n4×4 Matrix{Float64}:\n 0.0  0.0  0.0  0.0\n 0.0  1.0  1.0  0.0\n 0.0  1.0  1.0  0.0\n 0.0  0.0  0.0  0.0\n\n\n\n\n\n","category":"function"},{"location":"functions/#WaveOpticsPropagation.crop_center","page":"Function Docstrings","title":"WaveOpticsPropagation.crop_center","text":"crop_center(arr, new_size)\n\nExtracts a center of an array.  new_size_array must be list of sizes indicating the output size of each dimension. Centered means that a center frequency stays at the center position. Works for even and uneven. If length(new_size_array) < length(ndims(arr)) the remaining dimensions are untouched and copied.\n\nSee also pad, set_center!.\n\nExamples\n\njulia> crop_center([1 2; 3 4], (1,))\n1×2 Matrix{Int64}:\n 3  4\n\njulia> crop_center([1 2; 3 4], (1,1))\n1×1 Matrix{Int64}:\n 4\n\njulia> crop_center([1 2; 3 4], (2,2))\n2×2 Matrix{Int64}:\n 1  2\n 3  4\n\n\n\n\n\n","category":"function"},{"location":"functions/#WaveOpticsPropagation.set_center!","page":"Function Docstrings","title":"WaveOpticsPropagation.set_center!","text":"set_center!(arr_large, arr_small; broadcast=false)\n\nPuts the arr_small central into arr_large.\n\nThe convention, where the center is, is the same as the definition as for FFT based centered.\n\nFunction works both for even and uneven arrays. See also crop_center, pad, set_center!.\n\nKeyword\n\nIf broadcast==false then a lower dimensional arr_small will not be broadcasted\n\nalong the higher dimensions.\n\nIf broadcast==true it will be broadcasted along higher dims.\n\nSee also crop_center, pad.\n\nExamples\n\njulia> set_center!([1, 1, 1, 1, 1, 1], [5, 5, 5])\n6-element Vector{Int64}:\n 1\n 1\n 5\n 5\n 5\n 1\n\njulia> set_center!([1, 1, 1, 1, 1, 1], [5, 5, 5, 5])\n6-element Vector{Int64}:\n 1\n 5\n 5\n 5\n 5\n 1\n\njulia> set_center!(ones((3,3)), [5])\n3×3 Matrix{Float64}:\n 1.0  1.0  1.0\n 1.0  5.0  1.0\n 1.0  1.0  1.0\n\njulia> set_center!(ones((3,3)), [5], broadcast=true)\n3×3 Matrix{Float64}:\n 1.0  1.0  1.0\n 5.0  5.0  5.0\n 1.0  1.0  1.0\n\n\n\n\n\n","category":"function"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"<a  href=\"../docs/logo/logo.png\"><img src=\"../docs/logo/logo.png\"  width=\"150\"></a>","category":"page"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"⠀","category":"page"},{"location":"#WaveOpticsPropagation.jl","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"Propagate waves efficiently, optically, physically, differentiably with Julia Lang. Those functions are fast and memory efficient implemented and hence are suited to be used in inverse problems.","category":"page"},{"location":"#Installation","page":"WaveOpticsPropagation.jl","title":"Installation","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"Officially registered, so install with:","category":"page"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"julia> using Pkg; Pkg.add(\"WaveOpticsPropagation\")","category":"page"},{"location":"#Examples","page":"WaveOpticsPropagation.jl","title":"Examples","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"Look into the examples folder.","category":"page"},{"location":"#Features","page":"WaveOpticsPropagation.jl","title":"Features","text":"","category":"section"},{"location":"#Implemented","page":"WaveOpticsPropagation.jl","title":"Implemented","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"Propagate (electrical) fields based on wave propagation\nPropagations\n[x] Angular Spectrum Method of Plane Waves (AS)\n[x] Fraunhofer Diffraction\n[x] Scalable Angular Spectrum propagation\n[x] Shifted Angular Spectrum propagation\n[ ] Fresnel Propagation with Scaling Behaviour (no priority yet, PR are welcome for that. In principle very similar to the other methods.)\n[x] CUDA support\n[x] Differentiable (mainly based on Zygote.jl and ChainRulesCore.jl)","category":"page"},{"location":"#Planned","page":"WaveOpticsPropagation.jl","title":"Planned","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"Vectorial propagation in free space is just a propagation of each of the components. Right now, this is not a priority and is not implemented yet. But of course, each vectorial component can be propagated separately.","category":"page"},{"location":"#Citation","page":"WaveOpticsPropagation.jl","title":"Citation","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"This package was created as part of scientific work. Please consider citing it :)","category":"page"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"@misc{wechsler2024wave,\n      title={Wave optical model for tomographic volumetric additive manufacturing}, \n      author={Felix Wechsler and Carlo Gigli and Jorge Madrid-Wolff and Christophe Moser},\n      year={2024},\n      eprint={2402.06283},\n      archivePrefix={arXiv},\n      primaryClass={physics.optics}\n}","category":"page"},{"location":"#Development","page":"WaveOpticsPropagation.jl","title":"Development","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"Contributions are very welcome! File an issue on GitHub if you encounter any problems. Also file an issue if you want to discuss or propose features.","category":"page"},{"location":"#Related-packages","page":"WaveOpticsPropagation.jl","title":"Related packages","text":"","category":"section"},{"location":"","page":"WaveOpticsPropagation.jl","title":"WaveOpticsPropagation.jl","text":"There is the outdated PhysicalOptics.jl which provided similar methods. For geometrical ray tracing use OpticSim.jl.","category":"page"}]
}
