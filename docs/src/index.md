# WaveOpticsPropagation.jl

| **Build Status**                          | **Code Coverage**               |
|:-----------------------------------------:|:-------------------------------:|
| [![][CI-img]][CI-url] | [![][codecov-img]][codecov-url] |

Propagate waves efficiently, optically, physically, differentiably with [Julia Lang](https://julialang.org/).
Those functions are fast and memory efficiently implemented and hence are suited to be used in inverse problems.

⚠️ Under heavy development. Expect things to break.

## Installation
Not registered yet, hence install with:
```julia
julia> ]add https://github.com/JuliaPhysics/WaveOpticsPropagation.jl
```

## Features
### Implemented
* Propagate (electrical) fields based on wave propagation
* Propagations
    * [x] Angular Spectrum Method of Plane Waves (AS)
    * [x] Fraunhofer Diffraction
    * [ ] [Scalable Angular Spectrum propagation](https://opg.optica.org/optica/viewmedia.cfm?uri=optica-10-11-1407&html=true)
    * [ ] Fresnel Propagation with Scaling Behaviour (no priority yet, PR are welcome for that. In principle very similar to the other methods.)
* CUDA support
* Differentiable (mainly based on Zygote.jl and ChainRulesCore.jl)

### Planned
In principle vectorial propagation in free space is just a propagation of each of the components. Right now, this is not a priority and is not implemented yet.
But of course, each vectorial component can be propagated separately.

## Development
Contributions are very welcome!
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.
Also file an issue if you want to discuss or propose features.

## Related packages
There is the outdated [PhysicalOptics.jl](https://github.com/JuliaPhysics/PhysicalOptics.jl) which provided similar methods.
For geometrical ray tracing use [OpticSim.jl](https://github.com/brianguenter/OpticSim.jl).

[CI-img]: https://github.com/JuliaPhysics/WaveOpticsPropagation.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/JuliaPhysics/WaveOpticsPropagation.jl/actions/workflows/CI.yml

[codecov-img]: https://codecov.io/gh/JuliaPhysics/WaveOpticsPropagation.jl/branch/main/graph/badge.svg?token=6XWI1M1MPB
[codecov-url]: https://codecov.io/gh/JuliaPhysics/WaveOpticsPropagation.jl