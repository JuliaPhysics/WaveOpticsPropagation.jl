```@raw html
<a  href="../docs/logo/logo.png"><img src="../docs/logo/logo.png"  width="150"></a>
``` ⠀

# WaveOpticsPropagation.jl


Propagate waves efficiently, optically, physically, differentiably with [Julia Lang](https://julialang.org/).
Those functions are fast and memory efficient implemented and hence are suited to be used in inverse problems.

⚠️ Under development. Expect things to break. But feel free to try the examples, they should always work!

## Installation
Not registered yet, hence install with:
```julia
julia> ]add https://github.com/JuliaPhysics/WaveOpticsPropagation.jl
```

## Examples
See [the rendered notebooks here](https://juliaphysics.github.io/WaveOpticsPropagation.jl/).
Otherwise, just look into the [examples folder](examples/).

## Features
### Implemented
* Propagate (electrical) fields based on wave propagation
* Propagations
    * [x] Angular Spectrum Method of Plane Waves (AS)
    * [x] Fraunhofer Diffraction
    * [x] [Scalable Angular Spectrum propagation](https://opg.optica.org/optica/viewmedia.cfm?uri=optica-10-11-1407&html=true)
    * [ ] Fresnel Propagation with Scaling Behaviour (no priority yet, PR are welcome for that. In principle very similar to the other methods.)
* [x] CUDA support
* [x] Differentiable (mainly based on Zygote.jl and ChainRulesCore.jl)

### Planned
Vectorial propagation in free space is just a propagation of each of the components. Right now, this is not a priority and is not implemented yet.
But of course, each vectorial component can be propagated separately.

## Development
Contributions are very welcome!
File an [issue](https://github.com/JuliaPhysics/WaveOpticsPropagation.jl/issues) on [GitHub](https://github.com/JuliaPhysics/WaveOpticsPropagation.jl) if you encounter any problems.
Also file an issue if you want to discuss or propose features.

## Related packages
There is the outdated [PhysicalOptics.jl](https://github.com/JuliaPhysics/PhysicalOptics.jl) which provided similar methods.
For geometrical ray tracing use [OpticSim.jl](https://github.com/brianguenter/OpticSim.jl).

