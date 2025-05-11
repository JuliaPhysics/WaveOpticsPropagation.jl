using WaveOpticsPropagation, Documenter

DocMeta.setdocmeta!(WaveOpticsPropagation, :DocTestSetup, :(using WaveOpticsPropagation); recursive=true)

makedocs(;
    modules = [WaveOpticsPropagation],
    sitename = "WaveOpticsPropagation.jl",
    pages = Any[
        "WaveOpticsPropagation.jl" => "index.md",
        "Function Docstrings" => "functions.md",
    ],
    format = Documenter.HTML(;
        assets = [
            "assets/favicon.ico",
        ],
    ),
    warnonly = true,
)

deploydocs(
    repo = "github.com/JuliaPhysics/WaveOpticsPropagation.jl.git",
    devbranch = "main",
    push_preview = true,
)
