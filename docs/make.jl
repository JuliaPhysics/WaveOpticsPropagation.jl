using WaveOpticsPropagation, Documenter 

DocMeta.setdocmeta!(WaveOpticsPropagation, :DocTestSetup, :(using WaveOpticsPropagation); recursive=true)
makedocs(modules = [WaveOpticsPropagation], 
         sitename = "WaveOpticsPropagation.jl", 
         pages = Any[
            "WaveOpticsPropagation.jl" => "index.md",
            "Function Docstrings" =>  "functions.md"
         ],
         warnonly=true,
        )

deploydocs(repo = "github.com/roflmaostc/WaveOpticsPropagation.jl.git", devbranch="main")
