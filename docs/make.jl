using Documenter
using Maxvol

makedocs(
    sitename = "Maxvol",
    format = Documenter.HTML(),
    modules = [Maxvol]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
