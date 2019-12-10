# @copyright (c) 2019 RWTH Aachen. All rights reserved.
#
# @file docs/make.jl
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2019-12-10

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
