# What is this repo?

This is a repo for **Julia** implementation of **Maxvol**-related algorithms.

# What is Maxvol?

**Maxvol** is an algorithm which finds a submatrix of quasi-maximum volume in a
given matrix. Submatrices of maximum volume play crucial role in low-rank
cross (interpolative) approximations as well as in different optimization
problems. More on this can be found in the following literature:

- S.A. Goreinov, N.L. Zamarashkin and E.E. Tyrtyshnikov, 1997. *Pseudo-skeleton
approximations by matrices of maximal volume*. In *Mathematical Notes*, 62(4),
(pp. 515-519).

- S.A. Goreinov, E.E. Tyrtyshnikov and N.L. Zamarashkin, 1997. *A theory of
pseudoskeleton approximations*. In *Linear algebra and its applications*,
261(1-3), (pp. 1-21).

- S.A. Goreinov, I.V. Oseledets, D.V. Savostyanov, E.E. Tyrtyshnikov and
N.L. Zamarashkin, 2010. *How to find a good submatrix*. In *Matrix Methods:
Theory, Algorithms And Applications: Dedicated to the Memory of Gene Golub*
(pp. 247-256).

- I. Oseledets and E. Tyrtyshnikov, 2010. *TT-cross approximation for
multidimensional arrays*. In *Linear Algebra and its Applications*, 432(1),
(pp. 70-88).

- D.V. Savostyanov, 2014. *Quasioptimality of maximum-volume cross interpolation of
tensors*. In *Linear Algebra and its Applications*, 458, (pp. 217-244).

- N.L. Zamarashkin and A.I. Osinsky, 2016. *New accuracy estimates for pseudoskeleton
approximations of matrices*. In *Doklady Mathematics* (Vol. 94, No. 3, pp. 643-645).

- A.I. Osinsky and N.L. Zamarashkin, 2018. *Pseudo-skeleton approximations with
better accuracy estimates*. In *Linear Algebra and its Applications*, 537,
(pp. 221-249).

- A. Mikhalev and I.V. Oseledets, 2018. *Rectangular maximum-volume submatrices
and their applications*. In *Linear Algebra and its Applications*, 538,
(pp. 187-211).

# Installation

As it is a **Julia** package, it can be installed with a simple
```
julia> using Pkg
julia> Pkg.add("https://github.com/muxas/maxvol.jl")
```

# Documentation

Is available online [here](https://muxas.github.io/Maxvol.jl/).
