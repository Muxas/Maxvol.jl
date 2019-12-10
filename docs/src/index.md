# Documentation for Maxvol

This **Julia** package provides two routines:
- `maxvol_generic!`, generic implementation of **Maxvol** algorithm, which can
  be used for matrices of any numerical type (e.g. `Rational` or `BigFloat`).
- `maxvol!`, LAPACK-based implementation of **Maxvol** algorithm, which works
  only for standard numerical types: `Float32`, `Float64`, `ComplexF32` and
  `ComplexF64`.

# Methods

```@docs
maxvol_generic!
maxvol!
```

# License
This package is ditributed under BSD 3-Clause license. It can be found in the
root directory of repository in *LICENSE.md* file.
