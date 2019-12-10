# @copyright (c) 2019 RWTH Aachen. All rights reserved.
#
# @file src/Maxvol.jl
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2019-12-10

module Maxvol

using LinearAlgebra.LAPACK: getrf!
using LinearAlgebra.BLAS: trsm!, ger!, iamax

export maxvol_generic!, maxvol!

"""
    maxvol_generic!(A, tol=1.05, maxiters=100)

Generic maxvol, that does not use any **BLAS** or **LAPACK** calls.

Can be used for special arithmetics, like `BigFloat`, `Rational` or
`Complex{Float16}`.

# Arguments:
- `A::Matrix`: Input matrix on entry and output coefficients on exit.
- `tol::Float64`: Stop when determinant growth is less or equal to this.
- `maxiters::Int`: Maximum number of iterations.

# Returns:
- `piv::Vector{Int}`: Indexes of pivoted rows
- `niters::Int`: Number of performed swap iterations

# Example:
```julia
julia> using Random, LinearAlgebra, Maxvol
julia> rng = MersenneTwister(100);
julia> A = rand(rng, BigFloat, 1000, 100);
julia> C = copy(A);
julia> piv, niters = maxvol_generic!(C);
julia> norm(A-C*A[piv,:]) / norm(A)
2.030657951400512330834848952202721164346464876711701213634530270353170311161736e-76
julia> A = rand(rng, ComplexF64, 1000, 100);
julia> C = copy(A);
julia> piv, niters = maxvol_generic!(C);
julia> norm(A-C*A[piv,:]) / norm(A)
4.863490630095799e-15
```

# See also: [`maxvol!`](@ref)
"""
function maxvol_generic!(A::Matrix{T}, tol::Float64=1.05,
                         maxiters::Int=100) where T <: Number
    # Check input parameters
    if tol < 1
        throw(ArgumentError("Parameter `tol` must be at least 1.0"))
    end
    if maxiters < 0
        throw(ArgumentError("Parameter `maxiters` must be non-negative"))
    end
    # Compute pivoted LU at first
    N, r = size(A)
    ipiv = Vector{typeof(N)}(undef, r)
    basis::Vector{typeof(N)} = 1 : N
    # Cycle over all columns
    for j = 1 : r
        maxi = j
        maxv = abs(A[j, j])
        # Cycle over lower triangular elements in a given column
        # to find maximum absolute value
        for i = j+1 : N
            tmpv = abs(A[i, j])
            if maxv < tmpv
                maxi = i
                maxv = tmpv
            end
        end
        # Throw error if maximum value is 0. This happens only in case of
        # singular matrix
        if maxv == 0
            throw(ArgumentError("Input `A` was a singular matrix"))
        end
        # Swap j-th and maxi-th rows
        if maxi != j
            tmp = A[j, :]
            A[j, :] = A[maxi, :]
            A[maxi, :] = tmp
            tmp = basis[j]
            basis[j] = basis[maxi]
            basis[maxi] = tmp
        end
        # Update ipiv
        ipiv[j] = maxi
        # Normalize lower triangular part
        A[j+1:N, j] /= A[j, j]
        # Eliminate j-th row/column
        for i = j+1 : N
            for k = j+1 : r
                A[i, k] -= A[i, j] * A[j, k]
            end
        end
    end
    # Now solve C A[:r,:] = A[r+1:,:] and overwrite result to A[r+1:,:]
    # A is in LU form, so we need only one triangular solve
    # C L[:r,:] = L[r+1:,:]
    for i = r+1 : N
        # Since diagonal of L is an identity matrix, last column of C is equal
        # to last column of L[r+1:,:], so no need to update anything in that
        # column
        for j = r-1 : -1 : 1
            for k = j+1 : r
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
    # Now make A[:r,:] identity matrix, since A[:r,:]*inv(A[:r,:]) = I
    for i = 1 : r
        for j = 1 : r
            A[i, j] = 0
        end
        A[i, i] = 1
    end
    # Revert swaps, caused by LU
    for i = r : -1 : 1
        if ipiv[i] != i
            tmp = A[i, :]
            A[i, :] = A[ipiv[i], :]
            A[ipiv[i], :] = tmp
        end
    end
    # Exit if maxiters is 0
    if maxiters == 0
        return basis[1:r], 0
    end
    iter::Int = 0
    while true
        maxi, maxj, maxv = -1, -1, tol
        for i = 1 : N
            for j = 1 : r
                tmpv = abs(A[i, j])
                if tmpv > maxv
                    maxv = tmpv
                    maxi = i
                    maxj = j
                end
            end
        end
        if maxi != -1
            basis[maxj] = maxi
            tmp_row = A[maxi, :]
            tmp_row[maxj] -= 1
            tmp_col = A[:, maxj] / A[maxi, maxj]
            for i = 1 : N
                for j = 1 : r
                    A[i, j] -= tmp_row[j] * tmp_col[i]
                end
            end
        end
        iter += 1
        if (maxi == -1) || (iter == maxiters)
            return basis[1:r], iter
        end
    end
end

"""
    maxvol!(A, tol=1.05, maxiters=100)

Uses vendor-optimized LAPACK, provided by Julia.

Supports only 4 input types: `Float32` (single), `Float64` (double),
`ComplexF32` (single complex) and `ComplexF64` (double complex).

# Arguments:
- `A::Matrix{T}`: Input matrix on entry and output coefficients on exit. `T`
    must be one of `Float32`, `Float64`, `ComplexF32` and `ComplexF64`.
- `tol::Float64`: Stop when determinant growth is less or equal to this.
- `maxiters::Int`: Maximum number of iterations.

# Returns:
- `piv::Vector{Int}`: Indexes of pivoted rows
- `niters::Int`: Number of performed swap iterations

# Example:
```julia
julia> using Random, LinearAlgebra, Maxvol
julia> rng = MersenneTwister(100);
julia> A = rand(rng, Float64, 1000, 100);
julia> C = copy(A);
julia> piv, niters = maxvol!(C);
julia> norm(A-C*A[piv,:]) / norm(A)
2.3975097489579994e-15
julia> A = rand(rng, ComplexF32, 1000, 100);
julia> C = copy(A);
julia> piv, niters = maxvol!(C);
julia> norm(A-C*A[piv,:]) / norm(A)
2.0852597f-6
```

# See also: [`maxvol_generic!`](@ref)
"""
function maxvol!(A::Matrix{T}, tol::Float64=1.05, maxiters::Int=100) where
        T <: Union{Float32, Float64, ComplexF32, ComplexF64}
    # Check input parameters
    if tol < 1
        throw(ArgumentError("Parameter `tol` must be at least 1.0"))
    end
    if maxiters < 0
        throw(ArgumentError("Parameter `maxiters` must be non-negative"))
    end
    # Compute pivoted LU at first
    N, r = size(A)
    basis::Vector{typeof(N)} = 1 : N
    A, ipiv, info = getrf!(A)
    # Throw error if info is not 0. This happens only in case of
    # singular matrix
    if info != 0
        throw(ArgumentError("Input `A` was a singular matrix"))
    end
    # Compute basis rows
    for i = 1 : r
        if ipiv[i] != i
            tmp = basis[i]
            basis[i] = basis[ipiv[i]]
            basis[ipiv[i]] = tmp
        end
    end
    # Now solve C A[:r,:] = A[r+1:,:] and overwrite result to A[r+1:,:]
    # A is in LU form, so we need only one triangular solve
    # C L[:r,:] = L[r+1:,:]
    trsm!('R', 'L', 'N', 'U', one(T), view(A, 1:r, :), view(A, r+1:N, :))
    # Now make A[:r,:] identity matrix, since A[:r,:]*inv(A[:r,:]) = I
    # and revert swaps, caused by LU
    for i = r : -1 : 1
        if ipiv[i] != i
            A[i, :] = A[ipiv[i], :]
        end
        A[ipiv[i], :] .= 0
        A[ipiv[i], i] = 1
    end
    # Exit if maxiters is 0
    if maxiters == 0
        return basis[1:r], 0
    end
    iter::Int = 0
    while true
        tmpi = iamax(A)
        maxj, maxi = fldmod1(tmpi, size(A, 1))
        maxv = abs(A[tmpi])
        if maxv > tol
            basis[maxj] = maxi
            tmp_row = adjoint.(A[maxi, :])
            tmp_row[maxj] -= one(T)
            tmp_col = A[:, maxj] / A[maxi, maxj]
            alpha::T = -one(T)
            ger!(alpha, tmp_col, tmp_row, A)
        end
        iter += 1
        if (maxv <= tol) || (iter == maxiters)
            return basis[1:r], iter
        end
    end
end

end # module Maxvol

