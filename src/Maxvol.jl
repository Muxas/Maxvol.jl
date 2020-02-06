# @copyright (c) 2019-2020 RWTH Aachen. All rights reserved.
#
# @file src/Maxvol.jl
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2020-02-06

module Maxvol

using LinearAlgebra: I, norm
using LinearAlgebra.LAPACK: getrf!
using LinearAlgebra.BLAS: trsm!, ger!, iamax

export maxvol_generic!, maxvol!, rect_maxvol_generic, rect_maxvol

"""
    maxvol_generic!(A, tol=1.05, maxiters=100)

Generic maxvol method, that does not use any **BLAS** or **LAPACK** calls.

Finds good square submatrix. Uses greedy iterative maximization of 1-volume to
find good `r`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of rank `r`.
Returns good submatrix and coefficients of expansion (`N`-by-`r` matrix) of
rows of matrix `A` by rows of good submatrix.

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

Finds good square submatrix. Uses greedy iterative maximization of 1-volume to
find good `r`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of rank `r`.
Returns good submatrix and coefficients of expansion (`N`-by-`r` matrix) of
rows of matrix `A` by rows of good submatrix.

Uses vendor-optimized LAPACK, provided by Julia. Supports only 4 input types:
`Float32` (single), `Float64` (double), `ComplexF32` (single complex) and
`ComplexF64` (double complex).

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

"""
    rect_maxvol_generic(A, tol=1.0, maxK=size(A,1), min_add_K=0,
        min_K=size(A,2), start_maxvol_iters=10, identity_submatrix=true)

Generic rect_maxvol method, that does not call **BLAS** or **LAPACK** directly.

Finds good rectangular submatrix. Uses greedy iterative maximization of
2-volume to find good `K`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of
rank `r`. Returns good submatrix and least squares coefficients of expansion
(`N`-by-`K` matrix) of rows of matrix `A` by rows of good submatrix.

Can be used for special arithmetics, like `BigFloat`, `Rational` or
`Complex{Float16}`.

# Arguments:
- `A::Matrix`: Input matrix on entry.
- `tol::Float64`: Stop when volume growth is less or equal to this.
- `maxK::Int`: Maximum number of rows in good submatrix.
- `minK::Int`: Minimum number of rows in good submatrix.
- `min_add_K::Int`: Minimum number of rows to add to the square submatrix.
        Resulting good matrix will have minimum of `r+min_add_K` rows.
- `start_maxvol_iters::Int`: How many iterations of square maxvol (optimization
        of 1-volume) is required to be done before actual rectangular 2-volume
        maximization.
- `identity_submatrix::Bool`: Coefficients of expansions are computed as least
        squares solution. If `identity_submatrix` is True, returned matrix of
        coefficients will have submatrix, corresponding to good rows,
        set to identity.

# Returns:
- `piv::Vector{Int}`: Indexes of pivoted rows
- `C::Matrix`: N-by-size(piv) matrix of coefficients of expansion of all rows
        of A by ``good'' rows `piv`.

# Example:
```julia
julia> using Random, LinearAlgebra, Maxvol
julia> rng = MersenneTwister(100);
julia> A = rand(rng, BigFloat, 1000, 100);
julia> piv, C = rect_maxvol_generic(A);
julia> norm(A-C*A[piv,:]) / norm(A)
1.487413551535496062888389670275572427766752748825675538705145867354156169433079e-76
julia> A = rand(rng, ComplexF64, 1000, 100);
julia> piv, C = rect_maxvol_generic(A);
julia> norm(A-C*A[piv,:]) / norm(A)
3.5140903782712447e-15
```

# See also: [`rect_maxvol`](@ref)
"""
function rect_maxvol_generic(A::Matrix{T}, tol::Float64=1.0,
        maxK::Int=size(A,1), min_add_K::Int=0, minK::Int=size(A,2),
        start_maxvol_iters::Int=10, identity_submatrix::Bool=true) where
        T <: Number
    # Square of tolerance
    tol2 = tol^2
    # N - number of rows, r - number of columns of matrix A
    N, r = size(A)
    # some work on parameters
    if N ≤ r
        return Vector{Int}(1:N), Matrix{T}(I, N, N)
    end
    if maxK > N
        maxK = N
    end
    if maxK < r
        maxK = r
    end
    if minK < r
        minK = r
    end
    if minK > N
        minK = N
    end
    if min_add_K > 0
        minK = max(minK, r+min_add_K)
    end
    if minK > maxK
        minK = maxK
        #raise ValueError('minK value cannot be greater than maxK value')
    end
    # choose initial submatrix and coefficients according to maxvol
    # algorithm
    index = zeros(Int, N)
    chosen = falses(N)
    C = copy(A)
    tmp_index, _ = maxvol_generic!(C, 1.05, start_maxvol_iters)
    # Now matrix A contains coefficients of linear dependancy of old A on
    # pivoted A[tmp_index, :]
    index[1:r] = tmp_index
    chosen[tmp_index] .= true
    # compute square 2-norms of each row in coefficients matrix C
    row_norm_sqr = [chosen[i] ? 0 : norm(C[i, :])^2 for i = 1 : N]
    # find maximum value in row_norm_sqr
    i = argmax(row_norm_sqr)
    K = r
    # augment maxvol submatrix with each iteration
    while (row_norm_sqr[i] > tol2 && K < maxK) || K < minK
        # add i to index and recompute C and square norms of each row
        # by SVM-formula
        K += 1
        index[K] = i
        chosen[i] = true
        c = copy(C[i:i, :])
        #println(size(C), ' ', size(c), ' ', size(c'))
        v = C * (c')
        l::real(T) = 1.0 / (1+v[i]) # v[i] is always real
        c .*= l
        for j = 1 : N
            C[j, :] -= v[j] * c[1, :]
        end
        C = [C l*v]
        #C = [[(C[i, :]-v[i]*(l*c)) v[i]] for i = 1 : N]
        #ger(-l,v,c,a=C,overwrite_a=1)
        #C = np.hstack([C, l*v.reshape(-1,1)])
        row_norm_sqr -= [chosen[i] ? 0 : l*abs(v[i])^2 for i = 1 : N]
        row_norm_sqr[i] = 0
        #row_norm_sqr -= (l*v[:N]*v[:N].conj()).real
        #row_norm_sqr *= chosen
        # find maximum value in row_norm_sqr
        i = argmax(row_norm_sqr)
    end
    # parameter identity_submatrix is True, set submatrix,
    # corresponding to maxvol rows, equal to identity matrix
    if identity_submatrix
        C[index[1:K], :] = Matrix{T}(I, K, K)
    end
    return index[1:K], C
end

"""
    rect_maxvol(A, tol=1.0, maxK=size(A,1), min_add_K=0, min_K=size(A,2),
        start_maxvol_iters=10, identity_submatrix=true)

Finds good rectangular submatrix. Uses greedy iterative maximization of
2-volume to find good `K`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of
rank `r`. Returns good submatrix and least squares coefficients of expansion
(`N`-by-`K` matrix) of rows of matrix `A` by rows of good submatrix.

Uses vendor-optimized LAPACK, provided by Julia. Supports only 4 input types:
`Float32` (single), `Float64` (double), `ComplexF32` (single complex) and
`ComplexF64` (double complex).

# Arguments:
- `A::Matrix`: Input matrix on entry.
- `tol::Float64`: Stop when volume growth is less or equal to this.
- `maxK::Int`: Maximum number of rows in good submatrix.
- `minK::Int`: Minimum number of rows in good submatrix.
- `min_add_K::Int`: Minimum number of rows to add to the square submatrix.
        Resulting good matrix will have minimum of `r+min_add_K` rows.
- `start_maxvol_iters::Int`: How many iterations of square maxvol (optimization
        of 1-volume) is required to be done before actual rectangular 2-volume
        maximization.
- `identity_submatrix::Bool`: Coefficients of expansions are computed as least
        squares solution. If `identity_submatrix` is True, returned matrix of
        coefficients will have submatrix, corresponding to good rows,
        set to identity.

# Returns:
- `piv::Vector{Int}`: Indexes of pivoted rows
- `C::Matrix`: N-by-size(piv) matrix of coefficients of expansion of all rows
        of A by ``good'' rows `piv`.

# Example:
```julia
julia> using Random, LinearAlgebra, Maxvol
julia> rng = MersenneTwister(100);
julia> A = rand(rng, Float64, 1000, 100);
julia> piv, C = rect_maxvol(A);
julia> norm(A-C*A[piv,:]) / norm(A)
1.8426360389674326e-15
julia> A = rand(rng, ComplexF32, 1000, 100);
julia> piv, C = rect_maxvol(C);
julia> norm(A-C*A[piv,:]) / norm(A)
1.8014168f-6
```

# See also: [`rect_maxvol_generic`](@ref)
"""
function rect_maxvol(A::Matrix{T}, tol::Float64=1.0,
        maxK::Int=size(A, 1), min_add_K::Int=0, minK::Int=size(A, 2),
        start_maxvol_iters::Int=10, identity_submatrix::Bool=true) where
        T <: Union{Float32, Float64, ComplexF32, ComplexF64}
    # Square of tolerance
    tol2 = tol^2
    # N - number of rows, r - number of columns of matrix A
    N, r = size(A)
    # some work on parameters
    if N ≤ r
        return Vector{Int}(1:N), Matrix{T}(I, N, N)
    end
    if maxK > N
        maxK = N
    end
    if maxK < r
        maxK = r
    end
    if minK < r
        minK = r
    end
    if minK > N
        minK = N
    end
    if min_add_K > 0
        minK = max(minK, r+min_add_K)
    end
    if minK > maxK
        minK = maxK
        #raise ValueError('minK value cannot be greater than maxK value')
    end
    # choose initial submatrix and coefficients according to maxvol
    # algorithm
    index = zeros(Int, N)
    chosen = falses(N)
    C = copy(A)
    tmp_index, _ = maxvol!(C, 1.05, start_maxvol_iters)
    # Now matrix A contains coefficients of linear dependancy of old A on
    # pivoted A[tmp_index, :]
    index[1:r] = tmp_index
    chosen[tmp_index] .= true
    # compute square 2-norms of each row in coefficients matrix C
    row_norm_sqr = Vector{real(T)}(undef, N)
    row_norm_sqr[:] = [chosen[i] ? 0 : norm(C[i, :])^2 for i = 1 : N]
    # find maximum value in row_norm_sqr
    i = argmax(row_norm_sqr)
    K = r
    # augment maxvol submatrix with each iteration
    while (row_norm_sqr[i] > tol2 && K < maxK) || K < minK
        # add i to index and recompute C and square norms of each row
        # by SVM-formula
        K += 1
        index[K] = i
        chosen[i] = true
        c = Vector{T}(undef, K-1)
        c[:] = C[i, :]'
        v = C * c
        l::real(T) = 1.0 / (1+v[i]) # v[i] is always real
        ger!(T(-l), v, c, C)
        C = [C l*v]
        #C = [[(C[i, :]-v[i]*(l*c)) v[i]] for i = 1 : N]
        #ger(-l,v,c,a=C,overwrite_a=1)
        #C = np.hstack([C, l*v.reshape(-1,1)])
        row_norm_sqr -= [chosen[i] ? 0 : l*abs(v[i])^2 for i = 1 : N]
        row_norm_sqr[i] = 0
        #row_norm_sqr -= (l*v[:N]*v[:N].conj()).real
        #row_norm_sqr *= chosen
        # find maximum value in row_norm_sqr
        i = argmax(row_norm_sqr)
    end
    # parameter identity_submatrix is True, set submatrix,
    # corresponding to maxvol rows, equal to identity matrix
    if identity_submatrix
        C[index[1:K], :] = Matrix{T}(I, K, K)
    end
    return index[1:K], C
end

end # module Maxvol

