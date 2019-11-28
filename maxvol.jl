import LinearAlgebra.LAPACK.getrf!
import LinearAlgebra.BLAS.trsm!
import LinearAlgebra.BLAS.ger!
import LinearAlgebra.BLAS.iamax

"""
Generic maxvol
"""
function maxvol_generic!(A::Matrix{T}, tol::Float64=1.05, maxiters::Int=100) where
        T <: Number
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
        return basis[1:r]
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
            return basis[1:r]
        end
    end
end

"""
LAPACK-based maxvol
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
        return basis[1:r]
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
            return basis[1:r]
        end
    end
end

