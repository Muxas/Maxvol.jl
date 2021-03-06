# @copyright (c) 2019-2020 RWTH Aachen. All rights reserved.
#
# @file test/runtests.jl
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2020-02-06

using Maxvol, LinearAlgebra, Random, Test

# Size and number of test matrices
nrows = 100
ncols = 20
ntests = 50

# Parameters for maxvol
tol = 1.05
maxiters = 100
# Set random seed
rng = Random.MersenneTwister(100)

# Lapack-based Maxvol
println("Testing LAPACK-based Maxvol")
types = [Float32, Float64, ComplexF32, ComplexF64]
for T in types
    println("    ", T)
    # Generate orthogonal/unitary input
    A = Matrix(qr(randn(rng, T, nrows, ncols)).Q)
    B = copy(A)
    piv, niters = Maxvol.maxvol!(B, tol, maxiters)
    @test A ≈ B*A[piv,:]
    # Lapack-based maxvol in complex case uses izamax(), which returns maximum
    # of |re|+|im|, not |re|^2+|im|^2. So we do not check Inf norm, but
    # absolute value of matrix entry with maximum 1-norm.
    if T <: AbstractFloat
        @test norm(B, Inf) <= tol
    else
        C = abs.(real(B)) + abs.(imag(B))
        maxv, maxi = findmax(C)
        @test abs(B[maxi]) <= tol
    end
end

# Lapack-based Maxvol does not exist for certain types
types = [Float16, BigFloat, ComplexF16, Complex{BigFloat}, Int]
for T in types
    println("    ", T)
    A = zeros(T, nrows, ncols)
    @test_throws MethodError Maxvol.maxvol!(A, tol, maxiters)
end

# Lapack-based Maxvol throws ArgumentError if input is singular matrix
types = [Float32, Float64, ComplexF32, ComplexF64]
for T in types
    println("    ", T)
    A = ones(T, nrows, ncols)
    @test_throws ArgumentError Maxvol.maxvol!(A, tol, maxiters)
end

# Generic Maxvol
println("Testing Generic Maxvol")
types = [Float16, Float32, Float64, BigFloat, ComplexF16, ComplexF32,
         ComplexF64, Complex{BigFloat}]
for T in types
    println("    ", T)
    A = Matrix(qr(rand(rng, T, nrows, ncols)).Q)
    B = copy(A)
    piv, niters = Maxvol.maxvol_generic!(B, tol, maxiters)
    @test A ≈ B*A[piv,:]
    # Check that matrix B is upperbounded
    @test norm(B, Inf) <= tol
end

# Generic Maxvol throws ArgumentError if input is singular matrix
types = [Float16, Float32, Float64, BigFloat, ComplexF16, ComplexF32,
         ComplexF64, Complex{BigFloat}]
for T in types
    println("    ", T)
    A = ones(T, nrows, ncols)
    @test_throws ArgumentError Maxvol.maxvol_generic!(A, tol, maxiters)
end

# Parameters for rect_maxvol
tol = 1.0
# Set random seed
rng = Random.MersenneTwister(100)

# Lapack-based Rect_maxvol
println("Testing LAPACK-based Rect_maxvol")
types = [Float32, Float64, ComplexF32, ComplexF64]
for T in types
    println("    ", T)
    # Generate orthogonal/unitary input
    A = Matrix(qr(randn(rng, T, nrows, ncols)).Q)
    piv, C = Maxvol.rect_maxvol(A, tol)
    @test A ≈ C*A[piv,:]
    # Check that norm of each row of C is less than tol
    @test all([norm(C[i,:]) <= tol for i=1:nrows])
end

# Lapack-based Rect_maxvol does not exist for certain types
types = [Float16, BigFloat, ComplexF16, Complex{BigFloat}, Int]
for T in types
    println("    ", T)
    A = zeros(T, nrows, ncols)
    @test_throws MethodError Maxvol.rect_maxvol(A, tol)
end

# Lapack-based Rect_maxvol throws ArgumentError if input is singular matrix
types = [Float32, Float64, ComplexF32, ComplexF64]
for T in types
    println("    ", T)
    A = ones(T, nrows, ncols)
    @test_throws ArgumentError Maxvol.rect_maxvol(A, tol)
end

# Generic Rect_maxvol
println("Testing Generic Rect_maxvol")
types = [Float16, Float32, Float64, BigFloat, ComplexF16, ComplexF32,
         ComplexF64, Complex{BigFloat}]
for T in types
    println("    ", T)
    A = Matrix(qr(rand(rng, T, nrows, ncols)).Q)
    piv, C = Maxvol.rect_maxvol_generic(A, tol)
    @test A ≈ C*A[piv,:]
    # Check that norm of each row of C is less than tol
    @test all([norm(C[i,:]) <= tol for i=1:nrows])
end

# Generic Rect_maxvol throws ArgumentError if input is singular matrix
types = [Float16, Float32, Float64, BigFloat, ComplexF16, ComplexF32,
         ComplexF64, Complex{BigFloat}]
for T in types
    println("    ", T)
    A = ones(T, nrows, ncols)
    @test_throws ArgumentError Maxvol.rect_maxvol_generic(A, tol)
end

