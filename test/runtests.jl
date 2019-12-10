# @copyright (c) 2019 RWTH Aachen. All rights reserved.
#
# @file test/runtests.jl
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2019-12-10

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

