module KernelLearning

using LinearAlgebra
using Base.Threads
using Random


export rbf_kernel_gen, ridge_regression_kernel, predict_kernel_model, kernel_matrix, predict_kernel_model, random_features_transform
export ridge_regression_random_features, predict_random_features_model,rbf_random_features_parameter,rbf_random_features

"""
    rbf_kernel(x, y, σ)
Generate RBF kernel for given inputs `x` and `y` with bandwidth `σ`.
"""

function rbf_kernel_gen(σ)
    function rbf_kernel(x, y)
        return exp(-sum((x-y).^2) / (2σ^2))
    end
    return rbf_kernel
end
"""
    kernel_matrix(X, kernel_function, σ)
Generate a kernel matrix for dataset `X` using provided `kernel_function` and parameter `σ`.
"""
function kernel_matrix(X, kernel_function)
    n = size(X, 1)
    K = zeros(n, n)
    
    Threads.@threads for i in 1:n
        for j in 1:i  # Use symmetry to avoid redundant calculations
            K[i, j] = kernel_function(X[i, :,:], X[j, :,:])
            K[j, i] = K[i, j]  # Exploit symmetry
        end
    end
    
    return K
end

function kernel_matrix(X,Y, kernel_function)
    n = size(X, 1)
    m= size(Y,1)
    K = zeros(n, m)
    
    Threads.@threads for i in 1:n
        for j in 1:m  
            K[i, j] = kernel_function(X[i, :,:], Y[j, :,:])
        end
    end
    
    return K
end

"""
    ridge_regression_kernel(K, y, λ)
Train Ridge Regression with kernel matrix `K`, target `y` and regularization `λ`.
Returns the alpha coefficients.
"""
function ridge_regression_kernel(K, y, λ)
    return (K + λ * I)\ y
end

"""
    predict(K, α)
Predict with kernel matrix `K` and alpha coefficients `α`.
"""

function predict_kernel_model(kernel_function, α, X, Y )
    return kernel_matrix(X, Y, kernel_function) * α
end

function rbf_random_features_parameter(d, D, σ)
    
    w = randn(D, d) / σ
    b = 2π * rand(D)
    
    return (weights=w, bias=b)
end

function rbf_random_features(X::AbstractMatrix, D::Int, params::NamedTuple)
    w=params.weights
    b=params.bias
    return sqrt(2/D) * cos.(X * transpose(w) .+ b')
end

function random_features_transform(X::AbstractMatrix, transform_func::Function, params::NamedTuple)
    return transform_func(X, params)
end

function ridge_regression_random_features(Z::AbstractMatrix, y::Vector, λ::Float64=1.0)
    return (transpose(Z) * Z + λ * I) \ (transpose(Z) * y)  # Solves the system of linear equations using backslash operator
end

function predict_random_features_model(Z::AbstractMatrix, weights::Vector)
    return Z * weights
end



end  # module