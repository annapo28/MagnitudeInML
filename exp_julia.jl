# mag_laplacian_speedup.jl
# ========================
# Julia-версия эксперимента: ускорение вычисления магнитуды
# через итеративные методы (Laplacian Paradigm)

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using BenchmarkTools
using Statistics
using CSV
using DataFrames
using Plots
using Random
using Distances
using Printf

println("Julia: Laplacian Paradigm для вычисления магнитуды")

"""
    zeta(X, t)
Вычисляет матрицу сходства ζ_ij = exp(-t * d(x_i, x_j))
"""
function zeta(X::Matrix{Float64}, t::Float64)
    n = size(X, 1)
    D = pairwise(Euclidean(), X, dims=1)
    return exp.(-t .* D)
end

"""
Прямое решение через LU-разложение
"""
function solve_direct(Z::Matrix{Float64})
    return Z \ ones(size(Z, 1))
end

"""
Сопряжённые градиенты без предобуславливания
"""
function solve_cg(Z::Matrix{Float64}, tol::Float64=1e-10, maxiter::Int=5000)
    n = size(Z, 1)
    b = ones(n)
    x, history = cg(Z, b; abstol=tol, reltol=tol, maxiter=maxiter, log=true)
    return x, history
end

"""
Preconditioned CG с диагональным preconditioner
"""
function solve_pcg(Z::Matrix{Float64}, tol::Float64=1e-10, maxiter::Int=5000)
    n = size(Z, 1)
    b = ones(n)
    M = Diagonal(1.0 ./ diag(Z))  # диагональный preconditioner
    x, history = cg(Z, b; Pl=M, abstol=tol, reltol=tol, maxiter=maxiter, log=true)
    return x, history
end

"""
Число обусловленности (для симметричной положительно определённой)
"""
function condition_number(Z::Matrix{Float64})
    evals = eigvals(Symmetric(Z))           # ← переименовали переменную
    evals = evals[evals .> 1e-14]
    return maximum(evals) / minimum(evals)
end

"""
Минимальное диагональное доминирование: min(Z_ii - Σ|Z_ij|, j≠i)
"""
function diagonal_dominance(Z::Matrix{Float64})
    n = size(Z, 1)
    dd = Float64[]
    for i in 1:n
        row_sum = sum(abs.(Z[i, :])) - abs(Z[i, i])
        push!(dd, Z[i, i] - row_sum)
    end
    return minimum(dd)
end

# Эксперимент 1 (число обусловленности и число итераций cg как функция t)
function exp_kappa_vs_t(n::Int=100, t_values=nothing, seed::Int=42)
    if t_values === nothing
        t_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    end
    
    Random.seed!(seed)
    X = randn(n, 2)
    
    records = Dict{String, Vector{Float64}}()
    records["t"] = Float64[]
    records["kappa"] = Float64[]
    records["iter_cg"] = Float64[]
    records["iter_pcg"] = Float64[]
    records["diag_dom"] = Float64[]
    
    for t in t_values
        println("  t = $t ...")
        Z = zeta(X, t)
        
        kappa = condition_number(Z)
        
        # CG
        _, hist_cg = solve_cg(Z)
        iter_cg = length(hist_cg.data[:resnorm])
        
        # PCG
        _, hist_pcg = solve_pcg(Z)
        iter_pcg = length(hist_pcg.data[:resnorm])
        
        dd = diagonal_dominance(Z)
        
        push!(records["t"], t)
        push!(records["kappa"], kappa)
        push!(records["iter_cg"], iter_cg)
        push!(records["iter_pcg"], iter_pcg)
        push!(records["diag_dom"], dd)
        
        @printf("  t=%5.1f | κ=%10.2f | iter_CG=%4d | iter_PCG=%4d | dd=%+.4f\n", 
                t, kappa, iter_cg, iter_pcg, dd)
    end
    
    return DataFrame(records)
end

# Эксперимент 2: время Direct vs CG vs PCG
function exp_speed_vs_t_n(sizes, t_values; n_rep::Int=5, seed::Int=42)
    Random.seed!(seed)
    
    records = Dict{String, Vector{Float64}}()
    records["n"] = Int[]
    records["t"] = Float64[]
    records["Direct"] = Float64[]
    records["CG"] = Float64[]
    records["PCG"] = Float64[]
    records["kappa"] = Float64[]
    records["iter_cg"] = Float64[]
    records["iter_pcg"] = Float64[]
    records["speedup_cg"] = Float64[]
    records["speedup_pcg"] = Float64[]
    
    for n in sizes
        println("  n = $n ...")
        X = randn(n, 2)
        
        for t in t_values
            Z = zeta(X, t)
            
            # Бенчмарки
            tm_direct = @belapsed solve_direct($Z) samples=n_rep
            tm_cg = @belapsed solve_cg($Z) samples=n_rep
            tm_pcg = @belapsed solve_pcg($Z) samples=n_rep
            _, hist_cg = solve_cg(Z)
            _, hist_pcg = solve_pcg(Z)
            
            # Извлекаем время из истории (последняя итерация)
            iter_cg = length(hist_cg.data[:resnorm])
            iter_pcg = length(hist_pcg.data[:resnorm])
            
            kappa = condition_number(Z)
            
            push!(records["n"], n)
            push!(records["t"], t)
            push!(records["Direct"], tm_direct)
            push!(records["CG"], tm_cg)
            push!(records["PCG"], tm_pcg)
            push!(records["kappa"], kappa)
            push!(records["iter_cg"], iter_cg)
            push!(records["iter_pcg"], iter_pcg)
            push!(records["speedup_cg"], tm_direct / tm_cg)
            push!(records["speedup_pcg"], tm_direct / tm_pcg)
        end
    end
    
    return DataFrame(records)
end

# Визуализация
function plot_results(df_kappa, df_speed)
    gr()  
    
    p1 = plot(df_kappa.t, df_kappa.kappa, 
              marker=:circle, linewidth=2, label="κ(ζ)",
              title="Число обусловленности",
              xlabel="t", ylabel="κ (log)",
              yscale=:log10, legend=:topright)
    
    p2 = plot(df_kappa.t, df_kappa.iter_cg, 
              marker=:circle, label="CG", linewidth=2)
    plot!(df_kappa.t, df_kappa.iter_pcg, 
          marker=:square, label="PCG", linewidth=2,
          title="Итерации до сходимости",
          xlabel="t", ylabel="Итерации")
    
    p3 = plot(df_kappa.t, df_kappa.diag_dom,
              marker=:circle, linewidth=2, label="diag_dom",
              title="Диагональное доминирование",
              xlabel="t", ylabel="min(Z_ii - Σ|Z_ij|)")
    hline!([0], linestyle=:dash, color=:red, label="порог")
    
    # Время vs t
    n_fix = unique(df_speed.n)[end ÷ 2]
    sub = df_speed[df_speed.n .== n_fix, :]
    
    p4 = plot(sub.t, sub.Direct, marker=:circle, label="Direct", linewidth=2)
    plot!(sub.t, sub.CG, marker=:circle, label="CG", linewidth=2)
    plot!(sub.t, sub.PCG, marker=:square, label="PCG", linewidth=2,
          title="Время vs t (n=$n_fix)",
          xlabel="t", ylabel="Время (с)")
    
    # Ускорение
    p5 = plot(sub.t, sub.speedup_cg, marker=:circle, label="CG", linewidth=2)
    plot!(sub.t, sub.speedup_pcg, marker=:square, label="PCG", linewidth=2,
          title="Ускорение vs t (n=$n_fix)",
          xlabel="t", ylabel="Speedup (×)")
    hline!([1.0], linestyle=:dash, color=:gray, label="")
    
    # Ускорение vs n
    t_big = maximum(df_speed.t)
    sub2 = df_speed[df_speed.t .== t_big, :]
    
    p6 = plot(sub2.n, sub2.speedup_cg, marker=:circle, label="CG", linewidth=2)
    plot!(sub2.n, sub2.speedup_pcg, marker=:square, label="PCG", linewidth=2,
          title="Ускорение vs n (t=$t_big)",
          xlabel="n", ylabel="Speedup (×)")
    hline!([1.0], linestyle=:dash, color=:gray, label="")
    
    # Сохраняем
    plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(1500, 800))
    savefig("mag_laplacian_julia_results.png")
    println("[✓] График сохранён: mag_laplacian_julia_results.png")
end


function main()
    T_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    SIZES = [50, 100, 150, 200, 300]
    
    println("\n▶ Эксперимент 1: число обуслвленности и итерации CG vs t (n=100)...")
    df_kappa = exp_kappa_vs_t(100, T_VALUES)
    CSV.write("mag_kappa_julia.csv", df_kappa)
    
    println("\n▶ Эксперимент 2: скорость Direct vs CG vs PCG...")
    df_speed = exp_speed_vs_t_n(SIZES, T_VALUES, n_rep=5)
    CSV.write("mag_speed_julia.csv", df_speed)
    
    println("\n▶ Графики...")
    plot_results(df_kappa, df_speed)
    
    # Итоговые таблицы
    println("\n" * "="^70)
    println("Таблица 1: κ(ζ), итерации и диагональное доминирование (n=100)")
    println("="^70)
    @printf("%6s | %12s | %8s | %9s | %8s\n", "t", "κ(ζ)", "iter CG", "iter PCG", "dd")
    println("-"^70)
    for row in eachrow(df_kappa)
        @printf("%6.1f | %12.1f | %8d | %9d | %+.4f\n", 
                row.t, row.kappa, row.iter_cg, row.iter_pcg, row.diag_dom)
    end
    
    println("\n" * "="^70)
    println("Таблица 2: ускорение при t=10.0")
    println("="^70)
    sub = df_speed[df_speed.t .== 10.0, :]
    @printf("%5s | %8s | %8s | %8s | %7s | %8s\n", 
            "n", "Direct", "CG", "PCG", "sp_CG", "sp_PCG")
    println("-"^70)
    for row in eachrow(sub)
        @printf("%5d | %8.4f | %8.4f | %8.4f | %6.2f× | %7.2f×\n",
                row.n, row.Direct, row.CG, row.PCG, row.speedup_cg, row.speedup_pcg)
    end
    println("="^70)
end

main()