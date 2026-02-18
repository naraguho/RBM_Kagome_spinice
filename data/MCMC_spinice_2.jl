using Random
using Printf
using Plots

# ============================================================
# Kagome NN neighbor list (graph edges r=1)
# ============================================================
function kagome_neighbors(Lx::Int, Ly::Int)
    N = 3 * Lx * Ly
    nbr = [Int[] for _ in 1:N]

    idx(x,y,s) = 1 + s + 3*(mod(x,Lx) + Lx*mod(y,Ly))  # 1-based

    for x in 0:Lx-1, y in 0:Ly-1, s in 0:2
        i = idx(x,y,s)

        if s == 0
            push!(nbr[i], idx(x,y,1))
            push!(nbr[i], idx(x,y,2))
            push!(nbr[i], idx(x-1,y,1))
            push!(nbr[i], idx(x,y-1,2))
        elseif s == 1
            push!(nbr[i], idx(x,y,0))
            push!(nbr[i], idx(x,y,2))
            push!(nbr[i], idx(x+1,y,0))
            push!(nbr[i], idx(x+1,y-1,2))
        else
            push!(nbr[i], idx(x,y,0))
            push!(nbr[i], idx(x,y,1))
            push!(nbr[i], idx(x,y+1,0))
            push!(nbr[i], idx(x-1,y+1,1))
        end
    end

    return nbr
end

# ------------------------------------------------------------
# BFS shells from a reference site:
# shells[d+1] = sites at graph distance d, d=0..rmax
# ------------------------------------------------------------
function graph_distance_shells(ref::Int, nbr; rmax::Int)
    N = length(nbr)

    dist  = fill(-1, N)
    queue = Vector{Int}(undef, N)

    head = 1
    tail = 1
    queue[1] = ref
    dist[ref] = 0

    while head <= tail
        v = queue[head]; head += 1
        dv = dist[v]
        dv == rmax && continue

        for w in nbr[v]
            if dist[w] == -1
                dist[w] = dv + 1
                tail += 1
                queue[tail] = w
            end
        end
    end

    shells = [Int[] for _ in 0:rmax]
    for i in 1:N
        d = dist[i]
        if 0 <= d <= rmax
            push!(shells[d+1], i)
        end
    end

    return shells
end

# ------------------------------------------------------------
# Build shells for ALL sites: shells_all[r][i] = vector of neighbors at dist r
# where r = 1..rmax
# ------------------------------------------------------------
function all_shells(nbr; rmax::Int)
    N = length(nbr)
    shells_all = [ [Int[] for _ in 1:N] for _ in 1:rmax ]  # shells_all[1]=r1, etc.

    dist  = fill(-1, N)
    queue = Vector{Int}(undef, N)

    @inbounds for ref in 1:N
        fill!(dist, -1)

        head = 1
        tail = 1
        queue[1] = ref
        dist[ref] = 0

        while head <= tail
            v = queue[head]; head += 1
            dv = dist[v]
            dv == rmax && continue

            for w in nbr[v]
                if dist[w] == -1
                    dist[w] = dv + 1
                    tail += 1
                    queue[tail] = w
                end
            end
        end

        for i in 1:N
            d = dist[i]
            if 1 <= d <= rmax
                push!(shells_all[d][ref], i)
            end
        end
    end

    return shells_all
end

# ============================================================
# Hamiltonian convention 
#   H = Σ_r J_r Σ_{<ij>_r} s_i s_j
# Important : second, thrid nearest neighbor --> second neighbor in graph
# Single flip ΔE:
#   ΔE = -2 s_i Σ_r J_r Σ_{j∈N_r(i)} s_j
# ============================================================
@inline function deltaE_flip_J123(spins::Vector{Int8}, shells_all, i::Int,
                                  J1::Float64, J2::Float64, J3::Float64)
    si = spins[i]
    s1 = 0; s2 = 0; s3 = 0
    @inbounds for j in shells_all[1][i]; s1 += spins[j]; end
    @inbounds for j in shells_all[2][i]; s2 += spins[j]; end
    @inbounds for j in shells_all[3][i]; s3 += spins[j]; end
    return -2.0 * si * (J1*s1 + J2*s2 + J3*s3)
end

function metropolis_sweep_J123!(rng::AbstractRNG, spins::Vector{Int8}, shells_all,
                               β::Float64, J1::Float64, J2::Float64, J3::Float64)
    N = length(spins)
    @inbounds for _ in 1:N
        i = rand(rng, 1:N)
        dE = deltaE_flip_J123(spins, shells_all, i, J1, J2, J3)
        if dE ≤ 0 || rand(rng) < exp(-β * dE)
            spins[i] = -spins[i]
        end
    end
    return nothing
end

# ------------------------------------------------------------
# Optional: compute total energy (for sanity checks / monitoring)
# Using j>i to count each pair once
# ------------------------------------------------------------
function total_energy(spins::Vector{Int8}, shells_all, J1::Float64, J2::Float64, J3::Float64)
    N = length(spins)
    E = 0.0
    Js = (J1, J2, J3)
    @inbounds for r in 1:3
        Jr = Js[r]
        for i in 1:N
            si = spins[i]
            for j in shells_all[r][i]
                if j > i
                    E += Jr * si * spins[j]
                end
            end
        end
    end
    return E
end

# ----------------------------
# index helper (same convention as your code)
# ----------------------------
@inline function idx_site(Lx::Int, Ly::Int, x::Int, y::Int, s::Int)
    return 1 + s + 3*(mod(x, Lx) + Lx*mod(y, Ly))
end

# ----------------------------
# kagome positions (same embedding you used)
# ----------------------------
function kagome_positions(Lx::Int, Ly::Int)
    N = 3 * Lx * Ly
    xs = zeros(Float64, N)
    ys = zeros(Float64, N)

    a1x, a1y = 1.0, 0.0
    a2x, a2y = 0.5, sqrt(3)/2

    bx = (0.0, 0.5, 0.25)
    by = (0.0, 0.0, sqrt(3)/4)

    @inbounds for y in 0:Ly-1, x in 0:Lx-1, s in 0:2
        i  = 1 + s + 3*(x + Lx*y)
        Rx = x*a1x + y*a2x
        Ry = x*a1y + y*a2y
        xs[i] = Rx + bx[s+1]
        ys[i] = Ry + by[s+1]
    end
    return xs, ys
end

# ----------------------------
# For the diagnostic : Plot only UP triangles (unit-cell triangles)
# ----------------------------
function plot_up_triangle_sums(spins::Vector{Int8}, Lx::Int, Ly::Int;
                               outpath::Union{Nothing,String}=nothing,
                               title_str::String="")

    xs, ys = kagome_positions(Lx, Ly)
    N = length(spins)

    up_idx = findall(==(Int8(1)), spins)
    dn_idx = findall(==(Int8(-1)), spins)

    plt = scatter(xs[dn_idx], ys[dn_idx];
        color=:blue, markersize=6, markerstrokewidth=0,
        legend=false, aspect_ratio=:equal, grid=false,
        xlabel="x", ylabel="y",
        title = title_str == "" ? @sprintf("Up-triangle sums (unit-cell) | Lx=%d Ly=%d", Lx, Ly) : title_str
    )
    scatter!(plt, xs[up_idx], ys[up_idx];
        color=:red, markersize=6, markerstrokewidth=0
    )

    # annotate Q_up(x,y) at centroid of the three sites in the unit cell
    @inbounds for y in 0:Ly-1, x in 0:Lx-1
        i0 = idx_site(Lx, Ly, x, y, 0)
        i1 = idx_site(Lx, Ly, x, y, 1)
        i2 = idx_site(Lx, Ly, x, y, 2)

        Q = Int(spins[i0] + spins[i1] + spins[i2])

        xc = (xs[i0] + xs[i1] + xs[i2]) / 3
        yc = (ys[i0] + ys[i1] + ys[i2]) / 3

        annotate!(plt, xc, yc, text(string(Q), 10, :black))
    end

    if outpath !== nothing
        savefig(plt, outpath)
        println("Saved plot → $outpath")
    end

    return plt
end


# ============================================================
# MAIN MCMC DRIVER (with plotting every meas_every sweeps)
# ============================================================
function run_kagome_J123_mcmc(;
        Lx::Int = 8,
        Ly::Int = 8,
        T::Float64 = 0.2,
        J1::Float64 = +1.0,
        J2::Float64 = -0.1,
        J3::Float64 = 0.0,
        n_therm::Int = 2_000,
        n_sweeps::Int = 20_000,
        meas_every::Int = 200,
        seed::Int = 42,
        plot_every::Int = 200,
        plot_dir::String = "plots"
    )

    isdir(plot_dir) || mkpath(plot_dir)

    rng = MersenneTwister(seed)

    nbr = kagome_neighbors(Lx, Ly)
    shells_all = all_shells(nbr; rmax=3)

    N = 3 * Lx * Ly
    spins = rand(rng, Int8[-1, 1], N)

    β = 1.0 / T

    println("Kagome MCMC: Lx=$Lx Ly=$Ly N=$N")
    println("H = Σ_r J_r Σ_{<ij>_r} s_i s_j   (J>0 AF, J<0 ferro)")
    println("(J1,J2,J3)=($J1,$J2,$J3),  T=$T, β=$β")

    println("Thermalizing ($n_therm sweeps)...")
    for _ in 1:n_therm
        metropolis_sweep_J123!(rng, spins, shells_all, β, J1, J2, J3)
    end

    println("Running ($n_sweeps sweeps), measuring every $meas_every sweeps, plotting every $plot_every sweeps...")

    for sweep in 1:n_sweeps
        metropolis_sweep_J123!(rng, spins, shells_all, β, J1, J2, J3)
        if sweep == 20000
            for i in 1:N
                deltaE = deltaE_flip_J123(spins, shells_all, i, J1, J2, J3)
                println("i = $i, deltaE = $deltaE")
            end
        end

        if sweep % meas_every == 0
            E = total_energy(spins, shells_all, J1, J2, J3)
            @printf("sweep %d | E/N = %.6f\n", sweep, E / N)
        end

        if sweep % plot_every == 0
            outpng = joinpath(plot_dir, @sprintf("up_tri_%07d.png", sweep))
            title_str = @sprintf("Up-triangle sums | sweep %d", sweep)
            plt = plot_up_triangle_sums(spins, Lx, Ly; outpath=outpng, title_str=title_str)
            display(plt)
        end
        
        
    end

    println("\nDONE.")
    return spins
end
function save_spins_txt(path::String, spins::Vector{Int8})
    open(path, "w") do io
        println(io, "# i   s_i")
        for (i, s) in enumerate(spins)
            @printf(io, "%d  %d\n", i, Int(s))
        end
    end
    println("Saved final spins → $path")
end


Lx, Ly = 24, 24

spins_final = run_kagome_J123_mcmc(
    Lx=Lx, Ly=Ly,
    T=1e-10,
    J1=+1.0,
    J2=-0.1,
    J3=-0.0, #Again, second, third nearest neighbor is second neighbor in graph-wise.
    n_therm=2000,
    n_sweeps=20000,
    meas_every=20000,
    plot_every=20000,
    seed=43,
    plot_dir="plots_kagome_24"
)

save_spins_txt("spins_final_try_24.txt", spins_final)
