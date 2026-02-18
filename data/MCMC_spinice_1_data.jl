using Random
using Printf

# ============================================================
# Kagome neighbor list
# ============================================================
function kagome_neighbors(Lx, Ly)
    N = 3Lx*Ly
    nbr = [Int[] for _ in 1:N]

    idx(x,y,s) = 1 + s + 3*(mod(x,Lx) + Lx*mod(y,Ly))

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

# ============================================================
# Metropolis update
# ============================================================
@inline function deltaE_flip(spins, nbr, i, J)
    si = spins[i]
    ssum = 0
    @inbounds for j in nbr[i]
        ssum += spins[j]
    end
    return -2.0 * J * si * ssum
end

function metropolis_sweep!(rng, spins, nbr, β, J)
    N = length(spins)
    @inbounds for _ in 1:N
        i = rand(rng, 1:N)
        dE = deltaE_flip(spins, nbr, i, J)
        if dE ≤ 0 || rand(rng) < exp(-β*dE)
            spins[i] = -spins[i]
        end
    end
end

# ============================================================
# Writers
# ============================================================
@inline spin_to_binary(s::Int8) = (s == 1) ? UInt8(1) : UInt8(0)

"""
write_configuration(io, spins; format=:binary or :pm)

- :binary  -> writes 0/1 (UInt8) for RBM
- :pm      -> writes -1/+1 (Int8) for physics
"""
function write_configuration(io, spins; format::Symbol=:binary)
    N = length(spins)

    @inbounds for i in 1:N
        if format === :binary
            print(io, spin_to_binary(spins[i]))
        elseif format === :pm
            print(io, spins[i])  # -1 or +1
        else
            error("Unknown format=$format. Use :binary or :pm.")
        end
        i < N && print(io, ' ')
    end

    print(io, '\n')
end

# ============================================================
# MAIN GENERATOR
# ============================================================
function generate_kagome_dataset(; 
        Lx=16,
        Ly=16,
        T=1e-10,
        J=1.0,
        n_therm=2000,
        save_every=100,
        n_configs=5000,
        outfile_bin="X_train_bin.txt",
        outfile_pm=nothing,          # set to a path to also save ±1
        seed=42
    )

    rng = MersenneTwister(seed)

    nbr = kagome_neighbors(Lx, Ly)

    N = 3Lx*Ly
    spins = rand(rng, Int8[-1,1], N)

    β = 1/T

    println("Thermalizing...")
    for _ in 1:n_therm
        metropolis_sweep!(rng, spins, nbr, β, J)
    end

    println("Generating configurations...")

    io_bin = open(outfile_bin, "w")
    io_pm  = (outfile_pm === nothing) ? nothing : open(outfile_pm, "w")

    saved = 0
    total_sweeps = 0

    while saved < n_configs
        for _ in 1:save_every
            metropolis_sweep!(rng, spins, nbr, β, J)
            total_sweeps += 1
        end

        write_configuration(io_bin, spins; format=:binary)
        if io_pm !== nothing
            write_configuration(io_pm, spins; format=:pm)
        end

        saved += 1
        @printf("Saved %d / %d\n", saved, n_configs)
    end

    close(io_bin)
    io_pm !== nothing && close(io_pm)

    println("\nDONE.")
    println("Saved binary dataset → ", outfile_bin)
    if outfile_pm !== nothing
        println("Saved ±1 dataset     → ", outfile_pm)
    end
end

# ============================================================
# RUN
# ============================================================
generate_kagome_dataset(
    Lx=32,
    Ly=32,
    T=1e-10,
    J=1.0,
    n_therm=2000,
    save_every=100,
    n_configs=5000,
    outfile_pm="Your directory/MC_L=32.txt",
    seed=42
)
