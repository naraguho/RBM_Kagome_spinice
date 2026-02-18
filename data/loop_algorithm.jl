#!/usr/bin/env julia
using Random
using Printf

# Main idea : make loop visiting minority spin-majority spin-minority spin-..., when tail find head flip all the spin.


# ----------------------------
# Index helper (1-based)
# ----------------------------
@inline function idx_site(Lx::Int, Ly::Int, x::Int, y::Int, s::Int)::Int
    return 1 + s + 3*(mod(x,Lx) + Lx*mod(y,Ly))
end

# ----------------------------
# Triangle definitions (your convention)
#   Up(x,y)   = (x,y,0),(x,y,1),(x,y,2)
#   Down(x,y) = (x,y,2),(x,y+1,0),(x-1,y+1,1)
# ----------------------------
@inline function up_triangle(Lx::Int, Ly::Int, x::Int, y::Int)
    return (idx_site(Lx,Ly,x,y,0),
            idx_site(Lx,Ly,x,y,1),
            idx_site(Lx,Ly,x,y,2))
end

@inline function down_triangle(Lx::Int, Ly::Int, x::Int, y::Int)
    return (idx_site(Lx,Ly,x,   y,   2),
            idx_site(Lx,Ly,x,   y+1, 0),
            idx_site(Lx,Ly,x-1, y+1, 1))
end

@inline function tri_sites(kind::Int, Lx::Int, Ly::Int, x::Int, y::Int)
    return kind == 0 ? up_triangle(Lx,Ly,x,y) : down_triangle(Lx,Ly,x,y)
end

# ----------------------------
# Save spins_txt: "# i s_i" + one line per site
# ----------------------------
function save_spins_txt(path::String, spins::Vector{Int8})
    open(path, "w") do io
        println(io, "# i   s_i")
        for (i, s) in enumerate(spins)
            @printf(io, "%d  %d\n", i, Int(s))
        end
    end
    println("Saved spins -> $path")
end

# ----------------------------
# Map each site -> its unique UP (x,y) and DOWN (x,y)
# ----------------------------
function build_site_to_triangle_maps(Lx::Int, Ly::Int)
    N = 3*Lx*Ly
    up_x   = fill(-1, N); up_y   = fill(-1, N)
    down_x = fill(-1, N); down_y = fill(-1, N)

    @inbounds for x in 0:Lx-1, y in 0:Ly-1
        (a,b,c) = up_triangle(Lx,Ly,x,y)
        up_x[a]=x; up_y[a]=y
        up_x[b]=x; up_y[b]=y
        up_x[c]=x; up_y[c]=y

        (d,e,f) = down_triangle(Lx,Ly,x,y)
        down_x[d]=x; down_y[d]=y
        down_x[e]=x; down_y[e]=y
        down_x[f]=x; down_y[f]=y
    end

    @inbounds for i in 1:N
        (up_x[i]   >= 0 && up_y[i]   >= 0) || error("site $i missing UP membership")
        (down_x[i] >= 0 && down_y[i] >= 0) || error("site $i missing DOWN membership")
    end
    return up_x, up_y, down_x, down_y
end

# ----------------------------
# Minority site in a triangle (requires ice-rule abs(sum)=1)
# ----------------------------
function minority_site_in_triangle(spins::Vector{Int8}, tri::NTuple{3,Int})::Int
    (a,b,c) = tri
    sa, sb, sc = spins[a], spins[b], spins[c]
    nplus = (sa==1) + (sb==1) + (sc==1)
    nminus = 3 - nplus
    (nplus==2 && nminus==1) || (nplus==1 && nminus==2) ||
        error("Triangle violates ice-rule: spins=($sa,$sb,$sc)")

    minority_sign = (nplus==1) ? Int8(1) : Int8(-1)
    if sa == minority_sign
        return a
    elseif sb == minority_sign
        return b
    else
        return c
    end
end


# ----------------------------
# One closed ice-loop and flip
# ----------------------------
# ----------------------------
# Choose exit site in same triangle with opposite spin to entry
#   - avoid revisiting sites already in path
#   - allow only the start_site for closure
# ----------------------------
function choose_exit_site_norevisit(rng::AbstractRNG,
                                   spins::Vector{Int8},
                                   tri::NTuple{3,Int},
                                   entry_site::Int,
                                   visited_site::AbstractVector{Bool},
                                   start_site::Int)::Int
    (a,b,c) = tri
    entry_spin  = spins[entry_site]
    target_spin = Int8(-entry_spin)

    cand = Int[]
    @inbounds begin
        if a != entry_site && spins[a] == target_spin && (!visited_site[a] || a == start_site); push!(cand, a); end
        if b != entry_site && spins[b] == target_spin && (!visited_site[b] || b == start_site); push!(cand, b); end
        if c != entry_site && spins[c] == target_spin && (!visited_site[c] || c == start_site); push!(cand, c); end
    end

    isempty(cand) && return 0
    return cand[rand(rng, 1:length(cand))]
end




# ----------------------------
# One closed ice-loop and flip (no revisits inside loop)
# ----------------------------
function build_and_flip_ice_loop_retry!(rng::AbstractRNG,
                                       spins::Vector{Int8},
                                       up_x, up_y, down_x, down_y;
                                       Lx::Int, Ly::Int,
                                       max_steps::Int=100000,
                                       max_tries::Int=200,
                                       verbose::Bool=true)

    N = length(spins)

    for attempt in 1:max_tries
        # pick a fresh random starting triangle each attempt
        start_kind = rand(rng, 0:1)
        start_x    = rand(rng, 0:Lx-1)
        start_y    = rand(rng, 0:Ly-1)

        start_tri  = tri_sites(start_kind, Lx, Ly, start_x, start_y)
        start_site = minority_site_in_triangle(spins, start_tri)

        kind = start_kind
        x    = start_x
        y    = start_y
        entry_site = start_site

        path = Int[start_site]
        visited_site = falses(N)
        visited_site[start_site] = true

        closed = false

        for _ in 1:max_steps
            tri = tri_sites(kind, Lx, Ly, x, y)

            exit_site = choose_exit_site_norevisit(rng, spins, tri, entry_site, visited_site, start_site)
            if exit_site == 0
                # dead end -> abandon this attempt
                break
            end

            push!(path, exit_site)

            # hop to adjacent triangle sharing exit_site
            if kind == 0
                x = down_x[exit_site]; y = down_y[exit_site]; kind = 1
            else
                x = up_x[exit_site];   y = up_y[exit_site];   kind = 0
            end

            entry_site = exit_site

            # closed when we return to the exact starting state
            if kind == start_kind && x == start_x && y == start_y && entry_site == start_site
                closed = true
                break
            end

            visited_site[entry_site] = true
        end

        if closed
            loop_set = unique(path)
            @inbounds for i in loop_set
                spins[i] = -spins[i]
            end
            verbose && println("Loop closed after attempt $attempt (unique sites = $(length(loop_set)), steps = $(length(path))).")
            return loop_set, path, attempt
        end
    end

    error("No non-revisiting loop closed after max_tries=$max_tries. Try increasing max_tries or relaxing the no-revisit constraint.")
end



function save_spin_plot_with_loop(path_png::String,
                                  spins::Vector{Int8},
                                  Lx::Int, Ly::Int,
                                  loop_path::Vector{Int};
                                  title_str::String="")

    xs, ys = kagome_positions(Lx, Ly)

    up_idx = findall(==(Int8(1)), spins)
    dn_idx = findall(==(Int8(-1)), spins)

    # Base spin scatter
    plt = scatter(xs[dn_idx], ys[dn_idx];
        color=:blue, markersize=6, markerstrokewidth=0,
        legend=false, aspect_ratio=:equal, grid=false,
        xlabel="", ylabel="", title=title_str)

    scatter!(plt, xs[up_idx], ys[up_idx];
        color=:red, markersize=6, markerstrokewidth=0)

    # ---- Overlay the loop polyline (ordered) ----
    # Ensure the polyline is explicitly closed for visualization
    lp = loop_path
    if !isempty(lp) && lp[end] != lp[1]
        lp = vcat(lp, lp[1])
    end

    xlp = xs[lp]
    ylp = ys[lp]

    plot!(plt, xlp, ylp;
        color=:black, linewidth=3, alpha=0.9)

    # Highlight loop vertices
    scatter!(plt, xlp, ylp;
        color=:black, markersize=4, markerstrokewidth=0)

    # (Optional) annotate with step order numbers (can get busy)
    # for (k, site) in enumerate(lp[1:end-1])
    #     annotate!(plt, xs[site], ys[site], text(string(k), :black, 8))
    # end

    savefig(plt, path_png)
    println("Saved plot -> $path_png")
    return nothing
end

# ----------------------------
# Sanity check: all triangles satisfy ice-rule
# ----------------------------
function check_all_triangles(spins::Vector{Int8}, Lx::Int, Ly::Int)
    @inbounds for x in 0:Lx-1, y in 0:Ly-1
        (a,b,c) = up_triangle(Lx,Ly,x,y)
        s = Int(spins[a] + spins[b] + spins[c])
        abs(s) == 1 || error("UP ($x,$y) violates ice-rule, sum=$s")

        (d,e,f) = down_triangle(Lx,Ly,x,y)
        s2 = Int(spins[d] + spins[e] + spins[f])
        abs(s2) == 1 || error("DOWN ($x,$y) violates ice-rule, sum=$s2")
    end
    return true
end

using Plots   # add this

# ----------------------------
# Kagome positions (same embedding as before)
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
# Save a PNG plot of spins
# ----------------------------
function save_spin_plot_with_loop_labels(path_png::String,
                                        spins::Vector{Int8},
                                        Lx::Int, Ly::Int,
                                        loop_path::Vector{Int};
                                        title_str::String="",
                                        fontsize::Int=10)

    xs, ys = kagome_positions(Lx, Ly)

    up_idx = findall(==(Int8(1)), spins)
    dn_idx = findall(==(Int8(-1)), spins)

    # Base spin scatter
    plt = scatter(xs[dn_idx], ys[dn_idx];
        color=:blue, markersize=6, markerstrokewidth=0,
        legend=false, aspect_ratio=:equal, grid=false,
        xlabel="", ylabel="", title=title_str)

    scatter!(plt, xs[up_idx], ys[up_idx];
        color=:red, markersize=6, markerstrokewidth=0)

    # --- Build "first-visit order" list ---
    visited = falses(length(spins))
    ordered_sites = Int[]

    for site in loop_path
        if !visited[site]
            visited[site] = true
            push!(ordered_sites, site)
        end
    end

    # If loop_path ends by returning to the start, avoid double-labeling start
    if !isempty(loop_path) && loop_path[end] == loop_path[1]
        # already handled by first-visit logic, so nothing to do
    end

    # --- Annotate loop order: 1..M ---
    for (k, site) in enumerate(ordered_sites)
        annotate!(plt, xs[site], ys[site], text(string(k), fontsize, :black))
    end

    savefig(plt, path_png)
    println("Saved labeled plot -> $path_png")
    return nothing
end



# ----------------------------
# MAIN (no globals)
# ----------------------------
function main(Lx::Int, Ly::Int; nloops::Int=50, seed::Int=1)
    rng = MersenneTwister(seed)

    spins = load_spins_txt("spins_tiled_24x24.txt")
    up_x, up_y, down_x, down_y = build_site_to_triangle_maps(Lx, Ly)

    check_all_triangles(spins, Lx, Ly)
    save_spins_txt("spins_uniform_init.txt", spins)

    println("Running $nloops loop updates...")
    for t in 1:nloops
        loop_sites, loop_path, attempts = build_and_flip_ice_loop_retry!(
            rng, spins, up_x, up_y, down_x, down_y; Lx=Lx, Ly=Ly, max_tries=500, verbose=true
        )


        println("Loop $t indices (unique sites):")
        println(loop_sites)
        println("Loop length (unique): ", length(loop_sites))
        println("Path length (steps): ", length(loop_path))

        # Plot after this loop update (or before flipping if you want “pre-flip” picture)
        save_spin_plot_with_loop_labels("loop_$t.png", spins, Lx, Ly, loop_path; title_str="loop $t")


        println("Loop $t indices:")
        println(loop_sites)
        println(length(loop_sites))
        check_all_triangles(spins, Lx, Ly)
    end

    save_spins_txt("spins_after_loops.txt", spins)
    save_spin_plot("spins_after_loops.png", spins, Lx, Ly; title_str="after loops")
    println("Done.")
end

# ---- run ----
main(8, 8; nloops=50, seed=1)
