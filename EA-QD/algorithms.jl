#=
Created on 01/12/2022 09:42:23
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Didactic Julia implementation of diverse EA and 
QD algorithms
=#

using DataStructures, LinearAlgebra

function DataStructures.Deque(collection::AbstractVector{T}) where {T}
    deque = Deque{T}()
    for x in collection
        push!(deque, x)
    end
    return deque
end

# TYPES

abstract type Mutatator end

abstract type Crossover end

# recombinator combines mutation and crossover
struct Recombinator{M, C}
    MUT::M
    CO::C
end

abstract type Selector end

abstract type Descriptor end

# ALGORITHMS

function random_search(f, s, M::Mutatator; n_gen)
    p = f(s)
    for _ in 1:n_gen
        s′ = mutate(M, s)
        p′ = f(s′)
        p, s = min((p′, s′), (p, s))
    end
    return s, p
end

function simulated_annealing(f, s, M::Mutatator;
        kT, Tmax, Tmin, r)
    @assert Tmin < Tmax && 0 ≤ r < 1
    T = Tmax
    p = f(s)
    while T > Tmin
        for _ in 1:kT
            s′ = mutate(M, s)
            p′ = f(s′)
            if rand() ≤ exp(-(p′ - p) / T)
                p, s = p′, s′
            end
        end
        T *= r  # cool
    end
    return s, p
end

function genetic_algorithm(f, population, S::Selector,
                    R::Recombinator; n_gen)
    for _ in 1:n_gen
        parents = select(S, f, population)
        population = recombinate(R, parents)
    end
    return population
end

function get_novelty(B::Descriptor, s, references, k)
    dists = Float64[]
    b = get_descriptor(B, s)
    for s′ in references
        b′ = get_descriptor(B, s′)
        heappush!(dists, norm(b .- b′))
    end
    ρ = 0.0  # novelty is sum of k smallest distances
    for _ in 1:k
        ρ += heappop!(dists)
    end
    return ρ
end


function novelty_search(population, R::Recombinator, B::Descriptor;
                    n_gen, k, archive_size,
                    n_add_arch=length(population)÷4)
    # make archive of descriptors
    archive = Deque(population)
    for _ in 1:n_gen
        children = recombinate(R, population)
        references = Iterators.flatten((population, children, archive))
        novelty_pop = [(get_novelty(B, s, references, k+1), s) 
                                for s in Iterators.flatten((population, children))]
        # take most novel solutions as new population
        sort!(novelty_pop, rev=true)
        for (i, (ρ, s)) in enumerate(novelty_pop)
            # update population
            i ≤ length(population) && (population[i] = s)
            i ≤ n_add_arch && push!(archive, s)
            length(archive) > archive_size && popfirst!(archive)
        end
    end
    return population
end




function minimum_criteria_novelty_search(population, criterion,
                R::Recombinator, B::Descriptor;
                n_gen, k, archive_size)
    archive = Deque(population)
	n = length(population)
    for _ in 1:n_gen
        children = recombinate(R, population)
		# remove non-viable children
		filter!(criterion, children)
		append!(population, children)
		# take most novel solutions as new population
		if length(population) > n
	        reference_set = Iterators.flatten((population, children, archive))
	        novelty_pop = [(get_novelty(B, s, reference_set, k), s)
	                                for s in population]
		end
        sort!(novelty_pop, rev=true)
        # take most novel solutions as new population
        sort!(novelty_pop, rev=true)
        for (i, (ρ, s)) in enumerate(novelty_pop)
            # update population
            i ≤ n && (population[i] = s)
            i ≤ n_add_arch && push!(archive, s)
            length(archive) > archive_size && popfirst!(archive)
        end
    end
    return population
end

function map_elites(f, population, R::Recombinator,
                B::Descriptor; n_gen)
    # create archive linking descriptors with solution
    archive = Dict(get_descriptors(B, s) => (f(s), s)
                    for s in population)
    for _ in 1:n_gen
        # recombinate the population
        population = recombinate(R, population)
        # store new solutions in archive
        for s in population
            p, b = f(s), get_descriptors(B, s)
            if !haskey(archive, b) || p ≤ archive[b][1]
                archive[b] = (p, s)
            end
        end     
    end
    return archive
end

function minimum_criteria_coevolution(population, environments,
                criterion, RP::Recombinator, RE::Recombinator; n_gen)
	population = Deque(population)
    environments = Deque(environments)
    n, m = length(population), length(environments)
	for _ in 1:n_gen
        # recombinate
		population_offspring = recombinate(RP, population)
        environments_offspring = recombinate(RE, environments)
        # only keep only populations that survive in at least
        # one environment
		filter!(s->any(e->criterion(s, e), environments_offspring), population_offspring)
        # ditto for the environments
        filter!(e->any(s->criterion(s, e), population_offspring), environments_offspring)
        # add to populations, keep the youngest
        for s in population_offspring
            push!(population, s)
            length(population) > n && popfirst!(population)
        end
        for e in environments_offspring
            push!(environments, e)
            length(environments) > m && popfirst!(environments)
        end
    end
    return population, environments
end