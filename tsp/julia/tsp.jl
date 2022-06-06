using LinearAlgebra



# open("../data/$(ARGS[1])") do f
open(ARGS[1]) do f
 
    # line_number
    line = 0  
    s = readline(f)         
    line += 1
    global N = parse(Int, s)
    # read till end of file
    global points = Vector{Float64}[]
    while ! eof(f)         

        s = readline(f)      
        push!(points, map(x -> parse(Float64, x), split(s)))
    end
   
end
adjacency_matrix = Vector{Float64}[]
for point in points
    push!(adjacency_matrix, map(x -> norm(point-x), points))
end
adjacency_matrix = mapreduce(permutedims, vcat, adjacency_matrix)


struct Point
    id::Int64 # probability
    in::Point   # duration of visit
    out::Int64 # profit of visit
end


function two_opt(path, weight, augmented_weight, points_for_swap, penalties, adjacency_matrix)
    u1, v1, u2, v2 = points_for_swap
    u1 != v1 ? nothing :  return path, weight, augmented_weight #throw(AssertionError("u1 == v1"))
    u1 != u2 ? nothing :  return path, weight, augmented_weight #throw(AssertionError("u1 == u2"))
    v2 != u1 ? nothing :  return path, weight, augmented_weight #throw(AssertionError("u1 == v2"))
    u2 != v1 ? nothing :  return path, weight, augmented_weight #throw(AssertionError("u2 == v1"))
    v2 != u2 ? nothing :  return path, weight, augmented_weight #throw(AssertionError("v2 == u2"))
    v2 != v1 ? nothing :  return path, weight, augmented_weight #throw(AssertionError("v1 == v2"))

    u1_index = 0
    u2_index = 0
    v1_index = 0
    v2_index = 0
    for (idx, point) in enumerate(path)
        u1 == point && (u1_index = idx)
        u2 == point && (u2_index = idx)
        v1 == point && (v1_index = idx)
        v2 == point && (v2_index = idx)
    end

    old_edges_weight = adjacency_matrix[u1, v1] + adjacency_matrix[u2, v2] 
    old_penalty = lambda * (penalties[u1, v1] + penalties[u2, v2])



    if (u1_index > v1_index) && !(u1_index == length(path) && v1_index == 1)
        u1_index, v1_index = v1_index, u1_index 
        u1, v1 = v1, u1
    end

    if (u2_index > v2_index) && !(u2_index == length(path) && v2_index == 1) 
        u2_index, v2_index = v2_index, u2_index
        u2, v2 = v2, u2
    end 

    if u1_index > u2_index
        u1, v1, u2, v2= u2, v2, u1, v1
        u1_index, v1_index, u2_index, v2_index= u2_index, v2_index, u1_index, v1_index

    end
    new_edges_weight = adjacency_matrix[u1, u2] + adjacency_matrix[v1, v2] 
    new_penalty = lambda * (penalties[u1, u2] + penalties[v1, v2])
    new_path = reverse(path, min(v1_index, u2_index), max(v1_index, u2_index))

    new_weight = weight - old_edges_weight + new_edges_weight
    new_augmented_weight = augmented_weight - old_edges_weight - old_penalty + new_edges_weight + new_penalty
    return new_path, new_weight, new_augmented_weight 
end

function FastLocalSearch(path, weight, augmented_weight, activation_bits, penalties, adjacency_matrix, best_weight, best_path)
    initial_path = path
    initial_weight = weight
    initial_augmented_weight = weight
    while findfirst(x -> x==1, activation_bits) != nothing
        for (point_generator, bit) in enumerate(activation_bits)
            bit == 0 && continue



            was_successfull = false
            idx = 1
            while idx <=length(path)
                u = path[idx]
                # u == point_generator && continue
                v = idx < length(path) ? path[idx+1] : path[1]


                
                point_generator_neighbor_left_index = findfirst(x -> x == point_generator, path) - 1
                point_generator_neighbor_left_index < 1 && (point_generator_neighbor_left_index = 1)
                point_generator_neighbor_left = path[point_generator_neighbor_left_index]

                points_for_swap_left = [point_generator_neighbor_left, point_generator, u, v]

                new_path, new_weight, new_augmented_weight = two_opt(path, weight, augmented_weight, points_for_swap_left, penalties, adjacency_matrix)
                # round(new_weight; digits=4) == round(weight_path(new_path, adjacency_matrix); digits=4) ? nothing : throw(AssertionError(
                #     "weight computed incorrectly, value $new_weight, correct value $(weight_path(new_path, adjacency_matrix)) previous path $path new path $new_path considered_poitns $points_for_swap_left"
                #     ))
                # println("\nleft old weight $weight new weight $new_weight")
                if new_augmented_weight < augmented_weight
                    path = new_path
                    weight = new_weight
                    augmented_weight = new_augmented_weight
                    was_successfull = true
                    activation_bits[point_generator] = 1
                    activation_bits[point_generator_neighbor_left] = 1
                    activation_bits[u] = 1
                    activation_bits[v] = 1
                end

                if new_weight < best_weight
                    best_weight = new_weight
                    best_path = new_path
                end

                u = path[idx]
                # u == point_generator && continue
                v = idx < length(path) ? path[idx+1] : path[1]
                
                point_generator_neighbor_right_index = findfirst(x -> x == point_generator, path) + 1
                point_generator_neighbor_right_index > length(initial_path) && (point_generator_neighbor_right_index = 1) 
                point_generator_neighbor_right = path[point_generator_neighbor_right_index]
                points_for_swap_right = [point_generator, point_generator_neighbor_right, u, v]
                

                new_path, new_weight, new_augmented_weight = two_opt(path, weight, augmented_weight, points_for_swap_right, penalties, adjacency_matrix)
                # round(new_weight; digits=4) == round(weight_path(new_path, adjacency_matrix); digits=4) ? nothing : throw(AssertionError(
                #     "weight computed incorrectly, value $new_weight, correct value $(weight_path(new_path, adjacency_matrix)) previous path $path new path $new_path considered_poitns $points_for_swap_right"
                #     ))
                # print("\nright old weight $weight new weight $new_weight")
                if new_augmented_weight < augmented_weight
                    path = new_path
                    weight = new_weight
                    augmented_weight = new_augmented_weight
                    was_successfull = true
                    activation_bits[point_generator] = 1
                    activation_bits[point_generator_neighbor_right] = 1
                    activation_bits[u] = 1
                    activation_bits[v] = 1
                end

                if new_weight < best_weight
                    best_weight = new_weight
                    best_path = new_path
                end

                idx += 1

            end
            # println()
            # println("was successfull $was_successfull")
            # println("path is $path")
            # println("activation_bits $activation_bits")

            !was_successfull && (activation_bits[point_generator] = 0)

        end
    end
    return path, weight, augmented_weight, best_path, best_weight

end


function weight_path(path, adjacency_matrix)
    weight = 0
    for (idx, point) in enumerate(path)
        # println(idx)
        next = idx == length(path) ? path[1] : path[idx+1] 
        weight += adjacency_matrix[point, next]
    end
    return weight
end

function GuidedFastLocalSearch(points, adjacency_matrix)
    k = 0
    initial_path = points
    penalties = [[0 for point in initial_path] for point in initial_path]
    penalties = mapreduce(permutedims, vcat, penalties)
    activation_bits = [1 for point in initial_path]
    weight = weight_path(initial_path, adjacency_matrix)
    augmented_weight = weight
    path = initial_path
    best_path = path
    best_weight = weight
    for epoch in 1:100000
        epoch % 100 == 0 && println("$epoch $(round(best_weight;digits=0)) ") 
        path, weight, augmented_weight, best_path, best_weight = FastLocalSearch(path, weight, augmented_weight, activation_bits, penalties, adjacency_matrix, best_weight, best_path)
        max_util = 0
        for (idx, point) in enumerate(path)
            if idx < length(path)
                u, v = path[idx], path[idx+1]
            else 
                u, v = path[idx], path[1]
            end
            util = adjacency_matrix[u, v] / ( 1 + penalties[u, v])
            global penalized
            if util > max_util
                penalized = [[u, v]]
                max_util = util
            elseif util == max_util 
                push!(penalized, [u, v])
            end
        end
        for (u, v) in penalized
            penalties[u, v] += 1
            penalties[v, u] += 1
            activation_bits[u] = 1
            activation_bits[v] = 1
        end

    end
    return best_path, best_weight

end
global lambda = 20
# display(adjacency_matrix)

best_path, best_weight = GuidedFastLocalSearch(1:N, adjacency_matrix)
# path = 1:5
# weight = weight_path(path, adjacency_matrix)
# println()
# println("path $(collect(path)) weight $weight")
# points_for_swap = [1, 2, 3, 4]
# penalties = [[0 for point in path] for point in path]
# penalties = mapreduce(permutedims, vcat, penalties)
# new_path, new_weight, new_augmented_weight = two_opt(path, weight, weight, points_for_swap, penalties, adjacency_matrix)
# println("new path $new_path new weight $new_weight")
println("$best_weight 0")
global best_path
while best_path[1] !=1 global best_path = circshift(best_path, 1) end
println(join(map(x -> x-1, best_path), ' '))


