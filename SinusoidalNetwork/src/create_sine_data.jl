using Random, Statistics, LinearAlgebra

function generate_sinusoid(freq::Float32, t_start::Float32, t_stop::Float32, dt::Float32)
    t = t_start:dt:t_stop
    y = sin.(2π*freq*t)
    return Float32.(y)
end

function create_data(
    rng::AbstractRNG, 
    data_size::Int64,
    freqs::Vector{Float32},
    t_locs::Vector{Float32}, 
    dt::Float32,
    )
    t = collect(t_locs[1]:dt:t_locs[2])
    inputs = zeros(Float32, 1, length(t), data_size)
    outputs = zeros(Float32, 1, length(t), data_size)
    freqs = (freqs[2] - freqs[1]) .* rand(rng, Float32, data_size) .+ freqs[1]

    for i in 1:data_size
        freq_inputs = (freqs[i] - 0.10f0) / 0.50f0 + 0.25f0
        inputs[1,:,i] = fill(freq_inputs, length(t))
        outputs[1,:,i] = sin.(freqs[i]*t)
    end
    return inputs, outputs
end

function loadModel(output_file::String)
	open(output_file, "r") do f
		ps = deserialize(f)
		accuracies = deserialize(f)
		return ps, accuracies
	end
end