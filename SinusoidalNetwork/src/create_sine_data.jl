using Random, Statistics, LinearAlgebra

function generate_sinusoid(freq::Float32, t_start::Float32, t_stop::Float32, dt::Float32)
    t = t_start:dt:t_stop
    y = sin.(2π*freq*t)
    return Float32.(y)
end

function create_data(
    freqs::Vector{Float32},
    t_start::Float32, t_stop::Float32, dt::Float32,
    )
    t = collect(t_start:dt:t_stop)
    inputs = zeros(Float32, 1, length(t), length(freqs))
    outputs = zeros(Float32, 1, length(t), length(freqs))

    for i in 1:length(freqs)
        freq = freqs[i]
        inputs[1,:,i] = fill(freq, length(t))
        outputs[1,:,i] = sin.(Float32(2π)*freq*t)
    end
    return inputs, outputs
end