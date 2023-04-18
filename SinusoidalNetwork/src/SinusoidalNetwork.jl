module SinusoidalNetwork

using Random, DataFrames, JLD2, Serialization, ComponentArrays

export create_data
include("create_sine_data.jl")

export create_training_model, create_testing_model
include("sinusoid_network.jl")

export train
include("training_funcs.jl")

export main
function main(seed::Int64, epochs::Int64, lr::Float32, neurons::Int64, recur_gain::Float32, out_gain::Float32, activation::String, train_freq::Vector{Float32}, train_t_stop::Float32, test_freq::Vector{Float32}, test_t_stop::Float32, t_start::Float32, dt::Float32)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    train_input, train_output = create_data(train_freq, t_start, train_t_stop, dt)
    test_input, test_output = create_data(test_freq, t_start, test_t_stop, dt)
    train_data = (train_input, train_output, )
    test_data = (test_input, test_output, )

    model_train, ps, st = create_training_model(rng, recur_gain, out_gain, neurons, activation, )
    model_test = create_testing_model(rng, neurons, activation, )

    ps_out, accuracies = train(rng, epochs, lr, model_train, model_test, ps, st, train_data, test_data, )

    return ps_out, accuracies
end

export schedule
function schedule(location::String, output::String, row_num::Int64)
    df_loaded = load(location, "df")
    row = eachrow(df_loaded)[row_num]

    ps_out, accuracies = main(row...)

    output_file = output * "model_$(row_num).jls"
    open(output_file, "w") do f
        serialize(f, ps_out)
        serialize(f, accuracies)
    end
end

end # module SinusoidalNetwork
