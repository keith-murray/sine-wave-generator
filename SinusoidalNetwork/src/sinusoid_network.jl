using Lux, Optimisers, LinearAlgebra, Random, Statistics, ComponentArrays, NNlib

function why_god(arr::Vector{Matrix{Float32}})
	return permutedims(cat(
			arr[1],
			arr[2],
			arr[3],
			arr[4],
			arr[5],
			arr[6],
			arr[7],
			arr[8],
			arr[9],
			arr[10],
			arr[11],
			arr[12],
			arr[13],
			arr[14],
			arr[15],
			arr[16],
			arr[17],
			arr[18],
			arr[19],
			arr[20],
			arr[21],
			arr[22],
			arr[23],
			arr[24],
			arr[25],
			arr[26],
			arr[27],
			arr[28],
			arr[29],
			arr[30],
			arr[31],
			arr[32],
			arr[33],
			arr[34],
			arr[35],
			arr[36],
			arr[37],
			arr[38],
			arr[39],
			arr[40],
			arr[41],
			arr[42],
			arr[43],
			arr[44],
			arr[45],
			arr[46],
			arr[47],
			arr[48],
			arr[49],
			arr[50],
			arr[51],
			arr[52],
			arr[53],
			arr[54],
			arr[55],
			arr[56],
			arr[57],
			arr[58],
			arr[59],
			arr[60],
			arr[61],
			arr[62],
			arr[63],
			arr[64],
			arr[65],
			arr[66],
			arr[67],
			arr[68],
			arr[69],
			arr[70],
			arr[71],
			arr[72],
			arr[73],
			arr[74],
			arr[75],
			arr[76],
			arr[77],
			arr[78],
			arr[79],
			arr[80],
			arr[81],
			arr[82],
			arr[83],
			arr[84],
			arr[85],
			arr[86],
			arr[87],
			arr[88],
			arr[89],
			arr[90],
			arr[91],
			arr[92],
			arr[93],
			arr[94],
			arr[95],
			arr[96],
			arr[97],
			arr[98],
			arr[99],
			arr[100],
			arr[101],
		dims=3), (1, 3, 2))
end

function create_training_model(
    rng::AbstractRNG, 
    recurrent_gain::Float32, 
    output_gain::Float32, 
    hidden_neurons::Int64,
    activation::String
    )
	recurrent_init(rng, dims...) = Lux.glorot_normal(rng, dims...; gain=recurrent_gain)
	output_init(rng, dims...) = Lux.glorot_normal(rng, dims...; gain=output_gain)
	
    if activation == "tanh"
    model = Chain(
		Recurrence(
			RNNCell(
				1 => hidden_neurons, tanh; 
				use_bias=false,
				train_state=true,
				init_weight=recurrent_init
			); 
			return_sequence = true
		),
		why_god,
		Dense(hidden_neurons, 1; init_weight=output_init, use_bias=false)
	)
    else
        model = Chain(
            Recurrence(
                RNNCell(
                    1 => hidden_neurons, identity; 
                    use_bias=false,
                    train_state=true,
                    init_weight=recurrent_init
                ); 
                return_sequence = true
            ),
            why_god,
            Dense(hidden_neurons, 1; init_weight=output_init, use_bias=false)
        )
    end
	
    ps, st = Lux.setup(rng, model)
	ps = ComponentArray(ps)

    return model, ps, st
end

function create_testing_model(
    rng::AbstractRNG, 
    hidden_neurons::Int64,
    activation::String
    )

    if activation == "tanh"
    model = Chain(
		Recurrence(
			RNNCell(
				1 => hidden_neurons, tanh; 
				use_bias=false,
				train_state=true,
			); 
			return_sequence = true
		),
		x -> permutedims(cat(x..., dims=3), (1, 3, 2)),
		Dense(hidden_neurons, 1; use_bias=false)
	)
    else
        model = Chain(
            Recurrence(
                RNNCell(
                    1 => hidden_neurons, identity; 
                    use_bias=false,
                    train_state=true,
                ); 
                return_sequence = true
            ),
            x -> permutedims(cat(x..., dims=3), (1, 3, 2)),
            Dense(hidden_neurons, 1; use_bias=false)
        )
    end

    return model
end