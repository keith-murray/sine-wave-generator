using Statistics, Lux, Optimisers, Zygote

meansquarederror(y_pred, y) =  mean(abs2, y_pred .- y)

function loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    l = meansquarederror(y_pred, y)
    return l, st
end

function test_loss(x, y, model, ps, st)
    st = Lux.testmode(st)
    y_pred, st = model(x, ps, st)
    return meansquarederror(y_pred, y)
end

function constructResults(model, ps, st, training_data, testing_data, )
    loss_training = test_loss(training_data[1], training_data[2], model, ps, st)
    loss_testing = test_loss(testing_data[1], testing_data[2], model, ps, st)
    return [loss_training, loss_testing, ]
end

function printUpdates(epoch, epochs, ttime, current_results)
    println("[$epoch/$epochs] \t Time $(round(ttime; digits=2))s \t Train Loss: " * "$(round(current_results[1]; digits=4)) \t " * "Test Loss: $(round(current_results[2]; digits=4))")
end

function train(rng::AbstractRNG, epochs::Int64, lr::Float32, model, ps, st, training_data, testing_data, )	
    opt = Optimisers.Adam(lr, )
    st_opt = Optimisers.setup(opt, ps)
    losses = Lux.zeros32(rng, 2, epochs+1)

    ### Warmup the Model
    stime = time()
    loss(training_data[1], training_data[2], model, ps, st)
    (l, _), back = pullback(p -> loss(training_data[1], training_data[2], model, p, st), ps)
    back((one(l), nothing))
    current_results = constructResults(model, ps, st, training_data, testing_data, )
    ttime = time() - stime
    losses[:, 1] = current_results
    printUpdates(0, epochs, ttime, current_results)

    ### Lets train the model
    for epoch in 1:epochs
        stime = time()

        (l, st), back = pullback(p -> loss(training_data[1], training_data[2], model, p, st), ps)
		gs = back((one(l), nothing))[1]
		st_opt, ps = Optimisers.update(st_opt, ps, gs)
        current_results = constructResults(model, ps, st, training_data, testing_data, )

        ttime = time() - stime
        losses[:, epoch+1] = current_results
        printUpdates(epoch, epochs, ttime, current_results)
    end
	return ps, losses
end