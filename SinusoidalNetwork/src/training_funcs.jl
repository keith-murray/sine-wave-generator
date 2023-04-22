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

function constructIterator(rng::AbstractRNG, batch::Int64, training_data)
    x_es = training_data[1]
    y_expecteds = training_data[2]
    rand_indexes = randperm(rng, size(y_expecteds)[3])
    training_iters = div(size(y_expecteds)[3], batch)

    iter_indexes = [rand_indexes[(i-1)*batch+1:i*batch] for i in 1:training_iters]
    iterator = [(x_es[:,:,iter_indexes[i]], y_expecteds[:,:,iter_indexes[i]]) for i in 1:training_iters]
    return iterator
end

function train(rng::AbstractRNG, epochs::Int64, batch::Int64, lr::Float32, model_train, model_test, ps, st, training_data, testing_data, )	
    opt = Optimisers.Adam(lr, )
    st_opt = Optimisers.setup(opt, ps)
    losses = Lux.zeros32(rng, 2, epochs+1)

    ### Warmup the Model
    stime = time()
    iterator = constructIterator(rng, batch, training_data)
    loss(iterator[1][1], iterator[1][2], model_train, ps, st)
    (l, _), back = pullback(p -> loss(iterator[1][1], iterator[1][2], model_train, p, st), ps)
    back((one(l), nothing))
    current_results = constructResults(model_test, ps, st, training_data, testing_data, )
    ttime = time() - stime
    losses[:, 1] = current_results
    printUpdates(0, epochs, ttime, current_results)

    ### Lets train the model
    for epoch in 1:epochs
        stime = time()
        iterator = constructIterator(rng, batch, training_data)

        for (x, y) in iterator
            (l, st), back = pullback(p -> loss(x, y, model_train, p, st), ps)
            gs = back((one(l), nothing))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end

        current_results = constructResults(model_test, ps, st, training_data, testing_data, )
        ttime = time() - stime
        losses[:, epoch+1] = current_results
        printUpdates(epoch, epochs, ttime, current_results)
    end
	return ps, losses
end