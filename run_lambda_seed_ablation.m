clear; clc; close all;

load("point_mass_dataset.mat");

% ODE coefficient ablation
lambdaList = [0, 0.01, 0.1, 1, 10];

% Fixed training seeds
seedList = [42, 123, 456];

% Training settings
numEpochs = 300;
miniBatchSize = 256;
learnRate = 1e-3;

numLambdas = length(lambdaList);
numSeeds = length(seedList);

% Store all results
nextStateMSE_all = zeros(numLambdas, numSeeds);
positionMSE_all = zeros(numLambdas, numSeeds);
velocityMSE_all = zeros(numLambdas, numSeeds);
overallRMSE_all = zeros(numLambdas, numSeeds);
odeResidualMSE_all = zeros(numLambdas, numSeeds);
trainingTime_all = zeros(numLambdas, numSeeds);

% Inputs and targets
X = [S, A];        % [p_t, v_t, a_t]
Y = Snext;        % [p_{t+1}, v_{t+1}]

N = size(X, 1);
Ntrain = round(0.8 * N);

for l = 1:numLambdas

    lambdaODE = lambdaList(l);

    for s = 1:numSeeds

        seed = seedList(s);
        rng(seed);

        fprintf("\nRunning lambdaODE = %.2f, seed = %d\n", lambdaODE, seed);

        % Train/test split controlled by training seed
        idx = randperm(N);

        trainIdx = idx(1:Ntrain);
        testIdx = idx(Ntrain+1:end);

        Xtrain = X(trainIdx, :);
        Ytrain = Y(trainIdx, :);

        Xtest = X(testIdx, :);
        Ytest = Y(testIdx, :);

        XtestDL = dlarray(Xtest', "CB");

        % Same architecture as existing ODE model
        layers = [
            featureInputLayer(3, "Normalization", "none")
            fullyConnectedLayer(64)
            tanhLayer
            fullyConnectedLayer(64)
            tanhLayer
            fullyConnectedLayer(2)
        ];

        net = dlnetwork(layers);

        trailingAvg = [];
        trailingAvgSq = [];

        numIterationsPerEpoch = floor(Ntrain / miniBatchSize);
        lossHistory = zeros(numEpochs, 1);

        tic;

        for epoch = 1:numEpochs

            shuffleIdx = randperm(Ntrain);
            epochLoss = 0;

            for i = 1:numIterationsPerEpoch

                batchIdx = shuffleIdx((i-1)*miniBatchSize + 1 : i*miniBatchSize);

                XBatch = dlarray(Xtrain(batchIdx, :)', "CB");
                YBatch = dlarray(Ytrain(batchIdx, :)', "CB");

                [loss, gradients] = dlfeval(@modelLoss, net, XBatch, YBatch, dt, lambdaODE);

                [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
                    trailingAvg, trailingAvgSq, ...
                    (epoch-1)*numIterationsPerEpoch + i, learnRate);

                epochLoss = epochLoss + double(gather(extractdata(loss)));
            end

            lossHistory(epoch) = epochLoss / numIterationsPerEpoch;

            if mod(epoch, 100) == 0
                fprintf("Epoch %d, Loss %.6f\n", epoch, lossHistory(epoch));
            end
        end

        trainingTime = toc;

        % Test prediction
        YpredDL = predict(net, XtestDL);
        Ypred = extractdata(YpredDL)';

        % Evaluation metrics
        nextStateMSE = mean(sum((Ypred - Ytest).^2, 2));

        positionMSE = mean((Ypred(:,1) - Ytest(:,1)).^2);
        velocityMSE = mean((Ypred(:,2) - Ytest(:,2)).^2);

        overallRMSE = sqrt(mean((Ypred - Ytest).^2, "all"));

        % ODE residual MSE on test set
        p_t = Xtest(:,1);
        v_t = Xtest(:,2);
        a_t = Xtest(:,3);

        p_pred = Ypred(:,1);
        v_pred = Ypred(:,2);

        r_p = (p_pred - p_t) ./ dt - v_t;
        r_v = (v_pred - v_t) ./ dt - a_t;

        odeResidualMSE = mean(r_p.^2 + r_v.^2);

        % Store result
        nextStateMSE_all(l, s) = nextStateMSE;
        positionMSE_all(l, s) = positionMSE;
        velocityMSE_all(l, s) = velocityMSE;
        overallRMSE_all(l, s) = overallRMSE;
        odeResidualMSE_all(l, s) = odeResidualMSE;
        trainingTime_all(l, s) = trainingTime;

        fprintf("Next-state MSE: %.6f\n", nextStateMSE);
        fprintf("Position MSE: %.6f\n", positionMSE);
        fprintf("Velocity MSE: %.6f\n", velocityMSE);
        fprintf("Overall RMSE: %.6f\n", overallRMSE);
        fprintf("ODE residual MSE: %.6f\n", odeResidualMSE);
        fprintf("Training time: %.2f seconds\n", trainingTime);
    end
end

% Mean and standard deviation across seeds
meanNextStateMSE = mean(nextStateMSE_all, 2);
stdNextStateMSE = std(nextStateMSE_all, 0, 2);

meanPositionMSE = mean(positionMSE_all, 2);
stdPositionMSE = std(positionMSE_all, 0, 2);

meanVelocityMSE = mean(velocityMSE_all, 2);
stdVelocityMSE = std(velocityMSE_all, 0, 2);

meanOverallRMSE = mean(overallRMSE_all, 2);
stdOverallRMSE = std(overallRMSE_all, 0, 2);

meanODEResidualMSE = mean(odeResidualMSE_all, 2);
stdODEResidualMSE = std(odeResidualMSE_all, 0, 2);

meanTrainingTime = mean(trainingTime_all, 2);
stdTrainingTime = std(trainingTime_all, 0, 2);

% Results table
resultsTable = table( ...
    lambdaList', ...
    meanNextStateMSE, stdNextStateMSE, ...
    meanPositionMSE, stdPositionMSE, ...
    meanVelocityMSE, stdVelocityMSE, ...
    meanOverallRMSE, stdOverallRMSE, ...
    meanODEResidualMSE, stdODEResidualMSE, ...
    meanTrainingTime, stdTrainingTime, ...
    'VariableNames', { ...
    'lambdaODE', ...
    'NextStateMSE_mean', 'NextStateMSE_std', ...
    'PositionMSE_mean', 'PositionMSE_std', ...
    'VelocityMSE_mean', 'VelocityMSE_std', ...
    'OverallRMSE_mean', 'OverallRMSE_std', ...
    'ODEResidualMSE_mean', 'ODEResidualMSE_std', ...
    'TrainingTime_mean', 'TrainingTime_std' ...
    });

disp(resultsTable);

% Save results
save("lambda_seed_ablation_results.mat", ...
    "lambdaList", "seedList", ...
    "nextStateMSE_all", "positionMSE_all", "velocityMSE_all", ...
    "overallRMSE_all", "odeResidualMSE_all", "trainingTime_all", ...
    "resultsTable");

writetable(resultsTable, "lambda_seed_ablation_results.csv");

% Plot next-state MSE
figure;
errorbar(lambdaList, meanNextStateMSE, stdNextStateMSE, "-o", "LineWidth", 1.5);
xlabel("\lambda_{ODE}");
ylabel("Next-state MSE");
title("Next-state MSE across ODE loss coefficients");
grid on;

% Plot ODE residual MSE
figure;
errorbar(lambdaList, meanODEResidualMSE, stdODEResidualMSE, "-o", "LineWidth", 1.5);
xlabel("\lambda_{ODE}");
ylabel("ODE residual MSE");
title("ODE residual MSE across ODE loss coefficients");
grid on;