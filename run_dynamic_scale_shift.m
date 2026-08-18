clear; clc; close all;

load("point_mass_dataset.mat");

% ODE coefficient ablation
lambdaList = [0, 0.01, 0.1, 1, 10];

% Training seeds
seedList = [42, 123, 456];

% Dynamics scale shift for testing
alphaList = [0.5, 0.8, 1.0, 1.2];

% Training settings
numEpochs = 300;
miniBatchSize = 256;
learnRate = 1e-3;

% Test noise setting
% Use sigma to test shifted stochastic dynamics.
% Use 0 to isolate deterministic dynamics-scale shift.
testSigma = sigma;

numLambdas = length(lambdaList);
numSeeds = length(seedList);
numAlphas = length(alphaList);

% Store results: lambda x alpha x seed
nextStateMSE_all = zeros(numLambdas, numAlphas, numSeeds);
positionMSE_all = zeros(numLambdas, numAlphas, numSeeds);
velocityMSE_all = zeros(numLambdas, numAlphas, numSeeds);
overallRMSE_all = zeros(numLambdas, numAlphas, numSeeds);
odeResidualMSE_all = zeros(numLambdas, numAlphas, numSeeds);
trainingTime_all = zeros(numLambdas, numSeeds);

% Training data from original dynamics
X = [S, A];      % [p_t, v_t, a_t]
Y = Snext;      % original next state from alpha = 1 dynamics + noise

N = size(X, 1);
Ntrain = round(0.8 * N);

for l = 1:numLambdas

    lambdaODE = lambdaList(l);

    for s = 1:numSeeds

        seed = seedList(s);
        rng(seed);

        fprintf("\nTraining lambdaODE = %.2f, seed = %d\n", lambdaODE, seed);

        % Train/test split controlled by seed
        idx = randperm(N);

        trainIdx = idx(1:Ntrain);
        testIdx = idx(Ntrain+1:end);

        Xtrain = X(trainIdx, :);
        Ytrain = Y(trainIdx, :);

        Xtest = X(testIdx, :);

        % Neural network architecture
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
        trainingTime_all(l, s) = trainingTime;

        % Predict once on Xtest
        XtestDL = dlarray(Xtest', "CB");
        YpredDL = predict(net, XtestDL);
        Ypred = extractdata(YpredDL)';

        % Test under shifted dynamics alpha
        for aIdx = 1:numAlphas

            alpha = alphaList(aIdx);

            % Current test state and action
            p_t = Xtest(:,1);
            v_t = Xtest(:,2);
            a_t = Xtest(:,3);

            % Shifted true dynamics:
            % p_{t+1} = p_t + dt*v_t
            % v_{t+1} = v_t + dt*alpha*a_t
            p_true_shift = p_t + dt .* v_t;
            v_true_shift = v_t + dt .* alpha .* a_t;

            % Add Gaussian noise to shifted test condition
            rng(seed + 1000*aIdx);
            testNoise = testSigma * randn(size(Xtest,1), 2);

            YtestShift = [p_true_shift, v_true_shift] + testNoise;

            % Evaluation metrics under shifted dynamics
            nextStateMSE = mean(sum((Ypred - YtestShift).^2, 2));

            positionMSE = mean((Ypred(:,1) - YtestShift(:,1)).^2);
            velocityMSE = mean((Ypred(:,2) - YtestShift(:,2)).^2);

            overallRMSE = sqrt(mean((Ypred - YtestShift).^2, "all"));

            % ODE residual MSE under shifted dynamics.
            % This checks consistency with the shifted true dynamics:
            % p_dot = v, v_dot = alpha*a_t
            p_pred = Ypred(:,1);
            v_pred = Ypred(:,2);

            r_p = (p_pred - p_t) ./ dt - v_t;
            r_v = (v_pred - v_t) ./ dt - alpha .* a_t;

            odeResidualMSE = mean(r_p.^2 + r_v.^2);

            % Store results
            nextStateMSE_all(l, aIdx, s) = nextStateMSE;
            positionMSE_all(l, aIdx, s) = positionMSE;
            velocityMSE_all(l, aIdx, s) = velocityMSE;
            overallRMSE_all(l, aIdx, s) = overallRMSE;
            odeResidualMSE_all(l, aIdx, s) = odeResidualMSE;

            fprintf("Test alpha = %.1f | Next-state MSE: %.6f | ODE residual MSE: %.6f\n", ...
                alpha, nextStateMSE, odeResidualMSE);
        end

        fprintf("Training time: %.2f seconds\n", trainingTime);
    end
end

% Collapse across seeds: lambda x alpha
meanNextStateMSE = mean(nextStateMSE_all, 3);
stdNextStateMSE = std(nextStateMSE_all, 0, 3);

meanPositionMSE = mean(positionMSE_all, 3);
stdPositionMSE = std(positionMSE_all, 0, 3);

meanVelocityMSE = mean(velocityMSE_all, 3);
stdVelocityMSE = std(velocityMSE_all, 0, 3);

meanOverallRMSE = mean(overallRMSE_all, 3);
stdOverallRMSE = std(overallRMSE_all, 0, 3);

meanODEResidualMSE = mean(odeResidualMSE_all, 3);
stdODEResidualMSE = std(odeResidualMSE_all, 0, 3);

% Build long results table
rows = [];

for l = 1:numLambdas
    for aIdx = 1:numAlphas

        rows = [rows; table( ...
            lambdaList(l), ...
            alphaList(aIdx), ...
            meanNextStateMSE(l,aIdx), stdNextStateMSE(l,aIdx), ...
            meanPositionMSE(l,aIdx), stdPositionMSE(l,aIdx), ...
            meanVelocityMSE(l,aIdx), stdVelocityMSE(l,aIdx), ...
            meanOverallRMSE(l,aIdx), stdOverallRMSE(l,aIdx), ...
            meanODEResidualMSE(l,aIdx), stdODEResidualMSE(l,aIdx), ...
            'VariableNames', { ...
            'lambdaODE', 'alpha', ...
            'NextStateMSE_mean', 'NextStateMSE_std', ...
            'PositionMSE_mean', 'PositionMSE_std', ...
            'VelocityMSE_mean', 'VelocityMSE_std', ...
            'OverallRMSE_mean', 'OverallRMSE_std', ...
            'ODEResidualMSE_mean', 'ODEResidualMSE_std' ...
            })];

    end
end

resultsTable = rows;

disp(resultsTable);

save("dynamic_scale_shift_results.mat", ...
    "lambdaList", "seedList", "alphaList", ...
    "nextStateMSE_all", "positionMSE_all", "velocityMSE_all", ...
    "overallRMSE_all", "odeResidualMSE_all", ...
    "meanNextStateMSE", "stdNextStateMSE", ...
    "meanODEResidualMSE", "stdODEResidualMSE", ...
    "resultsTable");

writetable(resultsTable, "dynamic_scale_shift_results.csv");

% Plot next-state MSE for each alpha
figure;
hold on;
for aIdx = 1:numAlphas
    errorbar(lambdaList, meanNextStateMSE(:,aIdx), stdNextStateMSE(:,aIdx), ...
        "-o", "LineWidth", 1.5);
end
xlabel("\lambda_{ODE}");
ylabel("Next-state MSE");
title("Dynamic scale shift: Next-state MSE");
legend("alpha=0.8", "alpha=1.0", "alpha=1.2", "Location", "best");
grid on;
hold off;

% Plot ODE residual MSE for each alpha
figure;
hold on;
for aIdx = 1:numAlphas
    errorbar(lambdaList, meanODEResidualMSE(:,aIdx), stdODEResidualMSE(:,aIdx), ...
        "-o", "LineWidth", 1.5);
end
xlabel("\lambda_{ODE}");
ylabel("ODE residual MSE");
title("Dynamic scale shift: ODE residual MSE");
legend("alpha=0.8", "alpha=1.0", "alpha=1.2", "Location", "best");
grid on;
hold off;