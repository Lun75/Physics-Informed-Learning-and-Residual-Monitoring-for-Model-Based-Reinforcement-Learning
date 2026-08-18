clear; clc; close all;

load("point_mass_dataset.mat");

% Case 1: State-dependent dynamic mismatch
% Nominal training dynamics:
%   p_dot = v
%   v_dot = a_t
% Case 1 evaluation dynamics:
%   g(s_t) = exp(alpha * p_t)
%   p_{t+1} = p_t + dt*v_t
%   v_{t+1} = v_t + dt*exp(alpha*p_t)*a_t
% This tests whether the learned transition model remains robust
% when the action effect changes across the state space.

% ODE coefficient ablation
lambdaList = [0, 0.01, 0.1, 1, 10];

% Training seeds
seedList = [42, 123, 456];

% Case 1: state-dependent dynamic mismatch
% alpha = 0 is the matched nominal case as exp(0*p_t)=1
alphaList = [-0.5, -0.25, 0, 0.25, 0.5];

% Training settings
numEpochs = 300;
miniBatchSize = 256;
learnRate = 1e-3;

% Test noise setting
% Use sigma to test shifted stochastic dynamics.
% Use 0 to isolate deterministic state-dependent mismatch.
testSigma = sigma;
% testSigma = 0;

% Optional safety clip for exp(alpha*p_t)
% false = exact Case 1 equation
% true  = use exp(alpha*clip(p_t,-1,1)) to avoid numerical explosion
useClippedPosition = false;

numLambdas = length(lambdaList);
numSeeds = length(seedList);
numAlphas = length(alphaList);

%% Store results: lambda x alpha x seed

nextStateMSE_all = zeros(numLambdas, numAlphas, numSeeds);
positionMSE_all = zeros(numLambdas, numAlphas, numSeeds);
velocityMSE_all = zeros(numLambdas, numAlphas, numSeeds);
overallRMSE_all = zeros(numLambdas, numAlphas, numSeeds);

% Residual measured against Case 1 shifted test dynamics
odeResidualMSE_all = zeros(numLambdas, numAlphas, numSeeds);

% Nominal residual discrepancy for later alert layer
nominalDiscrepancyMean_all = zeros(numLambdas, numAlphas, numSeeds);
nominalDiscrepancyMSE_all = zeros(numLambdas, numAlphas, numSeeds);

% Store raw D_t values for threshold analysis later
nominalDiscrepancy_cell = cell(numLambdas, numAlphas, numSeeds);

trainingTime_all = zeros(numLambdas, numSeeds);

%% Training data from original dynamics

X = [S, A];      % [p_t, v_t, a_t]
Y = Snext;      % original next state from nominal dynamics + noise

N = size(X, 1);
Ntrain = round(0.8 * N);

%% Main loop

for l = 1:numLambdas

    lambdaODE = lambdaList(l);

    for s = 1:numSeeds

        seed = seedList(s);
        rng(seed);

        fprintf("\nTraining lambdaODE = %.4g, seed = %d\n", lambdaODE, seed);

        % Train/test split controlled by seed
        idx = randperm(N);

        trainIdx = idx(1:Ntrain);
        testIdx = idx(Ntrain+1:end);

        Xtrain = X(trainIdx, :);
        Ytrain = Y(trainIdx, :);

        Xtest = X(testIdx, :);

        %% Neural network architecture

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

        %% Training

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

        %% Predict once on Xtest

        XtestDL = dlarray(Xtest', "CB");
        YpredDL = predict(net, XtestDL);
        Ypred = extractdata(YpredDL)';

        %% Evaluate under Case 1 state-dependent mismatch

        for aIdx = 1:numAlphas

            alpha = alphaList(aIdx);

            % Current test state and action
            p_t = Xtest(:,1);
            v_t = Xtest(:,2);
            a_t = Xtest(:,3);

            % Case 1 state-dependent scale:
            % g(s_t) = exp(alpha * p_t)
            if useClippedPosition
                p_for_exp = max(min(p_t, 1), -1);
            else
                p_for_exp = p_t;
            end

            g_t = exp(alpha .* p_for_exp);

            % Case 1 shifted true dynamics:
            % p_{t+1} = p_t + dt*v_t
            % v_{t+1} = v_t + dt*exp(alpha*p_t)*a_t
            p_true_shift = p_t + dt .* v_t;
            v_true_shift = v_t + dt .* g_t .* a_t;

            % Add Gaussian noise to shifted test condition
            rng(seed + 1000*aIdx);
            testNoise = testSigma * randn(size(Xtest,1), 2);

            YtestShift = [p_true_shift, v_true_shift] + testNoise;

            %% Nominal residual discrepancy for alert layer
            % This checks whether the observed transition violates the assumed nominal ODE:
            % p_dot = v
            % v_dot = a_t

            p_obs = YtestShift(:,1);
            v_obs = YtestShift(:,2);

            d_p_nominal = (p_obs - p_t) ./ dt - v_t;
            d_v_nominal = (v_obs - v_t) ./ dt - a_t;

            D_t = sqrt(d_p_nominal.^2 + d_v_nominal.^2);

            nominalDiscrepancyMean = mean(D_t);
            nominalDiscrepancyMSE = mean(D_t.^2);

            %% Evaluation metrics under Case 1 shifted dynamics

            nextStateMSE = mean(sum((Ypred - YtestShift).^2, 2));

            positionMSE = mean((Ypred(:,1) - YtestShift(:,1)).^2);
            velocityMSE = mean((Ypred(:,2) - YtestShift(:,2)).^2);

            overallRMSE = sqrt(mean((Ypred - YtestShift).^2, "all"));

            %% ODE residual MSE under Case 1 shifted dynamics
            % This checks consistency with:
            % p_dot = v
            % v_dot = exp(alpha*p_t)*a_t

            p_pred = Ypred(:,1);
            v_pred = Ypred(:,2);

            r_p = (p_pred - p_t) ./ dt - v_t;
            r_v = (v_pred - v_t) ./ dt - g_t .* a_t;

            odeResidualMSE = mean(r_p.^2 + r_v.^2);

            %% Store results

            nextStateMSE_all(l, aIdx, s) = nextStateMSE;
            positionMSE_all(l, aIdx, s) = positionMSE;
            velocityMSE_all(l, aIdx, s) = velocityMSE;
            overallRMSE_all(l, aIdx, s) = overallRMSE;
            odeResidualMSE_all(l, aIdx, s) = odeResidualMSE;

            nominalDiscrepancyMean_all(l, aIdx, s) = nominalDiscrepancyMean;
            nominalDiscrepancyMSE_all(l, aIdx, s) = nominalDiscrepancyMSE;
            nominalDiscrepancy_cell{l, aIdx, s} = D_t;

            fprintf("Test alpha = %.2f | Next-state MSE: %.6f | ODE residual MSE: %.6f | Nominal D mean: %.6f\n", ...
                alpha, nextStateMSE, odeResidualMSE, nominalDiscrepancyMean);
        end

        fprintf("Training time: %.2f seconds\n", trainingTime);
    end
end

%% Collapse across seeds: lambda x alpha

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

meanNominalDiscrepancy = mean(nominalDiscrepancyMean_all, 3);
stdNominalDiscrepancy = std(nominalDiscrepancyMean_all, 0, 3);

meanNominalDiscrepancyMSE = mean(nominalDiscrepancyMSE_all, 3);
stdNominalDiscrepancyMSE = std(nominalDiscrepancyMSE_all, 0, 3);

%% Build long results table

rows = table();

for l = 1:numLambdas
    for aIdx = 1:numAlphas

        newRow = table( ...
            lambdaList(l), ...
            alphaList(aIdx), ...
            meanNextStateMSE(l,aIdx), stdNextStateMSE(l,aIdx), ...
            meanPositionMSE(l,aIdx), stdPositionMSE(l,aIdx), ...
            meanVelocityMSE(l,aIdx), stdVelocityMSE(l,aIdx), ...
            meanOverallRMSE(l,aIdx), stdOverallRMSE(l,aIdx), ...
            meanODEResidualMSE(l,aIdx), stdODEResidualMSE(l,aIdx), ...
            meanNominalDiscrepancy(l,aIdx), stdNominalDiscrepancy(l,aIdx), ...
            meanNominalDiscrepancyMSE(l,aIdx), stdNominalDiscrepancyMSE(l,aIdx), ...
            'VariableNames', { ...
            'lambdaODE', 'alpha', ...
            'NextStateMSE_mean', 'NextStateMSE_std', ...
            'PositionMSE_mean', 'PositionMSE_std', ...
            'VelocityMSE_mean', 'VelocityMSE_std', ...
            'OverallRMSE_mean', 'OverallRMSE_std', ...
            'ODEResidualMSE_mean', 'ODEResidualMSE_std', ...
            'NominalDiscrepancyMean_mean', 'NominalDiscrepancyMean_std', ...
            'NominalDiscrepancyMSE_mean', 'NominalDiscrepancyMSE_std' ...
            });

        rows = [rows; newRow];

    end
end

resultsTable = rows;

disp(resultsTable);

%% Normalised AUC summary across alpha values

alphaRange = max(alphaList) - min(alphaList);

aucRows = table();

for l = 1:numLambdas

    aucNextStateMSE = trapz(alphaList, meanNextStateMSE(l,:)) / alphaRange;
    aucODEResidualMSE = trapz(alphaList, meanODEResidualMSE(l,:)) / alphaRange;
    aucOverallRMSE = trapz(alphaList, meanOverallRMSE(l,:)) / alphaRange;
    aucNominalDiscrepancy = trapz(alphaList, meanNominalDiscrepancy(l,:)) / alphaRange;
    aucNominalDiscrepancyMSE = trapz(alphaList, meanNominalDiscrepancyMSE(l,:)) / alphaRange;

    newAUCRow = table( ...
        lambdaList(l), ...
        aucNextStateMSE, ...
        aucODEResidualMSE, ...
        aucOverallRMSE, ...
        aucNominalDiscrepancy, ...
        aucNominalDiscrepancyMSE, ...
        'VariableNames', { ...
        'lambdaODE', ...
        'NormAUC_NextStateMSE', ...
        'NormAUC_ODEResidualMSE', ...
        'NormAUC_OverallRMSE', ...
        'NormAUC_NominalDiscrepancyMean', ...
        'NormAUC_NominalDiscrepancyMSE' ...
        });

    aucRows = [aucRows; newAUCRow];

end

aucTable = aucRows;

disp(aucTable);

%% Save results

save("case1_state_dependent_results.mat", ...
    "lambdaList", "seedList", "alphaList", ...
    "testSigma", "useClippedPosition", ...
    "nextStateMSE_all", "positionMSE_all", "velocityMSE_all", ...
    "overallRMSE_all", "odeResidualMSE_all", ...
    "nominalDiscrepancyMean_all", "nominalDiscrepancyMSE_all", ...
    "nominalDiscrepancy_cell", ...
    "trainingTime_all", ...
    "meanNextStateMSE", "stdNextStateMSE", ...
    "meanPositionMSE", "stdPositionMSE", ...
    "meanVelocityMSE", "stdVelocityMSE", ...
    "meanOverallRMSE", "stdOverallRMSE", ...
    "meanODEResidualMSE", "stdODEResidualMSE", ...
    "meanNominalDiscrepancy", "stdNominalDiscrepancy", ...
    "meanNominalDiscrepancyMSE", "stdNominalDiscrepancyMSE", ...
    "resultsTable", "aucTable");

writetable(resultsTable, "case1_state_dependent_results.csv");
writetable(aucTable, "case1_state_dependent_auc_summary.csv");

%% Plot next-state MSE for each alpha

figure;
hold on;
for aIdx = 1:numAlphas
    errorbar(lambdaList, meanNextStateMSE(:,aIdx), stdNextStateMSE(:,aIdx), ...
        "-o", "LineWidth", 1.5);
end
xlabel("\lambda_{ODE}");
ylabel("Next-state MSE");
title("Case 1 state-dependent mismatch: Next-state MSE");
legend("alpha=-0.5", "alpha=-0.25", "alpha=0", "alpha=0.25", "alpha=0.5", ...
    "Location", "best");
grid on;
hold off;

%% Plot ODE residual MSE for each alpha

figure;
hold on;
for aIdx = 1:numAlphas
    errorbar(lambdaList, meanODEResidualMSE(:,aIdx), stdODEResidualMSE(:,aIdx), ...
        "-o", "LineWidth", 1.5);
end
xlabel("\lambda_{ODE}");
ylabel("ODE residual MSE");
title("Case 1 state-dependent mismatch: ODE residual MSE");
legend("alpha=-0.5", "alpha=-0.25", "alpha=0", "alpha=0.25", "alpha=0.5", ...
    "Location", "best");
grid on;
hold off;

%% Plot nominal discrepancy for alert layer

figure;
hold on;
for l = 1:numLambdas
    plot(alphaList, meanNominalDiscrepancy(l,:), "-o", "LineWidth", 1.5);
end
xlabel("\alpha");
ylabel("Mean nominal discrepancy D_t");
title("Case 1: Nominal residual discrepancy for alert layer");
legend("\lambda=0", "\lambda=0.01", "\lambda=0.1", "\lambda=1", "\lambda=10", ...
    "Location", "best");
grid on;
hold off;

%% Heatmap: next-state MSE

figure;
imagesc(alphaList, 1:numLambdas, meanNextStateMSE);
colorbar;
yticks(1:numLambdas);
yticklabels(string(lambdaList));
xlabel("\alpha");
ylabel("\lambda_{ODE}");
title("Case 1 heatmap: Next-state MSE");

%% Heatmap: nominal discrepancy MSE

figure;
imagesc(alphaList, 1:numLambdas, meanNominalDiscrepancyMSE);
colorbar;
yticks(1:numLambdas);
yticklabels(string(lambdaList));
xlabel("\alpha");
ylabel("\lambda_{ODE}");
title("Case 1 heatmap: Nominal discrepancy MSE");

fprintf("\nCase 1 state-dependent mismatch experiment completed.\n");
fprintf("Saved: case1_state_dependent_results.mat\n");
fprintf("Saved: case1_state_dependent_results.csv\n");
fprintf("Saved: case1_state_dependent_auc_summary.csv\n");