clear; clc; close all;

load("point_mass_dataset.mat");

%% ============================================================
% Case 2: Gaussian velocity disturbance
%
% Nominal training dynamics:
%   p_dot = v
%   v_dot = a_t
%
% Case 2 evaluation dynamics:
%   p_{t+1} = p_t + dt*v_t
%   v_{t+1} = v_t + dt*a_t + epsilon_t
%   epsilon_t ~ N(0, sigma^2)
%
% This tests robustness under Gaussian stochastic disturbance.
%% ============================================================

% ODE coefficient ablation
lambdaList = [0, 0.01, 0.1, 1, 10];

% Training seeds
seedList = [42, 123, 456];

% Case 2: Gaussian noise levels
% sigma = 0 is deterministic
% sigma = 0.01 will be used later as noisy baseline for alert threshold
sigmaList = [0, 0.01, 0.05, 0.1, 0.2];

% Training settings
numEpochs = 300;
miniBatchSize = 256;
learnRate = 1e-3;

numLambdas = length(lambdaList);
numSeeds = length(seedList);
numSigmas = length(sigmaList);

%% Store results: lambda x sigma x seed

nextStateMSE_all = zeros(numLambdas, numSigmas, numSeeds);
positionMSE_all = zeros(numLambdas, numSigmas, numSeeds);
velocityMSE_all = zeros(numLambdas, numSigmas, numSeeds);
overallRMSE_all = zeros(numLambdas, numSigmas, numSeeds);

% ODE residual of model prediction against nominal deterministic physics
odeResidualMSE_all = zeros(numLambdas, numSigmas, numSeeds);

% Nominal residual discrepancy for alert layer
nominalDiscrepancyMean_all = zeros(numLambdas, numSigmas, numSeeds);
nominalDiscrepancyMSE_all = zeros(numLambdas, numSigmas, numSeeds);

% Store raw D_t values for threshold/distribution analysis later
nominalDiscrepancy_cell = cell(numLambdas, numSigmas, numSeeds);

% Store per-sample prediction error for threshold risk analysis later
predictionError_cell = cell(numLambdas, numSigmas, numSeeds);

trainingTime_all = zeros(numLambdas, numSeeds);

%% Training data from original dynamics

X = [S, A];      % [p_t, v_t, a_t]
Y = Snext;      % original next state from nominal dynamics + dataset noise

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

        %% Evaluate under Case 2 Gaussian velocity disturbance

        for sigIdx = 1:numSigmas

            sigmaLevel = sigmaList(sigIdx);

            % Current test state and action
            p_t = Xtest(:,1);
            v_t = Xtest(:,2);
            a_t = Xtest(:,3);

            % Case 2 Gaussian disturbance:
            % p_{t+1} = p_t + dt*v_t
            % v_{t+1} = v_t + dt*a_t + epsilon_t
            % epsilon_t ~ N(0, sigma^2)

            rng(seed + 1000*sigIdx);

            eps_v = sigmaLevel * randn(size(Xtest,1), 1);

            p_true_shift = p_t + dt .* v_t;
            v_true_shift = v_t + dt .* a_t + eps_v;

            YtestShift = [p_true_shift, v_true_shift];

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

            %% Evaluation metrics under Case 2 Gaussian disturbance

            predictionError_t = sum((Ypred - YtestShift).^2, 2);
            nextStateMSE = mean(predictionError_t);

            positionMSE = mean((Ypred(:,1) - YtestShift(:,1)).^2);
            velocityMSE = mean((Ypred(:,2) - YtestShift(:,2)).^2);

            overallRMSE = sqrt(mean((Ypred(:) - YtestShift(:)).^2));

            %% ODE residual MSE under nominal deterministic physics
            % The Gaussian disturbance is treated as stochastic deviation,
            % not as part of the deterministic ODE prior.

            p_pred = Ypred(:,1);
            v_pred = Ypred(:,2);

            r_p = (p_pred - p_t) ./ dt - v_t;
            r_v = (v_pred - v_t) ./ dt - a_t;

            odeResidualMSE = mean(r_p.^2 + r_v.^2);

            %% Store results

            nextStateMSE_all(l, sigIdx, s) = nextStateMSE;
            positionMSE_all(l, sigIdx, s) = positionMSE;
            velocityMSE_all(l, sigIdx, s) = velocityMSE;
            overallRMSE_all(l, sigIdx, s) = overallRMSE;
            odeResidualMSE_all(l, sigIdx, s) = odeResidualMSE;

            nominalDiscrepancyMean_all(l, sigIdx, s) = nominalDiscrepancyMean;
            nominalDiscrepancyMSE_all(l, sigIdx, s) = nominalDiscrepancyMSE;

            nominalDiscrepancy_cell{l, sigIdx, s} = D_t;
            predictionError_cell{l, sigIdx, s} = predictionError_t;

            fprintf("Test sigma = %.3f | Next-state MSE: %.6f | ODE residual MSE: %.6f | Nominal D mean: %.6f\n", ...
                sigmaLevel, nextStateMSE, odeResidualMSE, nominalDiscrepancyMean);
        end

        fprintf("Training time: %.2f seconds\n", trainingTime);
    end
end

%% Collapse across seeds: lambda x sigma

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
    for sigIdx = 1:numSigmas

        newRow = table( ...
            lambdaList(l), ...
            sigmaList(sigIdx), ...
            meanNextStateMSE(l,sigIdx), stdNextStateMSE(l,sigIdx), ...
            meanPositionMSE(l,sigIdx), stdPositionMSE(l,sigIdx), ...
            meanVelocityMSE(l,sigIdx), stdVelocityMSE(l,sigIdx), ...
            meanOverallRMSE(l,sigIdx), stdOverallRMSE(l,sigIdx), ...
            meanODEResidualMSE(l,sigIdx), stdODEResidualMSE(l,sigIdx), ...
            meanNominalDiscrepancy(l,sigIdx), stdNominalDiscrepancy(l,sigIdx), ...
            meanNominalDiscrepancyMSE(l,sigIdx), stdNominalDiscrepancyMSE(l,sigIdx), ...
            'VariableNames', { ...
            'lambdaODE', 'sigma', ...
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

%% Normalised AUC summary across sigma values

sigmaRange = max(sigmaList) - min(sigmaList);

aucRows = table();

for l = 1:numLambdas

    aucNextStateMSE = trapz(sigmaList, meanNextStateMSE(l,:)) / sigmaRange;
    aucODEResidualMSE = trapz(sigmaList, meanODEResidualMSE(l,:)) / sigmaRange;
    aucOverallRMSE = trapz(sigmaList, meanOverallRMSE(l,:)) / sigmaRange;
    aucNominalDiscrepancy = trapz(sigmaList, meanNominalDiscrepancy(l,:)) / sigmaRange;
    aucNominalDiscrepancyMSE = trapz(sigmaList, meanNominalDiscrepancyMSE(l,:)) / sigmaRange;

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

save("case2_gaussian_results.mat", ...
    "lambdaList", "seedList", "sigmaList", ...
    "nextStateMSE_all", "positionMSE_all", "velocityMSE_all", ...
    "overallRMSE_all", "odeResidualMSE_all", ...
    "nominalDiscrepancyMean_all", "nominalDiscrepancyMSE_all", ...
    "nominalDiscrepancy_cell", "predictionError_cell", ...
    "trainingTime_all", ...
    "meanNextStateMSE", "stdNextStateMSE", ...
    "meanPositionMSE", "stdPositionMSE", ...
    "meanVelocityMSE", "stdVelocityMSE", ...
    "meanOverallRMSE", "stdOverallRMSE", ...
    "meanODEResidualMSE", "stdODEResidualMSE", ...
    "meanNominalDiscrepancy", "stdNominalDiscrepancy", ...
    "meanNominalDiscrepancyMSE", "stdNominalDiscrepancyMSE", ...
    "resultsTable", "aucTable");

writetable(resultsTable, "case2_gaussian_results.csv");
writetable(aucTable, "case2_gaussian_auc_summary.csv");

%% Plot next-state MSE for each sigma

figure;
hold on;
for sigIdx = 1:numSigmas
    errorbar(lambdaList, meanNextStateMSE(:,sigIdx), stdNextStateMSE(:,sigIdx), ...
        "-o", "LineWidth", 1.5);
end
xlabel("\lambda_{ODE}");
ylabel("Next-state MSE");
title("Case 2 Gaussian disturbance: Next-state MSE");
legend("\sigma=0", "\sigma=0.01", "\sigma=0.05", "\sigma=0.1", "\sigma=0.2", ...
    "Location", "best");
grid on;
hold off;

%% Plot ODE residual MSE for each sigma

figure;
hold on;
for sigIdx = 1:numSigmas
    errorbar(lambdaList, meanODEResidualMSE(:,sigIdx), stdODEResidualMSE(:,sigIdx), ...
        "-o", "LineWidth", 1.5);
end
xlabel("\lambda_{ODE}");
ylabel("ODE residual MSE");
title("Case 2 Gaussian disturbance: ODE residual MSE");
legend("\sigma=0", "\sigma=0.01", "\sigma=0.05", "\sigma=0.1", "\sigma=0.2", ...
    "Location", "best");
grid on;
hold off;

%% Plot nominal discrepancy for alert layer

figure;
hold on;
for l = 1:numLambdas
    plot(sigmaList, meanNominalDiscrepancy(l,:), "-o", "LineWidth", 1.5);
end
xlabel("\sigma");
ylabel("Mean nominal discrepancy D_t");
title("Case 2: Nominal residual discrepancy for alert layer");
legend("\lambda=0", "\lambda=0.01", "\lambda=0.1", "\lambda=1", "\lambda=10", ...
    "Location", "best");
grid on;
hold off;

%% Heatmap: next-state MSE

figure;
imagesc(sigmaList, 1:numLambdas, meanNextStateMSE);
colorbar;
yticks(1:numLambdas);
yticklabels(string(lambdaList));
xlabel("\sigma");
ylabel("\lambda_{ODE}");
title("Case 2 heatmap: Next-state MSE");

%% Heatmap: nominal discrepancy MSE

figure;
imagesc(sigmaList, 1:numLambdas, meanNominalDiscrepancyMSE);
colorbar;
yticks(1:numLambdas);
yticklabels(string(lambdaList));
xlabel("\sigma");
ylabel("\lambda_{ODE}");
title("Case 2 heatmap: Nominal discrepancy MSE");

fprintf("\nCase 2 Gaussian disturbance experiment completed.\n");
fprintf("Saved: case2_gaussian_results.mat\n");
fprintf("Saved: case2_gaussian_results.csv\n");
fprintf("Saved: case2_gaussian_auc_summary.csv\n");

%% ============================================================
% Local loss function
% If you already have modelLoss.m as a separate file, this local function
% is still fine. If MATLAB complains about duplicate function names,
% remove this section and keep your separate modelLoss.m file.
%% ============================================================

function [loss, gradients] = modelLoss(net, XBatch, YBatch, dt, lambdaODE)

    YPred = forward(net, XBatch);

    % Transition prediction loss
    transitionLoss = mean(sum((YPred - YBatch).^2, 1));

    % Input components
    p_t = XBatch(1,:);
    v_t = XBatch(2,:);
    a_t = XBatch(3,:);

    % Predicted next state
    p_pred = YPred(1,:);
    v_pred = YPred(2,:);

    % Nominal ODE residual:
    % p_dot = v
    % v_dot = a_t
    r_p = (p_pred - p_t) ./ dt - v_t;
    r_v = (v_pred - v_t) ./ dt - a_t;

    odeLoss = mean(r_p.^2 + r_v.^2);

    % Total loss
    loss = transitionLoss + lambdaODE * odeLoss;

    gradients = dlgradient(loss, net.Learnables);

end