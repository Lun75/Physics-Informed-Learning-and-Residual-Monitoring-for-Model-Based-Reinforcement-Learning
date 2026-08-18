function [loss, gradients] = modelLoss(net, XBatch, YBatch, dt, lambdaODE)

    % XBatch rows:
    % 1 = p_t
    % 2 = v_t
    % 3 = a_t

    % Predicted next state
    YPred = forward(net, XBatch);

    % Transition prediction loss
    transitionLoss = mean(sum((YPred - YBatch).^2, 1));

    % Extract current state and action
    p_t = XBatch(1, :);
    v_t = XBatch(2, :);
    a_t = XBatch(3, :);

    % Extract predicted next state
    p_pred = YPred(1, :);
    v_pred = YPred(2, :);

    % ODE residual:
    % p_dot = v
    % v_dot = a_t
    r_p = (p_pred - p_t) ./ dt - v_t;
    r_v = (v_pred - v_t) ./ dt - a_t;

    % ODE residual loss
    odeLoss = mean(r_p.^2 + r_v.^2);

    % Total loss
    loss = transitionLoss + lambdaODE * odeLoss;

    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);

end