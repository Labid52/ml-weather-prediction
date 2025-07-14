function [mdl, Records] = Train_KNN(Input, Target)

setdemorandstream(491218382)
[trnx,trny,valx,valy,tsx,tsy] = Split_data(Input,Target);
K = 10; % Set the value of K (number of nearest neighbors to consider)
DistMetric = 'euclidean'; % Set the distance metric (e.g., 'euclidean', 'manhattan', 'cosine', etc.)

% Train KNN model on the training set
mdl = fitcknn(trnx, trny, 'NumNeighbors', K, 'Distance', DistMetric);

% % Extract coefficients and statistics
% Coefficients = mdl.Coefficients;
% Stats = mdl.Rsquared;




% Predict on training data
Predictor = predict(mdl, trnx);
y_pval = predict(mdl,valx);
y_pts = predict(mdl,tsx);

% Calculate the correlation coefficient on the validation dataset
trn_corr = corrcoef(trny,Predictor);
R_trn = trn_corr(1,2);
val_corr = corrcoef(valy, y_pval);
R_val = val_corr(1, 2); % Extract the correlation coefficient value
ts_corr = corrcoef(tsy, y_pts);
R_ts = ts_corr(1, 2); % Extract the correlation coefficient value

Accuracy_val = sum(y_pval == valy) / numel(valy);
Accuracy_ts = sum(y_pts == tsy) / numel(tsy);


% Manual calculation of performance
mse_train = mean((Predictor - trny).^2);
mse_val =  mean((y_pval - valy).^2);
mse_ts =  mean((y_pts - tsy).^2);



% Records for performance metrics
Records.mse_train = mse_train;
Records.mse_val = mse_val;
Records.mse_ts = mse_ts;
Records.R_train = R_trn;
Records.R_validation = R_val;
Records.R_test = R_ts;
Records.accuray_validation = Accuracy_val;
Records.accuray_tes = Accuracy_ts;



end