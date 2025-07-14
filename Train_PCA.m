function [mdl, Records] = Train_PCA(Input, Target)

setdemorandstream(491218382)
[trnx,trny,valx,valy,tsx,tsy] = Split_data(Input,Target);

coeff = pca(trnx);

% Select the top k principal components to retain
k = 1; % Number of principal components to retain
selected_features = coeff(:,1:k);

% Project the training set onto the selected principal components
trnx_PCA = trnx * selected_features;

% Project the validation and testing set onto the selected principal components
valx_PCA = valx * selected_features;
tsx_PCA = tsx * selected_features;
% Train KNN model on the training set
mdl = fitcecoc(trnx_PCA,trny);

% % Extract coefficients and statistics
% Coefficients = mdl.Coefficients;
% Stats = mdl.Rsquared;




% Predict on training data
Predictor = predict(mdl, trnx_PCA);
y_pval = predict(mdl,valx_PCA);
y_pts = predict(mdl,tsx_PCA);

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