function [mdl, Records] = Train_DT(Input, Target)

setdemorandstream(491218382)
[trnx,trny,valx,valy,tsx,tsy] = Split_data(Input,Target);
mdl = fitrtree(trnx,trny);

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


% Manual calculation of performance
mse_train = mean((Predictor - trny).^2);
mse_val =  mean((y_pval - valy).^2);
mse_ts =  mean((y_pts - tsy).^2);
% if fig =='on'
%     figure
%     histogram(ResiDue);
%     xlabel('Residuals');
%     ylabel('Frequency');
%     title('Histogram of Residuals');
%     
%     figure
%     plot(lm)
%     title('Train R =',num2str(R_trn))
%     % Custom plot of residuals
%     figure;
%     plot(ResiDue, 'o');
%     xlabel('Observation Index');
%     ylabel('Residual');
%     title('Residual Plot and MSE =',num2str(mse_train));
% elseif fig=='off'
%     fprintf('figure is off')
% end


% Records for performance metrics
Records.mse_train = mse_train;
Records.mse_val = mse_val;
Records.mse_ts = mse_ts;
Records.R_train = R_trn;
Records.R_validation = R_val;
Records.R_test = R_ts;



end