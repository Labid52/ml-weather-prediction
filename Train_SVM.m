function [mdl, Records] = Train_SVM(Input, Target)

setdemorandstream(491218382)
[trnx,trny,valx,valy,tsx,tsy] = Split_data(Input,Target);
mdl =fitrsvm(trnx, trny, 'KernelFunction',...
    'gaussian');

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

%     figure
%     histogram(ResiDue);
%     xlabel('Residuals');
%     ylabel('Frequency');
%     title('Histogram of Residuals');
    
%     figure
%     plot(mdl)
%     title('Train R =',num2str(R_trn))
%     % Custom plot of residuals
%     figure;
%     plot(ResiDue, 'o');
%     xlabel('Observation Index');
%     ylabel('Residual');
%     title('Residual Plot and MSE =',num2str(mse_train));


train_result = mdl.predict(Input);
max_data = 37.1277777;
min_data = -14.08888889;
train_result = ((train_result-0)*(max_data-min_data)/1)+min_data;
target_dataset = ((Target-0)*(max_data-min_data)/1)+min_data;
figure,
plot(train_result,'bo-')
hold on
plot(target_dataset,'ro-')
hold off
grid on
% title(strcat(['Predictiton vs Target with MSE Value = ',...
% num2str(error_MSE)]))
xlabel('Datapoints')
ylabel('Temperature')
legend('Prediction','Target','Location','Best')


% Records for performance metrics
Records.mse_train = mse_train;
Records.mse_val = mse_val;
Records.mse_ts = mse_ts;
Records.R_train = R_trn;
Records.R_validation = R_val;
Records.R_test = R_ts;



end