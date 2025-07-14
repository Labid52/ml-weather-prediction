function [lm, Records] = Train_LogReg(Input, Target)

setdemorandstream(491218382)
[trnx,trny,valx,valy,tsx,tsy] = Split_data(Input,Target);
lm = fitglm(trnx,trny,'link','logit');
ResiDue = lm.Residuals.Raw;

% Predict on training data
Predictor = predict(lm, trnx);
y_pval = predict(lm, valx);
y_pts = predict(lm, tsx);

% Calculate the classification accuracy on the training dataset
y_pred_trn = round(Predictor);
acc_trn = sum(y_pred_trn == trny) / numel(trny);

% Calculate the classification accuracy on the validation dataset
y_pred_val = round(y_pval);
acc_val = sum(y_pred_val == valy) / numel(valy);

% Calculate the classification accuracy on the test dataset
y_pred_ts = round(y_pts);
acc_ts = sum(y_pred_ts == tsy) / numel(tsy);

% Records for performance metrics
Records.accuracy_train = acc_trn;
Records.accuracy_validation = acc_val;
Records.accuracy_test = acc_ts;
Records.coefficients = lm.Coefficients;
Records.Residuals = mean(ResiDue);

end
