clc
close all
clear
%% Load weather dataset
warning('off')
Training_Dataset = 'weather data';
Training_Dataset_Options = detectImportOptions(Training_Dataset);
Training_Data = readtable(Training_Dataset, Training_Dataset_Options, "UseExcel", false);

%% Extract input features (x1, x2) and target variable (y1)
x1 = Training_Data.x1;
x2 = Training_Data.x2;
Training = [x1 x2];
Target = Training_Data.y1;

%% Step 1: Preprocessing (if needed)
% You can perform preprocessing steps such as feature scaling, 
% normalization, 
% handling missing values, etc., on the Training and Target data as per your specific requirements.

%% Step 2: Data splitting
rng('default'); % For reproducibility
rng(1); % Set random seed for consistent splitting

% Split the data into training set (70%), validation set (20%), and testing set (10%)
cvp = cvpartition(size(Training, 1), 'HoldOut', 0.3); % 70% training, 30% validation+testing
Training_Set = Training(cvp.training,:);
Validation_Test_Set = Training(cvp.test,:);
Target_Training_Set = Target(cvp.training,:);
Target_Validation_Test_Set = Target(cvp.test,:);

cvp2 = cvpartition(size(Validation_Test_Set, 1), 'HoldOut', 0.67); % 20% validation, 10% testing
Validation_Set = Validation_Test_Set(cvp2.training,:);
Testing_Set = Validation_Test_Set(cvp2.test,:);
Target_Validation_Set = Target_Validation_Test_Set(cvp2.training,:);
Target_Testing_Set = Target_Validation_Test_Set(cvp2.test,:);

%% Step 3: Training different models
% Train linear regression model
linearRegModel = fitlm(Training_Set, Target_Training_Set);

% Train decision tree model
dtModel = fitrtree(Training_Set, Target_Training_Set);

% Train random forest model
rfModel = TreeBagger(50, Training_Set, Target_Training_Set, 'Method', 'regression');

% Train support vector machine (SVM) model
svmModel = fitrsvm(Training_Set, Target_Training_Set, 'KernelFunction', 'gaussian');

% Train neural network model

net = Train_ANN(Training,Target);

%% Step 4: Model evaluation based on validation set
% Predict on validation set for each model
linearRegPred_Val = predict(linearRegModel, Validation_Set);
dtPred_Val = predict(dtModel, Validation_Set);
rfPred_Val = predict(rfModel, Validation_Set);
svmPred_Val = predict(svmModel, Validation_Set);
nnPred_Val = net(Validation_Set');

% Calculate mean squared error (MSE) for each model
mse_LinearReg_Val = mse(linearRegPred_Val - Target_Validation_Set);
mse_DT_Val = mse(dtPred_Val - Target_Validation_Set);
mse_RF_Val = mse(rfPred_Val - Target_Validation_Set);
mse_SVM_Val = mse(svmPred_Val - Target_Validation_Set);
mse_NN_Val = mse(nnPred_Val' - Target_Validation_Set);

%% Step 5: Model selection based on validation set
% Select model with best validation set MSE
mse_vals = [mse_LinearReg_Val, mse_DT_Val, mse_RF_Val, mse_SVM_Val, mse_NN_Val];
min_mse = min(mse_vals); % Find minimum MSE
best_model = ''; % Variable to store the best model name

switch min_mse
case mse_LinearReg_Val
best_model = 'Linear Regression';
case mse_DT_Val
best_model = 'Decision Tree';
case mse_RF_Val
best_model = 'Random Forest';
case mse_SVM_Val
best_model = 'Support Vector Machine';
case mse_NN_Val
best_model = 'Neural Network';
end

fprintf('Best Model based on Validation Set MSE: %s\n', best_model);

% Step 6: Testing the best model on test set
switch best_model
case 'Linear Regression'
linearRegPred_Test = predict(linearRegModel, Testing_Set);
mse_LinearReg_Test = mse(linearRegPred_Test - Target_Testing_Set);
fprintf('Testing Set MSE for Linear Regression: %.4f\n', mse_LinearReg_Test);
case 'Decision Tree'
dtPred_Test = predict(dtModel, Testing_Set);
mse_DT_Test = mse(dtPred_Test - Target_Testing_Set);
fprintf('Testing Set MSE for Decision Tree: %.4f\n', mse_DT_Test);
case 'Random Forest'
rfPred_Test = predict(rfModel, Testing_Set);
mse_RF_Test = mse(rfPred_Test - Target_Testing_Set);
fprintf('Testing Set MSE for Random Forest: %.4f\n', mse_RF_Test);
case 'Support Vector Machine'
svmPred_Test = predict(svmModel, Testing_Set);
mse_SVM_Test = mse(svmPred_Test - Target_Testing_Set);
fprintf('Testing Set MSE for Support Vector Machine: %.4f\n', mse_SVM_Test);
case 'Neural Network'
nnPred_Test = net(Testing_Set');
mse_NN_Test = mse(nnPred_Test' - Target_Testing_Set);
fprintf('Testing Set MSE for Neural Network: %.4f\n', mse_NN_Test);
end
