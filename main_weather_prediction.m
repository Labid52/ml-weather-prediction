clc
close all
clear
tic
%% Load Dataset
[trn, trg] = Load_Data('weather dataset')
%% network training 
[Pred, Rec] = Train_ANN(trn,trg)

%[Pred, Rec] = Train_RNN(trn,trg)
%[Pred, rec] = Train_LinReg(trn,trg)
%[Pred,rec] = Train_DT(trn,trg)
%[Pred,rec] = Train_RanFor(trn,trg)
% [Pred,rec] = Train_SVM(trn,trg)
% [Pred,rec] = Train_KNN(trn,trg)
% [Pred,rec] = Train_NaiBai(trn,trg)
 %[Pred,rec] = Train_PCA(trn,trg)
 %[Pred,rec] = Train_RidReg(trn,trg)
 %[Pred,rec] = Train_Lasso(trn,trg)
 %[Pred,rec] = Train_SVR(trn,trg)
  %[Pred, Rec] = Train_CNN(trn,trg);
  toc;




%% testing
data = [0.473 0.6];
% fn = Test_Network(Pred,data);
% fn = predict()


