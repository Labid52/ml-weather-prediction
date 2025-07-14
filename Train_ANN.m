function [predictor, Records] = Train_ANN(Input_Dataset, Target_Dataset)

Learning_Algorithm = 'trainlm';
Network_Architecture = [3];

setdemorandstream(491218382)

The_Network = fitnet(Network_Architecture, Learning_Algorithm);

% The_Network.divideFcn = 'dividerand';
% The_Network.divideMode = 'sample';
% The_Network.divideParam.trainRatio = 70/100;
% The_Network.divideParam.testRatio = 15/100;
% The_Network.divideParam.valRatio = 15/100;
train_ratio = 0.8;
val_ratio = 0.0;
test_ratio = 0.2;
[trainInd,valInd,testInd] = dividerand(size(Input_Dataset,1),train_ratio,val_ratio,test_ratio);
The_Network.divideFcn = 'divideind';
The_Network.divideMode = 'sample';
The_Network.divideParam.trainInd = trainInd;
The_Network.divideParam.valInd = valInd;
The_Network.divideParam.testInd = testInd;


The_Network.performFcn = 'mse';
The_Network.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotregression'};

[predictor, Records] = train(The_Network, Input_Dataset', Target_Dataset');


train_result = predictor(Input_Dataset');
max_data = 37.1277777;
min_data = -14.08888889;
train_result = ((train_result-0)*(max_data-min_data)/1)+min_data;
target_dataset = ((Target_Dataset-0)*(max_data-min_data)/1)+min_data;

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



end