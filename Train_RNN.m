function [Predictor, Records] = Train_RNN(Input_Dataset, Target_Dataset)

Learning_Algorithm = 'trainlm';
Network_Architecture = [1:2, 10, 1:2,10];

setdemorandstream(491218382)

The_Network = layrecnet(Network_Architecture(1),Network_Architecture(2)...
    ,Learning_Algorithm);

The_Network.divideFcn = 'dividerand';
The_Network.divideMode = 'sample';
The_Network.divideParam.trainRatio = 80/100;
The_Network.divideParam.testRatio = 10/100;
The_Network.divideParam.valRatio = 10/100;


The_Network.performFcn = 'mse';
The_Network.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotregression'};

[Predictor, Records] = train(The_Network, Input_Dataset', Target_Dataset');
train_result = Predictor(Input_Dataset');
figure,
plot(train_result,'bo-')
hold on
plot(Target_Dataset,'ro-')
hold off
grid on
% title(strcat(['Predictiton vs Target with MSE Value = ',...
% num2str(error_MSE)]))
xlabel('Datapoints')
ylabel('Temperature')
legend('Prediction','Target','Location','Best')
end