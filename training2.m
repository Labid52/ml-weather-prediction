clc;
clear;
close all;
warning off;
 
%% reading data from the dataset
filename = 'Data.xlsx';
sheet = 1;
xlRange = 'K3:M3506';
 
Data = xlsread(filename, sheet, xlRange);
inputs = Data(:,1:2)';
targets = Data(:,3)';
[m,n] = size(inputs);

%% create and train network
net1 = fitnet(10,'trainbr');
net2 = fitnet(10,'traingd');
view(net1)
view(net2)
[net1,tr1] = train(net1,inputs,targets);
[net2,tr2] = train(net2,inputs,targets);
%% result
y1 = net1(inputs);
y2 = net2(inputs);
perf1 = perform(net1,y1,targets)
perf2 = perform(net2,y2,targets)
figure,
plotperform(tr1)
figure,
plotperform(tr2)