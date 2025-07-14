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
net = layrecnet(1:2,10);
% [Xs,Xi,Ai,Ts] = preparets(net,inputs,targets);
[net, tr] = train(net,inputs,targets);
%% result
y = net(inputs);
perf1 = perform(net,y,targets)
figure,
plotperform(tr)

