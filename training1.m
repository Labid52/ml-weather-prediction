clc;
clear;
close all;
warning off;
 
%% Proses membaca data latih dari excel
filename = 'Data.xlsx';
sheet = 1;
xlRange = 'K3:M3506';
% xlRange = 'P3:R878';
 
Data = xlsread(filename, sheet, xlRange);
data_latih = Data(:,1:2)';
target_latih = Data(:,3)';
[m,n] = size(data_latih);
 
%% Pembuatan JST
net = feedforwardnet([20 20], 'trainbr');
 
% Memberikan nilai untuk mempengaruhi proses pelatihan
net.performFcn = 'mse';
net.trainParam.goal = 0.001;
net.trainParam.show = 20;
net.trainParam.epochs = 1000;
net.trainParam.mc = 0.95;
net.trainParam.lr = 0.1;
 
%% Proses training
[net_keluaran,tr,Y,E] = train(net,data_latih,target_latih);
 
% Hasil setelah pelatihan
% bobot_hidden_1 = net_keluaran.IW{1,1};
% bobot_hidden_2 = net_keluaran.IW{2,1};
% bobot_keluaran = net_keluaran.LW{3,2};
% bias_hidden_1 = net_keluaran.b{1,1};
% bias_hidden_2 = net_keluaran.b{2,1};
% bias_keluaran = net_keluaran.b{3,1};
jumlah_iterasi = tr.num_epochs;
nilai_keluaran = Y;
nilai_error = E;
error_MSE = (1/n)*sum(nilai_error.^2);
error_per = sqrt(error_MSE) / mean(nilai_keluaran) * 100
 
save net.mat net_keluaran
 
%% prediction result
hasil_latih = sim(net_keluaran,data_latih);
max_data = 31.4;
min_data = 13;
hasil_latih = ((hasil_latih-0)*(max_data-min_data)/1)+min_data;
 
%% Performansi hasil prediksi
filename = 'Data.xlsx';
sheet = 1;
xlRange = 'C3:C3506';
 
target_latih_asli = xlsread(filename, sheet, xlRange);
 
figure,
plotregression(target_latih_asli,hasil_latih,'Regression')
 
figure,
plotperform(tr)
 
figure,
plot(hasil_latih,'bo-')
hold on
plot(target_latih_asli,'ro-')
hold off
grid on
title(strcat(['Grafik Keluaran JST vs Target dengan nilai MSE = ',...
num2str(error_MSE)]))
xlabel('Pola ke-')
ylabel('Suhu')
legend('Keluaran JST','Target','Location','Best')
