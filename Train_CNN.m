function [Predictor, Records] = Train_CNN(Training_Dataset, Target_Dataset)

% Define CNN architecture
Layers = [
    imageInputLayer([inputSize1 inputSize2 inputSize3]) % Input layer with specified input size
    convolution2dLayer(filterSize1, numFilters1, 'Padding', 'same') % Convolutional layer 1
    reluLayer() % ReLU activation layer
    maxPooling2dLayer(poolSize1, 'Stride', stride1) % Max pooling layer 1
    convolution2dLayer(filterSize2, numFilters2, 'Padding', 'same') % Convolutional layer 2
    reluLayer() % ReLU activation layer
    maxPooling2dLayer(poolSize2, 'Stride', stride2) % Max pooling layer 2
    fullyConnectedLayer(numHiddenUnits) % Fully connected layer
    reluLayer() % ReLU activation layer
    fullyConnectedLayer(numClasses) % Fully connected output layer
    softmaxLayer() % Softmax activation layer
    classificationLayer() % Classification layer
];

% Create and configure the CNN
The_Network = trainNetwork(Training_Dataset, Target_Dataset, Layers, options);

[Predictor, Records] = The_Network;

end
