

clear; clc;
close all;

% gpuDevice(2)

% -----------load data: Face with Age
CollectionDirectory = './FaceAgeData/wiki/';

part_length = 9;
part_list = randperm(part_length);  % [3 1 2 4];
FaceData.Face = [];
FaceData.Age = [];
for part_i = 1:length(part_list)-1
    save_name = strcat('WIKI_FaceData_Part', num2str(part_list(part_i)));
    LoadData = load([CollectionDirectory save_name], 'FaceData');
    FaceData.Face = cat(4, FaceData.Face, LoadData.FaceData.Face);
    FaceData.Age = cat(2, FaceData.Age, LoadData.FaceData.Age);
end
clear LoadData
trainData.Data = FaceData.Face;
trainData.Label = FaceData.Age;
clear FaceData

for part_i = length(part_list)
    save_name = strcat('WIKI_FaceData_Part', num2str(part_list(part_i)));
    load([CollectionDirectory save_name], 'FaceData');
    %     FaceData.Face = cat(4, FaceData.Face, LoadData.FaceData.Face);
    %     FaceData.Age = cat(2, FaceData.Age, LoadData.FaceData.Age);
end
testData.Data = FaceData.Face;
testData.Label = FaceData.Age;
clear FaceData

% nbins = 0:1:100;
% figure;
% subplot(121);hist(trainData.Label, nbins);
% xlim([-1 101]); grid on;

%% Rearrange the age distribution
windows.Enable = 1;
windows.HalfLength = 10;
windows.MaxValue = 100;
windows.MinValue = 0;
windows.Step = 2;
windows.nbins = windows.MinValue :windows.Step: windows.MaxValue;
numClasses = (windows.MaxValue - windows.MinValue) / windows.Step + 1;

% if windows.Step ~= 1
% trainData.Label = X;
X = trainData.Label;
for j1 = 1:length(X)
    for i1 = 1:(length(windows.nbins)-1)
        if X(j1) >= windows.nbins(i1) & X(j1) < windows.nbins(i1+1)
            trainData.Label(j1) = i1*windows.Step;
        end
    end
end

Y = testData.Label;
for j1 = 1:length(Y)
    for i1 = 1:(length(windows.nbins)-1)
        if Y(j1) >= windows.nbins(i1) & Y(j1) < windows.nbins(i1+1)
            testData.Label(j1) = i1*windows.Step;
        end
    end
end

% windows.nbins = windows.nbins(2:end);
% subplot(122);hist(trainData.Label, windows.nbins);
[counts, centers] = hist(trainData.Label, windows.nbins);
windows.nbins(find(counts == 0)) = -1;
numClasses = length(find(windows.nbins>=0));
% xlim([-1 101]); grid on;
% end
trainData.Label = categorical(trainData.Label);
testData.Label = categorical(testData.Label);

validationData.Data = testData.Data(:,:,:,1:630*2);
validationData.Label = testData.Label(1:630*2);
clear X
clear Y

%% the loop for traning task -- 7 different NNs
% protofile = 'age.prototxt';
% datafile = 'dex_imdb_wiki.caffemodel';

protofile = 'age_LAP.prototxt';
datafile = 'dex_chalearn_iccv2015.caffemodel';
net = importCaffeNetwork(protofile, datafile);
layersTransfor = 'drop7';
%construct the connection between two layers
layers = [
    imageInputLayer([1 1 4096],'Name','input','Normalization','zerocenter')
    fullyConnectedLayer(numClasses,'Name',strcat('fc',num2str(numClasses)),'WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
    softmaxLayer('Name',strcat('fc',num2str(numClasses),'_softmax'))
    classificationLayer('Name',strcat('ClassificationLayer_fc',num2str(numClasses)))
    ];  
lgraph = layerGraph(layers);

% analyzeNetwork(lgraph);

try
    load('trainData.mat');
    load('validationData.mat');
    load('testData.mat');
catch
    trainData.Data = activations(net, trainData.Data, layersTransfor);
    validationData.Data = activations(net, validationData.Data, layersTransfor);
    testData.Data = activations(net, testData.Data, layersTransfor);
    save trainData trainData
    save validationData validationData
    save testData testData
end

options = trainingOptions('adam',...
    'ExecutionEnvironment','gpu', ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',32,...
    'MaxEpochs',30,...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',13, ...
    'InitialLearnRate',0.001,...
    'SequencePaddingValue', 0,...
    'ValidationData',{validationData.Data, validationData.Label}, ...
    'ValidationFrequency',30,...
    'Plots','training-progress');
netTransfer = trainNetworkFuzzy(trainData.Data, trainData.Label, lgraph, options, windows);

% save_name = strcat('WIKI_FaceData_netTransfer_', num2str(part_list(part_i)));
% save([CollectionDirectory save_name], 'netTransfer');



%% Predict the score and Calculate the RMSE or MAE
%% ---------------------------------------------------------------
testData.Data = testData.Data(:,:,:,1:200);
testData.Label = testData.Label(1:200);
[YPred scores] = classify(net, testData.Data);
predictions_squeeze = scores';
response_size = size(predictions_squeeze);
score_sequence = [1:1:response_size(1)];

[predicted_score_row predicted_score_col] = max(predictions_squeeze);

predicted_score = zeros(1, length(predicted_score_col));
for ii = 1:length(predicted_score_col)
    
    cor_list = [predicted_score_col(ii)-windows.HalfLength : predicted_score_col(ii)+windows.HalfLength];
    cor_list = cor_list(find(cor_list >= windows.MinValue/windows.Step & cor_list <= windows.MaxValue/windows.Step));
    predicted_temp = predictions_squeeze((cor_list+1), ii);
    predicted_score(ii) = score_sequence(cor_list+1) * predicted_temp ./ sum(predicted_temp);
end
predicted_score = round(predicted_score*windows.Step) - windows.Step;


accuracy = sum(YPred == testData.Label')/numel(testData.Label)
[GN, ~, testValue] = unique(testData.Label);
RMSE = sqrt(mean(((testValue*windows.Step - windows.Step) - predicted_score').^2))/windows.Step  % Root Mean Squared Error
MAE = mae((testValue*windows.Step - windows.Step) - predicted_score')
