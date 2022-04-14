


clear; clc;
close all;

% gpuDevice(2)

% -----------load data: Face with Age
CollectionDirectory = './FaceAgeData/wiki/';

part_length = 3;
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

nbins = 0:1:100;
figure;
subplot(121);hist(trainData.Label, nbins);
xlim([-1 101]); grid on;

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
    subplot(122);hist(trainData.Label, windows.nbins);
    [counts, centers] = hist(trainData.Label, windows.nbins);
    windows.nbins(find(counts == 0)) = -1;
    numClasses = length(find(windows.nbins>=0));
    xlim([-1 101]); grid on;
% end
trainData.Label = categorical(trainData.Label);
testData.Label = categorical(testData.Label);

validationData.Data = testData.Data(:,:,:,1:630*2);
validationData.Label = testData.Label(1:630*2);
clear X
clear Y

%% the loop for traning task -- 7 different NNs
for network_num = 2 %1:num_networks
    
    if network_num == 1
        net = resnet18;
        NNname = 'Resnet18';
    else     if network_num == 2
            net = resnet50;
            NNname = 'Resnet50';
        else     if network_num == 3
                net = resnet101;
                NNname = 'Resnet101';
            else     if network_num == 4
                    net = vgg16;
                    NNname = 'VGG16';
                else     if network_num == 5
                        net = vgg19;
                        NNname = 'VGG19';
                    else if network_num == 6
                            net = googlenet;
                            NNname = 'GoogleNet';
                        else if network_num == 7
                                net = inceptionv3;
                                NNname = 'InceptionV3';
                            end
                        end
                    end
                end
            end
        end
    end
    analyzeNetwork(net);
    %%  1. transfer train; only train the last layer (fully connected layer)
%     % change the connection tabel
%     if length(net.Layers) == 72  % ResNet18
%         % change the structures of last 3 layers
%         layersTransfor = 'pool5';
%         layers = [
%             imageInputLayer([1 1 512],'Name','data','Normalization','zerocenter')
%             fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%             softmaxLayer('Name','fc4_softmax')
%             classificationLayer('Name','ClassificationLayer_fc4')
%             ];
%     else if length(net.Layers) == 177  % ResNet50
%             % change the structures of last 3 layers
%             layersTransfor = 'avg_pool';
%             layers = [
%                 imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
%                 fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                 softmaxLayer('Name','fc4_softmax')
%                 classificationLayer('Name','ClassificationLayer_fc4')
%                 ];
%         else if length(net.Layers) == 347  % ResNet101
%                 % change the structures of last 3 layers
%                 layersTransfor = 'pool5';
%                 layers = [
%                     imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
%                     fullyConnectedLayer(numClasses,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                     softmaxLayer('Name','fc2_softmax')
%                     classificationLayer('Name','ClassificationLayer_fc2')
%                     ];
%             else if length(net.Layers) == 25  % AlexNet
%                     % change the structures of last 3 layers
%                     layersTransfor = 'drop7';
%                     layers = [
%                         imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
%                         fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                         softmaxLayer('Name','fc4_softmax')
%                         classificationLayer('Name','ClassificationLayer_fc4')
%                         ];
%                 else if length(net.Layers) == 41  % VGG16
%                         % change the structures of last 3 layers
%                         layersTransfor = 'drop7';
%                         layers = [
%                             imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
%                             fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                             softmaxLayer('Name','fc4_softmax')
%                             classificationLayer('Name','ClassificationLayer_fc4')
%                             ];
%                     else if length(net.Layers) == 47  % VGG19
%                             % change the structures of last 3 layers
%                             layersTransfor = 'drop7';
%                             layers = [
%                                 imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
%                                 fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                                 softmaxLayer('Name','fc4_softmax')
%                                 classificationLayer('Name','ClassificationLayer_fc4')
%                                 ];
%                         else if length(net.Layers) == 144  % GoogleNet
%                                 % change the structures of last 3 layers
%                                 layersTransfor = 'pool5-drop_7x7_s1';
%                                 layers = [
%                                     imageInputLayer([1 1 1024],'Name','data','Normalization','zerocenter')
%                                     fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                                     softmaxLayer('Name','fc3_softmax')
%                                     classificationLayer('Name','ClassificationLayer_fc3')
%                                     ];
%                             else if length(net.Layers) == 316  % InceptionV3
%                                     % change the structures of last 3 layers
%                                     layersTransfor = 'avg_pool';
%                                     layers = [
%                                         imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
%                                         fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%                                         softmaxLayer('Name','fc3_softmax')
%                                         classificationLayer('Name','ClassificationLayer_fc3')
%                                         ];
%                                 end
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
    
    layersTransfor = net.Layers(1:end-3);
    layers = [
        layersTransfor
        fullyConnectedLayer(numClasses,'Name','fc1000','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
        softmaxLayer('Name','fc1000_softmax')
        classificationLayer('Name','ClassificationLayer_fc1000')
        ];
    
    net.Layers = layers;
    analyzeNetwork(lgraph); % show the structure of designed neural network
    
    netWidth = 64;
    numUnits = 6;
    lgraph = residualCIFARlgraph(netWidth, numUnits, "standard", numClasses);
%     analyzeNetwork(lgraph);

    % extracting some specific layers, and ploting
%     trainFeatures = activations(net, trainData.Data, layersTransfor);
%     validationFeatures = activations(net, validationData.Data, layersTransfor);
%     testFeatures = activations(net, testData.Data, layersTransfor);
    trainFeatures = trainData.Data, layersTransfor);
    validationFeatures = activations(net, validationData.Data, layersTransfor);
    testFeatures = activations(net, testData.Data, layersTransfor);
%     save trainFeatures trainFeatures
%     save validationFeatures validationFeatures
%     save testFeatures testFeatures
%     load('trainFeatures.mat');
%     load('validationFeatures.mat');
%     load('testFeatures.mat');

    
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
    
    save_name = strcat('WIKI_FaceData_netTransfer_', num2str(part_list(part_i)));
    save([CollectionDirectory save_name], 'netTransfer');
    
    
    
    %% Predict the score and Calculate the RMSE or MAE
    %% ---------------------------------------------------------------
    [YPred scores] = classify(net_initial, testFeature);
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
    
    
    accuracy = sum(YPred == testLabel)/numel(testLabel)
    [GN, ~, testValue] = unique(testLabel);
    RMSE = sqrt(mean(((testValue*windows.Step - windows.Step) - predicted_score').^2))/windows.Step  % Root Mean Squared Error
    MAE = mae((testValue*windows.Step - windows.Step) - predicted_score')
end


