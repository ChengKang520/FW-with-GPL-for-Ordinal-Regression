% 
% 
% %%
% % Comparing the DEX method and its mutation with windows
% %%
% 
% clear; clc;
% close all;
% 
% % -----------load data: Face with Age
% CollectionDirectory = './FaceAgeData/wiki/';
% fold_num = 1;
% for fold_i = 1:fold_num
%     part_length = 3;
%     part_list = [3 1 2];%randperm(part_length);  % [3 1 2 4]; [6 3 7 8 5 1 2 4 9]
%     FaceData.Face = [];
%     FaceData.Age = [];
%     for part_i = 1:length(part_list)-1
%         save_name = strcat('WIKI_FaceData_Part', num2str(part_list(part_i)));
%         LoadData = load([CollectionDirectory save_name], 'FaceData');
%         FaceData.Face = cat(4, FaceData.Face, LoadData.FaceData.Face);
%         FaceData.Age = cat(2, FaceData.Age, LoadData.FaceData.Age);
%     end
%     clear LoadData
%     trainData.Data = FaceData.Face;
%     trainData.Label = FaceData.Age;
%     clear FaceData
%     
%     for part_i = length(part_list)
%         save_name = strcat('WIKI_FaceData_Part', num2str(part_list(part_i)));
%         load([CollectionDirectory save_name], 'FaceData');
%         %     FaceData.Face = cat(4, FaceData.Face, LoadData.FaceData.Face);
%         %     FaceData.Age = cat(2, FaceData.Age, LoadData.FaceData.Age);
%     end
%     testData.Data = FaceData.Face;
%     testData.Label = FaceData.Age;
%     clear FaceData
%     
%     
%     %% draw the disttribution of the face-age data after the segment-loss
%     nbins = 0:1:100;
%     figure;
%     subplot(121);hist(trainData.Label, nbins);
%     xlim([-1 101]); grid on;
%     [counts, centers] = hist(trainData.Label, nbins);
% 
%     Label_list_cancel = [82,90,13,89,61,10,27,51,97,88,15,87,85,43,69,98,36,77,65,78]; %randperm(100,20)
%     for ii = 1:length(trainData.Label)
%         for jj = 1:length(Label_list_cancel)
%             if trainData.Label(ii) == Label_list_cancel(jj)
%                 trainData.Label(ii) = -1;
%     %             WIKI_Age_list(:,:,:,ii) = [];
%             end
%         end
%     end
%     trainData.Label(find(trainData.Label==-1)) = [];
% 
%     Label_list_cancel = find(counts < 20)
%     for ii = 1:length(trainData.Label)
%         for jj = 1:length(Label_list_cancel)
%             if trainData.Label(ii) == Label_list_cancel(jj)
%                 trainData.Label(ii) = -1;
%     %             WIKI_Age_list(:,:,:,ii) = [];
%             end
%         end
%     end
%     trainData.Label(find(trainData.Label==-1)) = [];
% 
%     subplot(122);hist(trainData.Label, nbins);
%     xlim([-1 101]); grid on;
%     
%     
%     %% Rearrange the age distribution
%     windows.Enable = 1;
%     windows.HalfLength = 20;
%     windows.MaxValue = 100;
%     windows.MinValue = 0;
%     windows.Step = 1;
%     windows.nbins = windows.MinValue :windows.Step: windows.MaxValue;
%     numClasses = (windows.MaxValue - windows.MinValue) / windows.Step + 1;
%     
%     if windows.Step ~= 1
%         % trainData.Label = X;
%         X = trainData.Label;
%         for j1 = 1:length(X)
%             for i1 = 1:(length(windows.nbins)-1)
%                 if X(j1) >= windows.nbins(i1) & X(j1) < windows.nbins(i1+1)
%                     trainData.Label(j1) = i1*windows.Step;
%                 end
%             end
%         end
%         
%         Y = testData.Label;
%         for j1 = 1:length(Y)
%             for i1 = 1:(length(windows.nbins)-1)
%                 if Y(j1) >= windows.nbins(i1) & Y(j1) < windows.nbins(i1+1)
%                     testData.Label(j1) = i1*windows.Step;
%                 end
%             end
%         end
%     end
%     
%     [counts, centers] = hist(trainData.Label, windows.nbins);
%     windows.nbins(find(counts == 0)) = -1;
%     numClasses = length(find(windows.nbins>=0));
%     trainData.Label = categorical(trainData.Label);
%     testData.Label = categorical(testData.Label);
%     
%     validationData.Data = testData.Data(:,:,:,1:630*2);
%     validationData.Label = testData.Label(1:630*2);
%     clear X
%     clear Y
%     %     clear trainData
%     %     clear validationData
%     
%     %% the loop for traning task -- 7 different NNs
%     % protofile = 'age.prototxt';
%     % datafile = 'dex_imdb_wiki.caffemodel';
%     protofile = 'age_LAP.prototxt';
%     datafile = 'dex_chalearn_iccv2015.caffemodel';
%     net = importCaffeNetwork(protofile, datafile);
%     layersTransfor = 'drop7';
%     %construct the connection between two layers
%     layers = [
%         imageInputLayer([1 1 4096],'Name','input','Normalization','zerocenter')
%         fullyConnectedLayer(numClasses,'Name',strcat('fc',num2str(numClasses)),'WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%         softmaxLayer('Name',strcat('fc',num2str(numClasses),'_softmax'))
%         classificationLayer('Name',strcat('ClassificationLayer_fc',num2str(numClasses)))
%         ];
%     lgraph = layerGraph(layers);
%     
%     % transfer from the last forth layer
% %     trainData.Data = activations(net, trainData.Data, layersTransfor);
% %     validationData.Data = activations(net, validationData.Data, layersTransfor);
% %     testData.Data = activations(net, testData.Data, layersTransfor);
%     load('C3_trainData.mat');
%     load('C3_validationData.mat');
%     load('C3_testData.mat');
% 
%     
%     %% Predict the score and Calculate the RMSE or MAE
%     %% ---------------------------------------------------------------
%     options = trainingOptions('adam',...
%         'ExecutionEnvironment','gpu', ...
%         'LearnRateSchedule','piecewise', ...
%         'MiniBatchSize',32,...
%         'MaxEpochs',30,...
%         'LearnRateDropFactor',0.1, ...
%         'LearnRateDropPeriod',13, ...
%         'InitialLearnRate',0.001,...
%         'SequencePaddingValue', 0,...
%         'ValidationData',{validationData.Data, validationData.Label}, ...
%         'ValidationFrequency',30,...
%         'Plots','training-progress');
%     netTransfer_windows = trainNetworkFuzzy(trainData.Data, trainData.Label, lgraph, options, windows);
% 
%     % testing the network
%     [YPred scores] = classify(netTransfer_windows, testData.Data);
%     YPred_cell = cellstr(YPred);
%     YPred_num = zeros(1, length(YPred));
%     for jj = 1:length(YPred_cell)
%         YPred_num(jj) = str2num(YPred_cell{jj}(6:end));
%     end
%     predictions_squeeze = scores';
%     response_size = size(predictions_squeeze);
%     score_sequence = 0:1:100;
%     
%     [predicted_score_row predicted_score_col] = max(predictions_squeeze);
%     
%     predicted_score = zeros(1, length(predicted_score_col));
%     for ii = 1:length(predicted_score_col)
%         
%         cor_list = [predicted_score_col(ii)-windows.HalfLength : predicted_score_col(ii)+windows.HalfLength];
%         cor_list = cor_list(find(cor_list > windows.MinValue/windows.Step & cor_list < windows.MaxValue/windows.Step));
%         predicted_temp = predictions_squeeze((cor_list), ii);
%         predicted_score(ii) = score_sequence(cor_list) * predicted_temp ./ sum(predicted_temp);
%     end
%     predicted_score = predicted_score*windows.Step;
%     
%     RMSE_window = sqrt(mean((testData.Label - predicted_score).^2))/windows.Step;  % Root Mean Squared Error
%     MAE_window = mae(testData.Label - predicted_score);
%     [RMSE_window MAE_window]
%     
%     %% ---------------------------------------------------------------
%     options = trainingOptions('adam',...
%         'ExecutionEnvironment','gpu', ...
%         'LearnRateSchedule','piecewise', ...
%         'MiniBatchSize',32,...
%         'MaxEpochs',30,...
%         'LearnRateDropFactor',0.1, ...
%         'LearnRateDropPeriod',13, ...
%         'InitialLearnRate',0.001,...
%         'SequencePaddingValue', 0,...
%         'ValidationData',{validationData.Data, validationData.Label}, ...
%         'ValidationFrequency',30,...
%         'Plots','training-progress');
%     netTransfer = train(trainData.Data, trainData.Label, lgraph, options);
%     [YPred scores] = classify(netTransfer, testData.Data);
%     nbins = 0:1:100;
%     YPred_num_multi = scores*nbins';
%     RMSE = sqrt(mean((testData.Label - YPred_num_multi').^2)); % Root Mean Squared Error
%     MAE = mae(testData.Label - YPred_num_multi');
%     [RMSE MAE]
%     
% end
% 
% 




%%
% Comparing the DEX method and its mutation with windows
%%

clear; clc;
close all;

% -----------load data: Face with Age
CollectionDirectory = './FaceAgeData/wiki/';
fold_num = 1;
for fold_i = 1:fold_num

    load('C3_trainData.mat');
    load('C3_validationData.mat');
    load('C3_testData.mat');

    
    %% draw the disttribution of the face-age data after the segment-loss
    nbins = 0:1:100;
    figure;
    subplot(121);hist(trainData.Label, nbins);
    xlim([-1 101]); grid on;
    [counts, centers] = hist(trainData.Label, nbins);

    Label_list_cancel = [82,90,13,89,61,10,27,51,97,88,15,87,85,43,69,98,36,77,65,78]; %randperm(100,20)
    for ii = 1:length(trainData.Label)
        for jj = 1:length(Label_list_cancel)
            if trainData.Label(ii) == Label_list_cancel(jj)
                trainData.Label(ii) = -1;
%                 trainData.Data(:,:,:,ii) = [];
            end
        end
    end
    trainData.Data(:,:,:,find(trainData.Label==-1)) = [];
    trainData.Label(find(trainData.Label==-1)) = [];

    Label_list_cancel = find(counts < 20)
    for ii = 1:length(trainData.Label)
        for jj = 1:length(Label_list_cancel)
            if trainData.Label(ii) == Label_list_cancel(jj)
                trainData.Label(ii) = -1;
%                 trainData.Data(:,:,:,ii) = [];
            end
        end
    end
    trainData.Data(:,:,:,find(trainData.Label==-1)) = [];
    trainData.Label(find(trainData.Label==-1)) = [];
    
    subplot(122);hist(trainData.Label, nbins);
    xlim([-1 101]); grid on;
    
    
    %% Rearrange the age distribution
    windows.Enable = 1;
    windows.HalfLength = 10;
    windows.MaxValue = 100;
    windows.MinValue = 0;
    windows.Step = 5;
    windows.nbins = windows.MinValue :windows.Step: windows.MaxValue;
    numClasses = (windows.MaxValue - windows.MinValue) / windows.Step + 1;
    
    if windows.Step ~= 1
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
    end
    
    [counts, centers] = hist(trainData.Label, windows.nbins);
    windows.nbins(find(counts == 0)) = -1;
    numClasses = length(find(windows.nbins>=0));
    trainData.Label = categorical(trainData.Label);
    testData.Label = categorical(testData.Label);
    
    validationData.Data = testData.Data(:,:,:,1:630*2);
    validationData.Label = testData.Label(1:630*2);
    clear X
    clear Y
    %     clear trainData
    %     clear validationData
    
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
    
    % transfer from the last forth layer
%     trainData.Data = activations(net, trainData.Data, layersTransfor);
%     validationData.Data = activations(net, validationData.Data, layersTransfor);
%     testData.Data = activations(net, testData.Data, layersTransfor);
    
    %% Predict the score and Calculate the RMSE or MAE
    %% ---------------------------------------------------------------
    options = trainingOptions('adam',...
        'LearnRateSchedule','piecewise', ...
        'MiniBatchSize',32,...
        'MaxEpochs',30,...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',13, ...
        'InitialLearnRate',0.001,...
        'SequencePaddingValue', 0,...
        'ValidationFrequency',30,...
        'Plots','training-progress');
    netTransfer_windows = trainNetworkRehearsal(trainData.Data, trainData.Label, lgraph, options, windows);

    % testing the network
    [YPred scores] = classify(netTransfer_windows, testData.Data);
    YPred_cell = cellstr(YPred);
    YPred_num = zeros(1, length(YPred));
    for jj = 1:length(YPred_cell)
        YPred_num(jj) = str2num(YPred_cell{jj}(6:end));
    end
    predictions_squeeze = scores';
    response_size = size(predictions_squeeze);
    score_sequence = 0:1:100;
    
    [predicted_score_row predicted_score_col] = max(predictions_squeeze);
    
    predicted_score = zeros(1, length(predicted_score_col));
    for ii = 1:length(predicted_score_col)
        
        cor_list = [predicted_score_col(ii)-windows.HalfLength : predicted_score_col(ii)+windows.HalfLength];
        cor_list = cor_list(find(cor_list > windows.MinValue/windows.Step & cor_list < windows.MaxValue/windows.Step));
        predicted_temp = predictions_squeeze((cor_list), ii);
        predicted_score(ii) = score_sequence(cor_list) * predicted_temp ./ sum(predicted_temp);
    end
    predicted_score = predicted_score*windows.Step;
    
    RMSE_window = sqrt(mean((testData.Label - predicted_score).^2))/windows.Step;  % Root Mean Squared Error
    MAE_window = mae(testData.Label - predicted_score);
    [RMSE_window MAE_window]
    
    %% ---------------------------------------------------------------
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
    netTransfer = train(trainData.Data, trainData.Label, lgraph, options);
    [YPred scores] = classify(netTransfer, testData.Data);
    nbins = 0:1:100;
    YPred_num_multi = scores*nbins';
    RMSE = sqrt(mean((testData.Label - YPred_num_multi').^2)); % Root Mean Squared Error
    MAE = mae(testData.Label - YPred_num_multi');
    [RMSE MAE]
    
end


