

%%
% Comparing the DEX method and its mutation with windows
%%

clear; clc;
close all;
% gpuDevice(2)
% -----------load data: Face with Age
CollectionDirectory = './FaceAgeData/wiki/';
window_num = [5 10 20 50];
for window_i = window_num
    window_i
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
    
    %% Rearrange the age distribution
    windows.Enable = 1;
    windows.HalfLength = window_i;
    windows.MaxValue = 100;
    windows.MinValue = 0;
    windows.Step = 1;
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
%     testData.Label = categorical(testData.Label);
    
    validationData.Data = testData.Data(:,:,:,1:630*2);
    validationData.Label = testData.Label(1:630*2);
    clear X
    clear Y
    %     clear trainData
    %     clear validationData
    
%     nbins = 1:1:101;
%     figure;
%     subplot(121);hist(trainData.Label, nbins);
%     xlim([-1 101]); grid on;
    
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

        trainData.Data = activations(net, trainData.Data, layersTransfor);
        validationData.Data = activations(net, validationData.Data, layersTransfor);
        testData.Data = activations(net, testData.Data, layersTransfor);

    
    %% Predict the score and Calculate the RMSE or MAE
    %% ---------------------------------------------------------------
    options = trainingOptions('adam',...
        'ExecutionEnvironment','gpu', ...
        'LearnRateSchedule','piecewise', ...
        'MiniBatchSize',32,...
        'MaxEpochs',25,...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',13, ...
        'InitialLearnRate',0.001,...
        'SequencePaddingValue', 0,...
        'ValidationFrequency',30,...
        'Plots','training-progress');
    netTransfer_windows = trainNetworkFuzzy(trainData.Data, trainData.Label, lgraph, options, windows);
    save netTransfer_windows netTransfer_windows
    % testing the network
    [YPred scores] = classify(netTransfer_windows, testData.Data);
    YPred_cell = cellstr(YPred);
    YPred_num = zeros(1, length(YPred));
    for jj = 1:length(YPred_cell)
        YPred_num(jj) = str2num(YPred_cell{jj});
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
%     Result_window = [RMSE_window MAE_window]
%     save Result_window5 Result_window
    
    %% ---------------------------------------------------------------
    options = trainingOptions('adam',...
        'ExecutionEnvironment','gpu', ...
        'LearnRateSchedule','piecewise', ...
        'MiniBatchSize',32,...
        'MaxEpochs',25,...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',13, ...
        'InitialLearnRate',0.001,...
        'SequencePaddingValue', 0,...
        'ValidationFrequency',30,...
        'Plots','training-progress');
    netTransfer = trainNetwork(trainData.Data, trainData.Label, lgraph, options);
    [YPred scores] = classify(netTransfer, testData.Data);
    nbins = 0:1:100;
    YPred_num_multi = scores*nbins';
    RMSE = sqrt(mean((testData.Label - YPred_num_multi').^2)); % Root Mean Squared Error
    MAE = mae(testData.Label - YPred_num_multi');
%     [RMSE MAE]
    
    Result = [RMSE_window MAE_window; RMSE MAE]
    
    save_name = strcat('C2_Result', num2str(window_i), 'second');
    save([save_name], 'Result');
end



