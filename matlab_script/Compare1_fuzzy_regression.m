


%%
% Comparing the DEX method and its mutation with windows
%%

clear; clc;
close all;

% -----------load data: Face with Age
CollectionDirectory = './FaceAgeData/wiki/';
fold_num = 1;
for fold_i = 1:fold_num
    part_length = 1;
    part_list = randperm(part_length);  % [3 1 2 4];
    FaceData.Face = [];
    FaceData.Age = [];
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
    
    for part_i = 1:length(part_list)
        save_name = strcat('WIKI_FaceData_Part', num2str(part_list(part_i)));
        LoadData = load([CollectionDirectory save_name], 'FaceData');
            FaceData.Face = cat(4, FaceData.Face, LoadData.FaceData.Face);
            FaceData.Age = cat(2, FaceData.Age, LoadData.FaceData.Age);
    end
    testData.Data = FaceData.Face;
    testData.Label = FaceData.Age;
    clear FaceData
    
    %% Rearrange the age distribution
    windows.Enable = 1;
    windows.HalfLength = 10;
    windows.MaxValue = 100;
    windows.MinValue = 0;
    windows.Step = 1;
    windows.nbins = windows.MinValue :windows.Step: windows.MaxValue;
    numClasses = (windows.MaxValue - windows.MinValue) / windows.Step + 1;
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
%     clear trainData
%     clear validationData
    
    %% the loop for traning task -- 7 different NNs
    % protofile = 'age.prototxt';
    % datafile = 'dex_imdb_wiki.caffemodel';
    protofile = 'age_LAP.prototxt';
    datafile = 'dex_chalearn_iccv2015.caffemodel';
    net = importCaffeNetwork(protofile, datafile);
    
    %% Predict the score and Calculate the RMSE or MAE
    %% ---------------------------------------------------------------
%     length_data = 500;
%     testData.Data = testData.Data(:,:,:,1:length_data);
%     testData.Label = testData.Label(1:length_data);
    [YPred scores] = classify(net, testData.Data);

    %% ---------------------------------------------------------------
    window_list = 0:5:50;
    Result_list = zeros(4, length(window_list));
    flag_i = 0;
    for window_length = window_list
        windows.HalfLength = window_length;
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

        %% ---------------------------------------------------------------
        nbins = 0:1:100;
        YPred_num_multi = scores*nbins';
        RMSE = sqrt(mean((testData.Label - YPred_num_multi').^2)); % Root Mean Squared Error
        MAE = mae(testData.Label - YPred_num_multi');

        flag_i = flag_i + 1;
        Result_list(:, flag_i) = [RMSE_window MAE_window RMSE MAE];
    end
    
end


