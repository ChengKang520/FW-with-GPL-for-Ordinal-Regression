



clear; clc;
close all;

% -----------load SEP data
CollectionDirectory = './FaceAgeData/wiki/';

windows.Enable = 1;
windows.HalfLength = 5;
windows.MaxValue = 100;
windows.MinValue = 0;
windows.Step = 1;
windows.nbins = windows.MinValue :windows.Step: windows.MaxValue;
numClasses = (windows.MaxValue - windows.MinValue) / windows.Step;

for part_i = 1 %1:9
    
    
    save_name = strcat('WIKI_FaceData_Part', num2str(part_i));
    load([CollectionDirectory save_name], 'FaceData');
    
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
        
        %%  1. transfer train; only train the last layer (fully connected layer)
        % change the connection tabel
        if length(net.Layers) == 72  % ResNet18
            % change the structures of last 3 layers
            layersTransfor = 'pool5';
            layers = [
                imageInputLayer([1 1 512],'Name','data','Normalization','zerocenter')
                fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                softmaxLayer('Name','fc4_softmax')
                classificationLayer('Name','ClassificationLayer_fc4')
                ];
            
        else if length(net.Layers) == 177  % ResNet50
                % change the structures of last 3 layers
                layersTransfor = 'avg_pool';
                layers = [
                    imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
                    fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                    softmaxLayer('Name','fc4_softmax')
                    classificationLayer('Name','ClassificationLayer_fc4')
                    ];
                
            else if length(net.Layers) == 347  % ResNet101
                    % change the structures of last 3 layers
                    layersTransfor = 'pool5';
                    layers = [
                        imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
                        fullyConnectedLayer(numClasses,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc2_softmax')
                        classificationLayer('Name','ClassificationLayer_fc2')
                        ];
                    
                else if length(net.Layers) == 25  % AlexNet
                        % change the structures of last 3 layers
                        layersTransfor = 'drop7';
                        layers = [
                            imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
                            fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                            softmaxLayer('Name','fc4_softmax')
                            classificationLayer('Name','ClassificationLayer_fc4')
                            ];
                    else if length(net.Layers) == 41  % VGG16
                            % change the structures of last 3 layers
                            layersTransfor = 'drop7';
                            layers = [
                                imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
                                fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                                softmaxLayer('Name','fc4_softmax')
                                classificationLayer('Name','ClassificationLayer_fc4')
                                ];
                        else if length(net.Layers) == 47  % VGG19
                                % change the structures of last 3 layers
                                layersTransfor = 'drop7';
                                layers = [
                                    imageInputLayer([1 1 4096],'Name','data','Normalization','zerocenter')
                                    fullyConnectedLayer(numClasses,'Name','fc4','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                                    softmaxLayer('Name','fc4_softmax')
                                    classificationLayer('Name','ClassificationLayer_fc4')
                                    ];
                                
                            else if length(net.Layers) == 144  % GoogleNet
                                    % change the structures of last 3 layers
                                    layersTransfor = 'pool5-drop_7x7_s1';
                                    layers = [
                                        imageInputLayer([1 1 1024],'Name','data','Normalization','zerocenter')
                                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                                        softmaxLayer('Name','fc3_softmax')
                                        classificationLayer('Name','ClassificationLayer_fc3')
                                        ];
                                else if length(net.Layers) == 316  % InceptionV3
                                        % change the structures of last 3 layers
                                        layersTransfor = 'avg_pool';
                                        layers = [
                                            imageInputLayer([1 1 2048],'Name','data','Normalization','zerocenter')
                                            fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                                            softmaxLayer('Name','fc3_softmax')
                                            classificationLayer('Name','ClassificationLayer_fc3')
                                            ];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        lgraph = layerGraph(layers);
        analyzeNetwork(lgraph); % show the structure of designed neural network
        
        %% Rearrange the age distribution 
        nbins = 0:1:100;
        figure;
        subplot(121);hist(FaceData.Age, nbins);
        xlim([-1 101]); grid on;

        X = FaceData.Age;
        for j1 = 1:length(X)
            for i1 = 1:(length(windows.nbins)-1)
                if X(j1) >= windows.nbins(i1) & X(j1) < windows.nbins(i1+1) | X(j1) == windows.nbins(i1+1)
                    FaceData.Age(j1) = i1;
                end
            end
        end
        subplot(122);hist(FaceData.Age, windows.nbins);
        xlim([-1 101]); grid on;
        
        trainData.Data = FaceData.Face(:,:,:,1:5040);
        trainData.Label = FaceData.Age(1:5040);
        validationData.Data = trainData.Data(:,:,:,500);
        validationData.Label = trainData.Label(1:500);
        testData.Data = FaceData.Face(:,:,:,5041:end);
        testData.Label = FaceData.Age(5041:end);
        
        % extracting some specific layers, and ploting
        trainFeatures = activations(net, trainData.Data, layersTransfor);
        validationFeatures = activations(net, validationData.Data, layersTransfor);
        testFeatures = activations(net, testData.Data, layersTransfor);

        options = trainingOptions('adam',...
            'LearnRateSchedule','piecewise', ...
            'MiniBatchSize',32,...
            'MaxEpochs',30,...
            'LearnRateDropFactor',0.1, ...
            'LearnRateDropPeriod',13, ...
            'InitialLearnRate',0.001,...
            'SequencePaddingValue', 0,...
            'ValidationData',{validationFeatures, validationData.Label}, ...
            'ValidationFrequency',30,...
            'Plots','training-progress');
        
        netTransfer = trainNetworkRehearsal(trainFeatures, trainData.Label, lgraph, options, windows);
        
        

        
        [counts, centers] = hist(X, windows.nbins);
        windows.nbins(find(counts == 0)) = -1;
        train.Labels = categorical(train.Labels);
        num_class = length(find(windows.nbins>=0));
        save X X
        
        
        
        load('../Data_Bank/mnist-data-mat/mnist_train.mat');
        load('../Data_Bank/mnist-data-mat/mnist_train_labels.mat');
        load('../Data_Bank/mnist-data-mat/mnist_test.mat');
        load('../Data_Bank/mnist-data-mat/mnist_test_labels.mat');
        
        
        epochs_Seqeuence = 10  % [5 10 15 20]
        inital_filter_num = 12
        fold_10 = 5;
        
        data_length = 1000;
        Pred_result = zeros(1, 10);
        
        TrainData_select = randi([1,60000],1,data_length);
        TestData_select = randi([1,10000],1,data_length);
        TrainData_length = length(TrainData_select);
        TestData_length = length(TestData_select);
        
        mnist_train1 = mnist_train(TrainData_select,:);
        mnist_train_labels1 = mnist_train_labels(TrainData_select)*10;
        mnist_test1 = mnist_test(TestData_select,:);
        mnist_test_labels1 = mnist_test_labels(TestData_select)*10;
        
        trainFeature = zeros(1, TrainData_length, 28, 28);
        trainFeature(:,:,:,:) = reshape(mnist_train1(1:TrainData_length,:), TrainData_length, 28, 28);
        trainFeature = permute(trainFeature,[3, 4, 1, 2]);
        trainLabel = mnist_train_labels1(1:TrainData_length);
        
        validationFeature = trainFeature(:,:,:,1:TrainData_length/10);
        
        testFeature = zeros(1, TestData_length, 28, 28);
        testFeature(:,:,:,:) = reshape(mnist_test1(1:TestData_length,:), TestData_length, 28, 28);
        testFeature = permute(testFeature,[3, 4, 1, 2]);
        
        %% ---------------------------------------------------------------
        windows.Enable = 1;
        windows.MaxValue = 90;
        windows.MinValue = 0;
        windows.Step = 10;
        windows.HalfLength = 2;
        windows.nbins = windows.MinValue :windows.Step: windows.MaxValue;
        num_class = length((windows.MinValue :windows.Step: windows.MaxValue)/windows.Step);
        
        
        trainLabel = categorical(round(mnist_train_labels1(1:TrainData_length) / windows.Step));
        validationLabel = trainLabel(1:TrainData_length/10);
        testLabel = categorical(round(mnist_test_labels1(1:TestData_length) / windows.Step));
        
        extend_FCL = 1;
        lgraph_initial = construct_CNN(extend_FCL, inital_filter_num, num_class);
        %     analyzeNetwork(lgraph_initial);
        
        options = trainingOptions('adam', ...
            'InitialLearnRate',0.01, ...
            'MaxEpochs',epochs_Seqeuence, ...
            'MiniBatchSize',16,...
            'ValidationData',{validationFeature, validationLabel}, ...
            'Shuffle','every-epoch', ...
            'ValidationFrequency',10, ...
            'Plots','training-progress',...
            'Verbose',false); %
        
        [net_initial Rehrearsal info] = trainNetworkRehearsal(trainFeature, trainLabel', lgraph_initial, options, windows);
        
        
        
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
        
        
    end
end






