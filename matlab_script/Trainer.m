classdef Trainer < handle
    % Trainer   Trains a network
    
    %   Copyright 2015-2019 The MathWorks, Inc.
    
    properties(Access = protected)
        Options
        Schedule
        Precision
        Reporter
        SummaryFcn
        Summary
        ExecutionStrategy
        StopTrainingFlag
        StopReason
        InterruptException
    end
    
    methods
        function this = Trainer( ...
                opts, precision, reporter, executionSettings, summaryFcn)
            % Trainer    Constructor for a network trainer
            %
            % opts - training options (nnet.cnn.TrainingOptionsSGDM)
            % precision - data precision
            this.Options = opts;
            this.Schedule = nnet.internal.cnn.LearnRateScheduleFactory.create(opts.LearnRateScheduleSettings);
            this.Precision = precision;
            this.Reporter = reporter;
            this.SummaryFcn = summaryFcn;
            % Declare execution strategy
            if ismember( executionSettings.executionEnvironment, {'gpu'} )
                this.ExecutionStrategy = nnet.internal.cnn.TrainerGPUStrategy;
            else
                this.ExecutionStrategy = nnet.internal.cnn.TrainerHostStrategy;
            end
            
            % Print execution environment if in verbose mode
            iPrintExecutionEnvironment(opts, executionSettings);
            
            % Register a listener to detect requests to terminate training
            addlistener( reporter, ...
                'TrainingInterruptEvent', @this.stopTrainingCallback);
        end
        
        function net = initializeNetwork(this, net, data)
            % Perform any initialization steps required by the input and
            % output layers
            
            % Store any classification labels or response names in the
            % appropriate output layers.
            net = net.storeResponseData(data.ResponseMetaData);
            
            if this.Options.ResetInputNormalization
                net = net.resetNetworkInitialization();
            end
            
            % Setup reporters
            willInputNormalizationBeComputed = net.hasEmptyInputStatistics();
            reporterSetupInfo = nnet.internal.cnn.util.ReporterSetupInfo(willInputNormalizationBeComputed);
            this.Reporter.setup(reporterSetupInfo);
            
            if willInputNormalizationBeComputed
                % Always use 'truncateLast' as we want to process all the data we have.
                savedEndOfEpoch = data.EndOfEpoch;
                data.EndOfEpoch = 'truncateLast';
                
                % Compute statistics on host or in distributed environment
                stats = iCreateStatisticsAccumulator(net.InputSizes,this.Precision);
                stats = this.computeInputStatisticsInEnvironment(data,stats,net);
                
                % Initialize input layers with statistics
                net = net.initializeNetwork(stats);
                
                data.EndOfEpoch = savedEndOfEpoch;
            end
        end
        
        function net = train(this, net, data)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            reporter = this.Reporter;
            schedule = this.Schedule;
            prms = collectSettings(this, net);
            data.start();
            this.Summary = this.SummaryFcn(data, prms.maxEpochs);
            
            regularizer = iCreateRegularizer('l2',net.LearnableParameters,this.Precision,this.Options);
            
            solver = iCreateSolver(net.LearnableParameters,this.Precision,this.Options);
            
            trainingTimer = tic;
            
            reporter.start();
            iteration = 0;
            this.StopTrainingFlag = false;
            this.StopReason = nnet.internal.cnn.util.TrainingStopReason.FinalIteration;
            gradientThresholdOptions = struct('Method', this.Options.GradientThresholdMethod,...
                'Threshold', this.Options.GradientThreshold);
            gradThresholder = nnet.internal.cnn.GradientThresholder(gradientThresholdOptions);
            learnRate = initializeLearning(this);
            
            for epoch = 1:prms.maxEpochs
                this.shuffle( data, prms.shuffleOption, epoch );
                data.start();
                while ~data.IsDone && ~this.StopTrainingFlag
                    [X, response] = data.next();
                    % Cast data to appropriate execution environment for
                    % training and apply transforms
                    X = this.ExecutionStrategy.environment(X);
                    response = this.ExecutionStrategy.environment(response);
                    
                    propagateState = iNeedsToPropagateState(data);
                    [gradients, predictions, states] = this.computeGradients(net, X, response, propagateState);
                    
                    % Reuse the layers outputs to compute loss
                    miniBatchLoss = net.loss( predictions, response );
                    
                    gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                    
                    gradients = thresholdGradients(gradThresholder,gradients);
                    
                    velocity = solver.calculateUpdate(gradients,learnRate);
                    
                    net = net.updateLearnableParameters(velocity);
                    
                    net = net.updateNetworkState(states);
                    
                    elapsedTime = toc(trainingTimer);
                    
                    iteration = iteration + 1;
                    this.Summary.update(predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate, data.IsDone);
                    % It is important that computeIteration is called
                    % before reportIteration, so that the summary is
                    % correctly updated before being reported
                    reporter.computeIteration( this.Summary, net );
                    reporter.reportIteration( this.Summary );
                end
                learnRate = schedule.update(learnRate, epoch);
                
                reporter.reportEpoch( epoch, iteration, net );
                
                % If an interrupt request has been made, break out of the
                % epoch loop
                if this.StopTrainingFlag
                    break;
                end
            end
            reporter.computeFinish( this.Summary, net );
            reporter.finish( this.Summary, this.StopReason );
        end
        
        function [net FuzzyValue] = trainFuzzy(this, net, data, windows)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            FuzzyValue = [];
            reporter = this.Reporter;
            schedule = this.Schedule;
            prms = collectSettings(this, net);
            data.start();
            this.Summary = this.SummaryFcn(data, prms.maxEpochs);
            
            regularizer = iCreateRegularizer('l2',net.LearnableParameters,this.Precision,this.Options);
            
            solver = iCreateSolver(net.LearnableParameters,this.Precision,this.Options);
            
            trainingTimer = tic;
            
            reporter.start();
            iteration = 0;
            this.StopTrainingFlag = false;
            this.StopReason = nnet.internal.cnn.util.TrainingStopReason.FinalIteration;
            gradientThresholdOptions = struct('Method', this.Options.GradientThresholdMethod,...
                'Threshold', this.Options.GradientThreshold);
            gradThresholder = nnet.internal.cnn.GradientThresholder(gradientThresholdOptions);
            learnRate = initializeLearning(this);
            
            for epoch = 1:prms.maxEpochs
                this.shuffle( data, prms.shuffleOption, epoch );
                data.start();
                while ~data.IsDone && ~this.StopTrainingFlag
                    [X, response] = data.next();
                    % Cast data to appropriate execution environment for
                    % training and apply transforms
                    X = this.ExecutionStrategy.environment(X);
                    response = this.ExecutionStrategy.environment(response);
                    propagateState = iNeedsToPropagateState(data);
                    
%                     if windows.Enable == 1
%                         % ---------------------------------------------------
%                         predictions_squeeze = squeeze(predictions);
%                         response_squeeze = squeeze(response);
%                         score_sequence = [windows.MinValue:windows.Step:windows.MaxValue]/windows.Step;
%                         
%                         [real_score_row real_score_col] = find(response_squeeze == 1);
%                         [predicted_score_row predicted_score_col] = max(predictions_squeeze);
% 
%                         predicted_score = zeros(1, length(predicted_score_col));
%                         for ii = 1:length(predicted_score_col)         
%                             cor_list = [predicted_score_col(ii)-windows.HalfLength : predicted_score_col(ii)+windows.HalfLength];
%                             cor_list = cor_list(find(cor_list >= windows.MinValue/windows.Step & cor_list <= windows.MaxValue/windows.Step));
%                             predicted_temp = predictions_squeeze((cor_list+1), ii);
%                             predicted_score(ii) = score_sequence(cor_list+1) * predicted_temp ./ sum(predicted_temp);
%                         end
%                         predicted_score = (predicted_score) * windows.Step;
%                         loss_FuzzyMSE = mse(score_sequence(real_score_row) * windows.Step - predicted_score);
%                         miniBatchLoss = loss_FuzzyMSE;
%                         % ---------------------------------------------------
%                     elseif windows.Enable == 0
%                         % Reuse the layers outputs to compute loss
%                         miniBatchLoss = net.loss( predictions, response );
%                     end
                    if windows.Enable == 1
                        [gradients, predictions, states] = this.computeGradientsFuzzy(net, X, response, propagateState, windows);
                        % ---------------------------------------------------
                        predictions_squeeze = gather(squeeze(predictions));
                        response_squeeze = gather(squeeze(response));
                        score_sequence_real = [windows.nbins]/windows.Step;
                        score_sequence_pre = score_sequence_real(find(windows.nbins >= 0));

                        [real_score_row real_score_col] = find(gather(response_squeeze) == 1);
                        [predicted_score_row predicted_score_col] = max(gather(predictions_squeeze));

                        predicted_score = zeros(1, length(predicted_score_col));
                        for ii = 1:length(predicted_score_col)
                            cor_list_pre = [];
                            cor_list_real = [];
                            sequence_list = [score_sequence_pre(predicted_score_col(ii))-windows.HalfLength : score_sequence_pre(predicted_score_col(ii))+windows.HalfLength];
                            sequence_list = sequence_list(find(sequence_list >= windows.MinValue/windows.Step & sequence_list <= windows.MaxValue/windows.Step));
                            
                            sequence_list = sequence_list(find(score_sequence_real(sequence_list+1)>=0));
                            
                            for jj = 1:length(sequence_list)
                                cor_list_real(jj) = find(score_sequence_real==sequence_list(jj));
                                cor_list_pre(jj) = find(score_sequence_pre==sequence_list(jj));
                            end
                            
                            predicted_temp = predictions_squeeze(cor_list_pre, ii);
                            predicted_score(ii) = score_sequence_real(cor_list_real) * predicted_temp ./ sum(predicted_temp);
                        end
                        predicted_score = (predicted_score) * windows.Step;
                        loss_FuzzyMSE = mse(score_sequence_pre(real_score_row) * windows.Step - predicted_score);
                        miniBatchLoss = loss_FuzzyMSE;
                        % ---------------------------------------------------
                    elseif windows.Enable == 0
                        [gradients, predictions, states] = this.computeGradients(net, X, response, propagateState);
                        % Reuse the layers outputs to compute loss
                        miniBatchLoss = net.loss( predictions, response );
                    end

                    gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);

                    gradients = thresholdGradients(gradThresholder,gradients);
                    
                    velocity = solver.calculateUpdate(gradients,learnRate);
                    
                    net = net.updateLearnableParameters(velocity);
                    
                    net = net.updateNetworkState(states);
                    
                    elapsedTime = toc(trainingTimer);
                    
                    iteration = iteration + 1;
                    this.Summary.update(predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate, data.IsDone);
                    % It is important that computeIteration is called
                    % before reportIteration, so that the summary is
                    % correctly updated before being reported
                    reporter.computeIteration( this.Summary, net );
                    reporter.reportIteration( this.Summary );

                    learnRate = schedule.update(learnRate, epoch);
                    
                    reporter.reportEpoch( epoch, iteration, net );
                    
                    % If an interrupt request has been made, break out of the
                    % epoch loop
                    if this.StopTrainingFlag
                        break;
                    end
                end
                reporter.computeFinish( this.Summary, net );
                reporter.finish( this.Summary, this.StopReason );
            end
        end
        
        function net = finalizeNetwork(this, net, data)
            % Perform any finalization steps required by the layers
            
            % Always use 'truncateLast' as we want to process all the data we have.
            savedEndOfEpoch = data.EndOfEpoch;
            data.EndOfEpoch = 'truncateLast';
            
            % Call shared implementation
            net = this.doFinalize(net, data);
            
            data.EndOfEpoch = savedEndOfEpoch;
        end
    end
    
    methods(Access = protected)
        function stopTrainingCallback(this, ~, evtData)
            % stopTraining  Callback triggered by interrupt events that
            % want to request training to stop
            this.StopTrainingFlag = true;
            this.StopReason = evtData.TrainingStopReason;
        end
        
        function settings = collectSettings(this, net)
            % collectSettings  Collect together fixed settings from the
            % Trainer and the data and put in the correct form.
            settings.maxEpochs = this.Options.MaxEpochs;
            settings.lossFunctionType = iGetLossFunctionType(net);
            settings.shuffleOption = this.Options.Shuffle;
            settings.numOutputs = numel(net.OutputSizes);
            settings.InputObservationDim = cellfun(@(sz)numel(sz)+1, net.InputSizes, 'UniformOutput', false);
            settings.OutputObservationDim = cellfun(@(sz)numel(sz)+1, net.OutputSizes, 'UniformOutput', false);
        end
        
        function learnRate = initializeLearning(this)
            % initializeLearning  Set initial learning rate.
            learnRate = this.Precision.cast( this.Options.InitialLearnRate );
        end
        
        function [gradients, predictions, states] = computeGradients(~, net, X, Y, propagateState)
            % computeGradients   Compute the gradients of the network. This
            % function returns also the network output so that we will not
            % need to perform the forward propagation step again.
            [gradients, predictions, states] = net.computeGradientsForTraining(X, Y, propagateState);
        end
        
        function [gradients, predictions, states] = computeGradientsFuzzy(~, net, X, Y, propagateState, windows)
            % computeGradients   Compute the gradients of the network. This
            % function returns also the network output so that we will not
            % need to perform the forward propagation step again.
            [gradients, predictions, states] = net.computeGradientsForTrainingFuzzy(X, Y, propagateState, windows);
        end
        
        function stats = computeInputStatisticsInEnvironment(this, data, stats, net )
            % Initialization steps running on the host
            stats = doComputeStatistics(this, data, stats, net);
        end
        
        function stats = doComputeStatistics(this, data, stats, net)
            % Compute statistics from the training data set
            % Do one epoch
            
            % If we have a sequence dispatcher which applies padding we
            % replace padding values with NaN. Then padding can be ignored
            % during computation of statistics
            replacePaddingWithNaN = iNeedToReplacePaddingWithNaNs(data);
            
            data.start();
            while ~data.IsDone
                if replacePaddingWithNaN
                    [X, ~, idxInfo] = data.next();
                    % Replace padded values with NaN, since NaNs will be
                    % ignored when computing statistics
                    dataIdx = idxInfo.UnpaddingIdx;
                    obsDim = numel(data.DataSize) + 1;
                    [batchSize, maxSeqLen] = size(X, obsDim:obsDim+1);
                    for n = 1:batchSize
                        paddingIdx = setdiff( 1:maxSeqLen, dataIdx{n} );
                        xInds = [ repelem({':'}, obsDim-1), {n}, {paddingIdx} ];
                        X( xInds{:} ) = NaN;
                    end
                else
                    X = data.next();
                end
                if ~isempty(X) % In the parallel case X can be empty
                    % Cast data to appropriate execution environment
                    X = this.ExecutionStrategy.environment(X);
                    X = iWrapInCell(X);
                    
                    % Accumulate statistics for each input
                    for i = 1:numel(stats)
                        % The ImageInputLayer might have data augmentation
                        % transforms that have to be applied before
                        % computing statistics (e.g. cropping).
                        if isa(net.InputLayers{i},'nnet.internal.cnn.layer.ImageInput')
                            X{i} = apply(net.InputLayers{i}.TrainTransforms,X{i});
                        end
                        
                        stats{i} = accumulate(stats{i}, X{i});
                    end
                end
            end
        end
        
        function net = doFinalize(this, net, data)
            % Perform any finalization steps required by the layers
            isFinalizable = @(l)isa(l,'nnet.internal.cnn.layer.Finalizable') && ...
                l.NeedsFinalize;
            needsFinalize = cellfun(isFinalizable, net.Layers);
            if any(needsFinalize)
                % Do one final epoch
                data.start();
                while ~data.IsDone
                    X = data.next();
                    if ~isempty(X) % In the parallel case X can be empty
                        % Cast data to appropriate execution environment for
                        % training and apply transforms
                        X = this.ExecutionStrategy.environment(X);
                        
                        % Ask the network to finalize
                        net = finalizeNetwork(net, X);
                    end
                end
            end
            this.Reporter.computeFinalIteration(this.Summary, net);
            this.Reporter.reportFinalIteration(this.Summary);
        end
    end
    
    methods(Access = protected)
        function shuffle(~, data, shuffleOption, epoch)
            % shuffle   Shuffle the data as per training options
            if ~isequal(shuffleOption, 'never') && ...
                    ( epoch == 1 || isequal(shuffleOption, 'every-epoch') )
                data.shuffle();
            end
        end
    end
end

function regularizer = iCreateRegularizer(name,learnableParameters,precision,regularizationOptions)
regularizer = nnet.internal.cnn.regularizer.RegularizerFactory.create(name,learnableParameters,precision,regularizationOptions);
end

function solver = iCreateSolver(learnableParameters,precision,trainingOptions)
solver = nnet.internal.cnn.solver.SolverFactory.create(learnableParameters,precision,trainingOptions);
end

function stats = iCreateStatisticsAccumulator(inputSizes,outputType)
stats = nnet.internal.cnn.statistics.AccumulatorFactory.create(inputSizes,outputType);
end

function tf = iNeedsToPropagateState(data)
tf = isa(data,'nnet.internal.cnn.SequenceDispatcher') && data.IsNextMiniBatchSameObs;
end

function tf = iNeedToReplacePaddingWithNaNs(data)
tf = isa(data,'nnet.internal.cnn.BuiltInSequenceDispatcher') ...
    && ~isequal(data.SequenceLength,'shortest') ...
    && ~isnan(data.PaddingValue);
end

function t = iGetLossFunctionType(net)
if isempty(net.Layers)
    t = 'nnet.internal.cnn.layer.NullLayer';
else
    t = class(net.Layers{end});
end
end

function iPrintMessage(messageID, varargin)
string = getString(message(messageID, varargin{:}));
fprintf( '%s\n', string );
end

function iPrintExecutionEnvironment(opts, executionSettings)
% Print execution environment if in 'auto' mode
if opts.Verbose
    if ismember(opts.ExecutionEnvironment, {'auto'})
        if ismember(executionSettings.executionEnvironment, {'cpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInSerialOnCPU');
        elseif ismember(executionSettings.executionEnvironment, {'gpu'})
            if executionSettings.useStateless
                iPrintMessage( ...
                    'nnet_cnn:internal:cnn:Trainer:TrainingStatelessInSerialOnGPU');
            else
                iPrintMessage( ...
                    'nnet_cnn:internal:cnn:Trainer:TrainingInSerialOnGPU');
            end
        end
    elseif ismember(opts.ExecutionEnvironment, {'parallel'})
        if ismember(executionSettings.executionEnvironment, {'cpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInParallelOnCPUs');
        elseif ismember(executionSettings.executionEnvironment, {'gpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInParallelOnGPUs');
        end
    end
end
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end