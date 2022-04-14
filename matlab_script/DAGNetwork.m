classdef DAGNetwork < nnet.internal.cnn.TrainableNetwork
    % DAGNetwork   Class for a directed acyclic graph network
    
    %   Copyright 2017-2020 The MathWorks, Inc.
    
    properties
        % Layers of the optimized LayerGraph
        Layers
        
        % Connections of the optimized LayerGraph
        Connections
    end
    
    properties(SetAccess = private)
        % NumInputLayers   The number of input layers for this network
        %   The number of input layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumInputLayers
        
        % NumOutputLayers   The number of output layers for this network
        %   The number of output layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumOutputLayers
        
        % InputLayerIndices   The indices of the input layers
        InputLayerIndices
        
        % OutputLayerIndices   The indices of the output layers
        OutputLayerIndices
        
        % FinalizableLayerIndices   The indices of the finalizable layers
        FinalizableLayerIndices
        
        % InputSizes   Sizes of the network inputs
        InputSizes
        
        % OutputSizes   Sizes of the network outputs
        OutputSizes
        
        % TopologicalOrder  Topological order of layers in the
        % OriginalLayers array
        TopologicalOrder
        
        % SortedConnections  Connections matrix for the externally
        % visible layer graph (sorted but unoptimized). This is stored
        % rather than recomputed on access for performance reasons.
        SortedConnections
        
        % StatefulIdx   Indices of layers which require an update of state
        % during forward propagation
        StatefulIdx
        
        % HasSequenceOutput   (1xNumOutputLayers logical) True for layers
        % with sequence output
        HasSequenceOutput
    end
    
    properties(Access = private)
        % LayerGraphExecutionInfo  Stores relationships between layers and
        % inputs and outputs stored in an activations buffer during
        % propagation
        LayerGraphExecutionInfo
        
        % Sizes   The output sizes for each activation
        Sizes
        
        % LayerOutputSizes  The output sizes for each layer
        LayerOutputSizes
        
        % UseGpu  Records if execution is taking place on the GPU, used to
        % ensure data is in the right place before propagation
        UseGpu = false
        
        % NetworkOptimizer  Object defining how layers have been optimized
        NetworkOptimizer = nnet.internal.cnn.optimizer.NoOpNetworkOptimizer()
    end
    
    properties(Dependent, Access = private)
        % NumLayers
        NumLayers
        
        % NumActivations   Number of activations
        %   The number of unique output activations in the network.
        NumActivations
        
        % EdgeTable  Used for efficient graph execution
        EdgeTable
        
        % ListOfBufferInputIndices   List of buffer input indices
        ListOfBufferInputIndices
        
        % ListOfBufferOutputIndices   List of buffer output indices
        ListOfBufferOutputIndices
        
        % ListOfBufferIndicesForClearingForward   List of buffer entries
        % that can be cleared as we move forward through the network
        ListOfBufferIndicesForClearingForward
        
        % ListOfBufferIndicesForClearingBackward   List of buffer entries
        % that can be cleared as we move backward through the network
        ListOfBufferIndicesForClearingBackward
        
        % ListOfBufferIndicesForClearingForwardForBackward   List of 
        % buffer entries not used for the backward pass that can be cleared
        % as we move forward through the network
        ListOfBufferIndicesForClearingForwardForBackward
    end
    
    properties(Dependent, SetAccess = private)
        % LearnableParameters    Learnable parameters of the networks
        %                        (vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters
        
        % LayerGraph    The optimized layer graph
        %   This contains an internal layer graph with the most recent
        %   learnable parameters and is created using the Layers and
        %   Connections properties.
        LayerGraph
        
        % OriginalLayers  Layers in the original order, unoptimized
        OriginalLayers
        
        % OriginalConnections  Connections in the original order,
        % unoptimized
        OriginalConnections
        
        % SortedLayers   Unoptimized Layers in a topologically sorted order
        SortedLayers

        % NumInputs   The number of input layers in the network
        NumInputs
        
        % NumOutputs   The number of output layers in the network
        NumOutputs
        
        % StatefulLayers   Layers which require an update of state
        StatefulLayers
        
        % IsRNN   True if the network contains a sequence input layer
        IsRNN
    end
    
    properties(Dependent)
        % InputLayers   The input layers for this network
        InputLayers
        
        % OutputLayers   The output layers for this network
        OutputLayers
    end
    
    methods
        function learnableParameters = get.LearnableParameters(this)
            learnableParameters = cell(this.NumLayers, 1);
            for el = 1:this.NumLayers
                P = this.Layers{el}.LearnableParameters;
                if ~isempty(P)
                    % Don't accumulate empty parameters because the type might not match that
                    % of other layers.
                    learnableParameters{el} = P;
                end
            end
            
            learnableParameters = [learnableParameters{:}];
        end
        
        function inputLayers = get.InputLayers(this)
            inputLayers = this.Layers(this.InputLayerIndices);
        end
        
        function outputLayers = get.OutputLayers(this)
            outputLayers = this.Layers(this.OutputLayerIndices);
        end
        
        function this = set.OutputLayers(this, val)
            this.Layers(this.OutputLayerIndices) = val;
        end

        function layerGraph = get.LayerGraph(this)
            layerGraph = makeTrainedLayerGraph(this);
        end
        
        function originalLayers = get.OriginalLayers(this)
            originalLayers = nnet.internal.cnn.LayerGraph.sortedToOriginalLayers(this.SortedLayers, this.TopologicalOrder);
        end
        
        function originalConnections = get.OriginalConnections(this)
            originalConnections = nnet.internal.cnn.LayerGraph.sortedToOriginalConnections(this.SortedConnections, this.TopologicalOrder);
            originalConnections = sortrows(originalConnections);
        end
        
        function sortedLayers = get.SortedLayers(this)
            numOriginalLayers = numel(this.TopologicalOrder);
            sortedLayers = cell(numOriginalLayers, 1);
            for l = 1:numel(this.Layers)
                thisLayer = this.Layers{l};
                originalLayerIndices = this.NetworkOptimizer.mapToOriginal(l);
                if isa(thisLayer, 'nnet.internal.cnn.layer.FusedLayer')
                    sortedLayers(originalLayerIndices) = thisLayer.OriginalLayers(:);
                else
                    sortedLayers{originalLayerIndices} = thisLayer;
                end
            end
        end

        function val = get.NumInputs(this)
            val = numel(this.InputLayers);
        end
        
        function val = get.NumOutputs(this)
            val = numel(this.OutputLayers);
        end
        
        function numActivations = get.NumActivations(this)
            numActivations = this.LayerGraphExecutionInfo.NumActivations;
        end
        
        function edgeTable = get.EdgeTable(this)
            edgeTable = this.LayerGraphExecutionInfo.EdgeTable;
        end
        
        function listOfBufferIndicesForClearingForwardForBackward =...
                get.ListOfBufferIndicesForClearingForwardForBackward(this)
            
            listOfBufferIndicesForClearingForwardForBackward =...
                this.LayerGraphExecutionInfo.ListOfBufferIndicesForClearingForwardForBackward;
        end
        
        function listOfBufferInputIndices = get.ListOfBufferInputIndices(this)
            listOfBufferInputIndices = this.LayerGraphExecutionInfo.ListOfBufferInputIndices;
        end
        
        function listOfBufferOutputIndices = get.ListOfBufferOutputIndices(this)
            listOfBufferOutputIndices = this.LayerGraphExecutionInfo.ListOfBufferOutputIndices;
        end
        
        function listOfBufferIndicesForClearingForward = get.ListOfBufferIndicesForClearingForward(this)
            listOfBufferIndicesForClearingForward = this.LayerGraphExecutionInfo.ListOfBufferIndicesForClearingForward;
        end
        
        function listOfBufferIndicesForClearingBackward = get.ListOfBufferIndicesForClearingBackward(this)
            listOfBufferIndicesForClearingBackward = this.LayerGraphExecutionInfo.ListOfBufferIndicesForClearingBackward;
        end

        function val = get.NumLayers(this)
            val = numel(this.Layers);
        end
        
        function layers = get.StatefulLayers(this)
            layers = this.Layers(this.StatefulIdx);
        end
        
        function tf = get.IsRNN(this)
            tf = nnet.internal.cnn.util.isRNN( this.InputLayers );
        end
        
        function [index, offset] = getInternalForExternalLayerIndex(this, index)
            index = this.TopologicalOrder(index);
            [index, offset] = this.NetworkOptimizer.mapFromOriginal(index);
        end
    end
    
    methods
        function this = DAGNetwork(sortedLayerGraph, topologicalOrder)
            %DAGNetwork - Create an internal DAGNetwork.
            %   this = DAGNetwork(sortedLayerGraph, topologicalOrder)
            %   creates an internal DAGNetwork. Input sortedLayerGraph is
            %   an internal LayerGraph containing a topologically sorted
            %   array of internal layers and input topologicalOrder is a
            %   vector representing the indices of the sorted internal
            %   layers in the original (unsorted) array of internal layers.
            
            % Save original connections. Rest of layer graph will be
            % optimized
            this.SortedConnections = sortedLayerGraph.Connections;
            this.TopologicalOrder = topologicalOrder;
            
            % Build additional metadata
            this = buildFromLayerGraphAnalysis( this, sortedLayerGraph );
        end
        
        function [activationsBuffer, memoryBuffer, layerIsLearning, states, hasPaged] = forwardPropagationWithMemory(this, X, propagateState, clearActivations)
            % forwardPropagationWithMemory   Forward propagation used by
            % training. Note, this version retains activations and memory,
            % but deletes any that won't be needed for backpropagation.
            %
            % Inputs
            %   X                      - an array containing the data
            %   propagateState         - logical scalar marking whether
            %                            recurrent state needs to be
            %                            propagated or not
            %   clearActivations       - logical scalar marking whether
            %                            activations which are not needed
            %                            for backward should be cleared or
            %                            not
            %
            % Output
            %   activationsBuffer     - cell array of activations required
            %                           for backpropagation
            %   memoryBuffer          - cell array of forward pass memory 
            %                           required for backpropagation
            %   layerIsLearning       - logical array which is true if a
            %                           layer has learnable parameters and
            %                           non-zero learning rate
            %   states                - cell array of state information
            %                           needed to update layer states after
            %                           gradient computation
            
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            listOfBufferIndicesForClearingForward = this.ListOfBufferIndicesForClearingForward;
            listOfBufferIndicesForClearingForwardForBackward =...
                this.ListOfBufferIndicesForClearingForwardForBackward;
            inputLayerIndices = this.InputLayerIndices;

            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations, memory and states
            activationsBuffer = cell(this.NumActivations,1);
            memoryBuffer = cell(this.NumLayers,1);
            states = cell(this.NumLayers, 1);
            
            % We can recover GPU memory by gathering the current
            % intermediate activations and memory cell arrays back to the
            % host.
            hasPaged = false;
            function gatherLayerOutputsAndMemory()
                hasPaged = true;
                activationsBuffer = iGatherGPUCell(activationsBuffer);
                memoryBuffer = iGatherGPUCell(memoryBuffer);
            end
            recoveryStrategies = {@gatherLayerOutputsAndMemory};
            
            layerIsLearning = false(this.NumLayers, 1);
            for i = 1:this.NumLayers
                % Mark whether this layer can learn, for backpropagation
                % optimisation
                thisLayer = this.Layers{i};
                learnablesThisLayer = thisLayer.LearnableParameters;
                layerIsLearning(i) = ~isempty(learnablesThisLayer) && ...
                    iHasANonZeroLearnRate(learnablesThisLayer);

                inputLayerMask = (i == inputLayerIndices);
                if any(inputLayerMask)
                    XForThisLayer = X{inputLayerMask};
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        listOfBufferInputIndices{i});
                    activationsBuffer = iClearActivationsFromBuffer( ...
                        activationsBuffer,...
                        listOfBufferIndicesForClearingForwardForBackward{i} );
                end
                if hasPaged && iIsACustomLayer(thisLayer)
                    XForThisLayer = iMoveToGpu( XForThisLayer );
                end
                [outputActivations, memory] = iExecuteWithStagedGPUOOMRecovery( ...
                    @() this.Layers{i}.forward( XForThisLayer ), ...
                    2, recoveryStrategies, i );
                
                bufferOutputIndices = listOfBufferOutputIndices{i};
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    bufferOutputIndices, ...
                    outputActivations);
                
                memoryBuffer = iAssignMemoryToBuffer( ...
                    memoryBuffer, i, memory);
                
                % Compute state information needed to update this layer if
                % this layer needs stateful training
                if any(i == this.StatefulIdx)
                    % bufferOutputIndices is scalar for recurrent layers.
                    % computeState does not use the second argument.
                    Z = activationsBuffer{bufferOutputIndices};
                    if hasPaged && iIsACustomLayer(thisLayer)
                        Z = iMoveToGpu(Z);
                    end
                    states{i} = computeState(this.Layers{i}, ...
                        [], Z, memoryBuffer{i}, propagateState);
                end
                
                % Throw away data from layers that aren't going to be
                % visited on the backward pass
                if clearActivations && ~any(layerIsLearning) && i > 1
                    indicesToClear = listOfBufferIndicesForClearingForward{i};
                    activationsBuffer = iClearActivationsFromBuffer( ...
                        activationsBuffer, indicesToClear );
                    memoryBuffer = iClearActivationsFromBuffer( ...
                        memoryBuffer, i );
                end
            end
        end
        
        function Y = predict(this, X)
            
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            listOfBufferIndicesForClearingForward = this.ListOfBufferIndicesForClearingForward;
            inputLayerIndices = this.InputLayerIndices;
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            
            % Loop over topologically sorted layers to perform forward
            % propagation. Clear memory when activations are no longer
            % needed.
            for i = 1:this.NumLayers
                thisLayer = this.Layers{i};
                inputLayerMask = (i == inputLayerIndices);
                if any(inputLayerMask)
                    outputActivations = thisLayer.predict(X{inputLayerMask});
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        listOfBufferInputIndices{i});
                    
                    outputActivations = thisLayer.predict(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    listOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    listOfBufferIndicesForClearingForward{i});
            end
            
            % Return activations corresponding to output layers.
            Y = activationsBuffer( ...
                [listOfBufferOutputIndices{this.OutputLayerIndices}] );
        end
        
        function [Y, states, finalStates] = statefulPredict(this, X, propagateState)
            % statefulPredict    Forward input data and returns a cell
            % array containing the output of each layer.
            %
            % Inputs
            %   X                - the input data
            %   propagateState   - a logical which is true when the
            %                      recurrent state from one batch should be
            %                      passed to the next
            % Outputs
            %   Y            - the output of the network
            %   states       - a cell array containing the states of any
            %                  layer with state parameters, used by the
            %                  updateNetworkState method
            %   finalStates  - a cell array containing the states of any
            %                  layer with state parameters, used by the
            %                  external network predict method
            
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            listOfBufferIndicesForClearingForward = this.ListOfBufferIndicesForClearingForward;
            inputLayerIndices = this.InputLayerIndices;
            
            states = cell(this.NumLayers, 1);
            finalStates = states;
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            
            % Loop over topologically sorted layers to perform forward
            % propagation. Clear memory when activations are no longer
            % needed.
            for i = 1:this.NumLayers
                thisLayer = this.Layers{i};
                inputLayerMask = (i == inputLayerIndices);
                if any(inputLayerMask)
                    outputActivations = thisLayer.predict(X{inputLayerMask});
                else
                    % Get X from the buffer
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        listOfBufferInputIndices{i});
                    if any(i == this.StatefulIdx)
                        % Forward propagate with memory
                        [outputActivations, memory] = thisLayer.forward( XForThisLayer );
                        
                        % Compute the state used for the updateNetworkState
                        % method
                        states{i} = thisLayer.computeState( ...
                            XForThisLayer, outputActivations, memory, propagateState );
                        
                        % Compute final state to be returned by the predict
                        % method
                        if propagateState
                            finalStates{i} = states{i};
                        else
                            finalStates{i} = thisLayer.computeState( ...
                                XForThisLayer, outputActivations, memory, true );
                        end
                    else
                        % Forward propagate without memory
                        outputActivations = thisLayer.predict( XForThisLayer );
                    end
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    listOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    listOfBufferIndicesForClearingForward{i});
            end
            
            % Return activations corresponding to output layers.
            Y = activationsBuffer( ...
                [listOfBufferOutputIndices{this.OutputLayerIndices}] );
        end
        
        function [Z, states] = activations(this, X, layerIndices, layerOutputIndices, propagateState)
            % activations   Support Fused Layers by calling into
            % activations method if requested output is internal to a
            % FusedLayer

            % Convert layerIndices into indices into the optimized layers
            % plus offsets
            [layerIndices, layerOffsets] = this.NetworkOptimizer.mapFromOriginal(layerIndices);
            
            % Convert statefulLayers into indices of the optimized layers
            states = cell(this.NumLayers, 1);
            
            % For fused layers where the user has requested an internal
            % activation, replace that with a request for the inputs to the
            % fused layer. Later we will compute the internal activations
            % for each of those requests.
            %
            % Get the 'true' indices and offsets for layerIndices that are
            % FusedLayers
            internalFusedLayers = cellfun(@iIsAFusedLayer, this.Layers);
            whichInputsAreFusedLayers = ismember(layerIndices, find(internalFusedLayers));
            fusedLayerIndices = layerIndices(whichInputsAreFusedLayers);
            fusedLayerOffsets = layerOffsets(whichInputsAreFusedLayers);
            fusedLayerOutputIndices = layerOutputIndices(whichInputsAreFusedLayers)';
            numFusedLayerOutputs = numel(fusedLayerIndices);
            %
            % Get the 'true' indices and offsets for the non-FusedLayers
            normalLayerIndices = layerIndices(~whichInputsAreFusedLayers);
            normalLayerOutputIndices = layerOutputIndices(~whichInputsAreFusedLayers);
            numNormalLayerIndices = numel(normalLayerIndices);
            %
            % Create a new list of layer indices and output port indices
            % for non-FusedLayers that includes the inputs to FusedLayers
            layerInputConnections = this.LayerGraphExecutionInfo.LayerInputConnections;
            for i = 1:numFusedLayerOutputs
                inputsToFusedLayers = layerInputConnections{fusedLayerIndices(i)};
                if ~isempty(inputsToFusedLayers)
                    inputsToFusedLayers = cat(1, inputsToFusedLayers{:});
                    normalLayerIndices = [normalLayerIndices; inputsToFusedLayers(:,1)]; %#ok<AGROW>
                    normalLayerOutputIndices = [normalLayerOutputIndices(:); inputsToFusedLayers(:,2)];
                end
            end
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Preparation
            numActivations = this.NumActivations;
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            listOfBufferIndicesForClearingForward = this.ListOfBufferIndicesForClearingForward;

            % Allocate space for the activations.
            activationsBuffer = cell(numActivations,1);
            
            % Convert layer indices and layer output indices into indices
            % for the activations buffer.
            normalLayerActivationIndices = cellfun( ...
               @(i,o)i(o), listOfBufferOutputIndices(normalLayerIndices), ...
               num2cell(normalLayerOutputIndices) );
            
            % Loop over topologically sorted layers to perform forward
            % propagation. Clear memory when activations are no longer
            % needed.
            maxLayerIndex = max(normalLayerIndices);
            for i = 1:maxLayerIndex
                thisLayer = this.Layers{i};
                inputLayerMask = (i == this.InputLayerIndices);
                if any(inputLayerMask)
                    outputActivations = thisLayer.predict(X{inputLayerMask});
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        listOfBufferInputIndices{i});
                    
                    if any(i == this.StatefulIdx)
                        % Forward propagate with memory
                        [outputActivations, memory] = thisLayer.forward( XForThisLayer );
                        
                        % Compute the state used for the updateNetworkState
                        % method
                        states{i} = thisLayer.computeState( XForThisLayer, ...
                            outputActivations, memory, propagateState );
                    else
                        % Forward propagate without memory
                        outputActivations = thisLayer.predict(XForThisLayer);
                    end
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    listOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                indicesToClear = setdiff( ...
                    listOfBufferIndicesForClearingForward{i}, ...
                    normalLayerActivationIndices);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    indicesToClear);
            end
            
            % Now compute the activations for internal fused layers
            fusedLayerActivations = cell(numFusedLayerOutputs,1);
            for j = 1:numFusedLayerOutputs
                % Get location of inputs and outputs
                layerIndex = fusedLayerIndices(j);
                layerOffset = fusedLayerOffsets(j);
                layerOutputIndex = fusedLayerOutputIndices(j);
                
                inputLayerMask = (layerIndex == this.InputLayerIndices);
                if any(inputLayerMask) % FusedLayer is input
                    XForThisLayer = X{inputLayerMask};
                else
                    inputsToFusedLayers = layerInputConnections{layerIndex};
                    inputsToFusedLayers = cat(1, inputsToFusedLayers{:});
                    
                    % Get input activations for this FusedLayer
                    bufferIndicesForAllOutputsFromInputs = listOfBufferOutputIndices(inputsToFusedLayers(:,1));
                    bufferIndicesForInputs = cellfun( @(x,indices)x(indices), ...
                        bufferIndicesForAllOutputsFromInputs, num2cell(inputsToFusedLayers(:,2)) );
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferIndicesForInputs );
                end
                
                % Call into FusedLayer's activations method
                fusedLayerActivations{j} = activations( this.Layers{layerIndex}, ...
                    XForThisLayer, layerOffset, layerOutputIndex );
            end
            
            % Reassemble the output activations from the normal layers and
            % the fused layers, in the order requested
            normalLayerActivationIndices = normalLayerActivationIndices(1:numNormalLayerIndices);
            Z(~whichInputsAreFusedLayers) = activationsBuffer(normalLayerActivationIndices);
            Z(whichInputsAreFusedLayers) = fusedLayerActivations;
        end
        
        function [gradients, predictions, states, inputGradients] = computeGradientsForTraining( ...
                this, X, Y, propagateState, dLossdOutput)
            % computeGradientsForTraining    Computes the gradients of the
            % loss with respect to the learnable parameters, from the
            % network input and response. This is used during training to
            % avoid the need to store intermediate activations and
            % derivatives any longer than is necessary.
            %
            % Inputs
            %   X                      - an array containing the data
            %   Y                      - expected responses
            %   propagateState         - logical scalar marking whether
            %                            recurrent state needs to be
            %                            propagated or not
            %   dLossdOutput           - The gradient of some quantity
            %                            w.r.t the output(s) of the 
            %                            network. Note that if you pass
            %                            this in, you can pass an empty
            %                            array for Y (since you won't need
            %                            labels for the loss calculation).
            %
            % Output
            %   gradients   - cell array of gradients with one element for
            %                 each learnable parameter array
            %   predictions - the output from the last layer, needs to be
            %                 preserved during training to report progress
            %   states      - cell array of state information needed to
            %                 update layer states after gradient
            %                 computation
            %   inputGradients
            %               - cell array of loss gradient w.r.t the inputs
            %                 of the network. NOTE: If you are training on
            %                 a GPU, this output COULD be a cell array of
            %                 non-GPU arrays if you are running low on GPU
            %                 memory.
            
            % Work out if we need to return input gradients
            needToCalculateInputGradients = (nargout == 4);
            if needToCalculateInputGradients
                indicesForInputActivations = ...
                    cell2mat(this.ListOfBufferOutputIndices(this.InputLayerIndices)');
            end
            
            % Work out if we need to use input adjoints (instead of
            % calculating the loss).
            useInputAdjoints = (nargin == 5);
            if useInputAdjoints
                dLossdOutput = iWrapInCell(dLossdOutput);
            end
            
            % Wrap X and Y in cell if needed
            X = iWrapInCell(X);
            Y = iWrapInCell(Y);
            
            % Do forward and get all activations
            [activationsBuffer, memoryBuffer, layerIsLearning, states, hasPaged] = ...
                this.forwardPropagationWithMemory( X, ...
                propagateState, ~needToCalculateInputGradients);
            
            % Set up the backpropagation function, which calls backwards on
            % each layer and then discards the activations and memory when
            % they are no longer needed
            dLossdXBuffer = cell(this.NumActivations,1);
            function dLossdW = efficientBackProp(currentLayer)
                
                % Preparation
                thisLayer = this.Layers{currentLayer};
                bufferInputIndices = this.ListOfBufferInputIndices{currentLayer};
                bufferOutputIndices = this.ListOfBufferOutputIndices{currentLayer};
                learnablesThisLayer = thisLayer.LearnableParameters;
                dLossdW = cell(size(learnablesThisLayer));
                
                % Output layers
                if any(currentLayer == this.OutputLayerIndices)
                    % Perform backpropagation for an output layer
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices);
                    if hasPaged && iIsACustomLayer(thisLayer)
                        ZForThisLayer = iMoveToGpu(ZForThisLayer);
                    end
                    [~, currentInputLayer] = find(this.OutputLayerIndices == currentLayer);
                    
                    if useInputAdjoints
                        % If the user has specified adjoints, use these
                        % instead of calculating the loss gradient.
                        dLossdX = dLossdOutput{currentInputLayer};
                    else
                        TForThisLayer = Y{currentInputLayer};
                        dLossdX = thisLayer.backwardLoss( ...
                            ZForThisLayer, TForThisLayer);
                    end
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX);
                    
                % Input layers
                elseif any(currentLayer == this.InputLayerIndices)
                    % Do nothing
                    
                % Other layers
                else
                    % Perform backpropagation for some other kind of
                    % layer
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferInputIndices);
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices);
                    dLossdZ = iGetTheseActivationsFromBuffer( ...
                        dLossdXBuffer, bufferOutputIndices);
                    memory = iGetTheseActivationsFromBuffer( ...
                        memoryBuffer, currentLayer);
                    
                    % Compute either all gradients or only the activations
                    % gradients depending on whether this layer is learning
                    backwardArgs = ...
                        { XForThisLayer, ZForThisLayer, dLossdZ, memory };
                    if hasPaged && iIsACustomLayer(thisLayer)
                        backwardArgs = iMoveToGpu(backwardArgs);
                    end
                    if layerIsLearning(currentLayer)
                        [dLossdX, dLossdW] = thisLayer.backward( backwardArgs{:} );
                    else
                        dLossdX = thisLayer.backward( backwardArgs{:} );
                    end
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX );
                end
                
                % Delete data that is no longer needed
                indicesToClear = this.ListOfBufferIndicesForClearingBackward{currentLayer};
                if needToCalculateInputGradients
                    % If we need to return input gradients, we need to make
                    % sure we don't clear them.
                    indicesToClear = setdiff(indicesToClear, indicesForInputActivations);
                end
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, indicesToClear );
                memoryBuffer = iClearActivationsFromBuffer( ...
                    memoryBuffer, currentLayer );
                dLossdXBuffer = iClearActivationsFromBuffer( ...
                    dLossdXBuffer, indicesToClear );
            end
            
            % We can recover GPU memory by gathering the current
            % intermediate activations back to the host.
            function gatherActivations()
                hasPaged = true;
                activationsBuffer = iGatherGPUCell(activationsBuffer);
            end
            recoveryStrategies = {@gatherActivations};
            %
            % We could also recover the memory and backward loss buffers
            function gatherBuffers()
                hasPaged = true;
                memoryBuffer = iGatherGPUCell(memoryBuffer);
                dLossdXBuffer = iGatherGPUCell(dLossdXBuffer);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherBuffers} ];
            %
            % We could also return gradients on the host instead of the GPU
            gradients = {};
            function gatherGradients()
                hasPaged = true;
                gradients = iGatherGPUCell(gradients);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherGradients} ];
            
            % To optimize away unnecessary backpropagation, determine
            % the earliest layer that needs its weight gradients computed
            if needToCalculateInputGradients
                earliestLearningLayer = 1;
            else
                earliestLearningLayer = find( layerIsLearning, 1, 'first' );
            end
                
            % Propagate loss and gradient back through the network
            for i = this.NumLayers:-1:1
                if i >= earliestLearningLayer
                    theseGradients = iExecuteWithStagedGPUOOMRecovery( ...
                        @() efficientBackProp(i), ...
                        1, recoveryStrategies, i );
                else
                    % Pad output even if propagation has stopped
                    theseGradients = cell(1, numel(this.Layers{i}.LearnableParameters) );
                end
                gradients = [theseGradients gradients]; %#ok<AGROW>
            end
            
            % Predict
            predictions = cell(1, this.NumOutputLayers);
            for i = 1:this.NumOutputLayers
                outputLayerBufferIndex = this.ListOfBufferOutputIndices{this.OutputLayerIndices(i)};
                predictions{i} = activationsBuffer{outputLayerBufferIndex};
            end
            if this.NumOutputLayers == 1
                predictions = predictions{1};
            end
            
            if hasPaged
                predictions = iMoveToGpu(predictions);
            end
            
            % Assign the input gradients if the caller has requested them
            % (4th output argument).
            if needToCalculateInputGradients
                inputGradients = dLossdXBuffer(indicesForInputActivations);
            end
        end
            
        function [gradients, predictions, states, inputGradients] = computeGradientsForTrainingFuzzy( ...
        this, X, Y, propagateState, windows, dLossdOutput)
            % computeGradientsForTraining    Computes the gradients of the
            % loss with respect to the learnable parameters, from the
            % network input and response. This is used during training to
            % avoid the need to store intermediate activations and
            % derivatives any longer than is necessary.
            %
            % Inputs
            %   X                      - an array containing the data
            %   Y                      - expected responses
            %   propagateState         - logical scalar marking whether
            %                            recurrent state needs to be
            %                            propagated or not
            %   dLossdOutput           - The gradient of some quantity
            %                            w.r.t the output(s) of the 
            %                            network. Note that if you pass
            %                            this in, you can pass an empty
            %                            array for Y (since you won't need
            %                            labels for the loss calculation).
            %
            % Output
            %   gradients   - cell array of gradients with one element for
            %                 each learnable parameter array
            %   predictions - the output from the last layer, needs to be
            %                 preserved during training to report progress
            %   states      - cell array of state information needed to
            %                 update layer states after gradient
            %                 computation
            %   inputGradients
            %               - cell array of loss gradient w.r.t the inputs
            %                 of the network. NOTE: If you are training on
            %                 a GPU, this output COULD be a cell array of
            %                 non-GPU arrays if you are running low on GPU
            %                 memory.
            
            % Work out if we need to return input gradients
            needToCalculateInputGradients = (nargout == 4);
            if needToCalculateInputGradients
                indicesForInputActivations = ...
                    cell2mat(this.ListOfBufferOutputIndices(this.InputLayerIndices)');
            end
            
            % Work out if we need to use input adjoints (instead of
            % calculating the loss).
            useInputAdjoints = (nargin == 6);
            if useInputAdjoints
                dLossdOutput = iWrapInCell(dLossdOutput);
            end
            
            % Wrap X and Y in cell if needed
            X = iWrapInCell(X);
            Y = iWrapInCell(Y);
            
            % Do forward and get all activations
            [activationsBuffer, memoryBuffer, layerIsLearning, states, hasPaged] = ...
                this.forwardPropagationWithMemory( X, ...
                propagateState, ~needToCalculateInputGradients);
            
            % Set up the backpropagation function, which calls backwards on
            % each layer and then discards the activations and memory when
            % they are no longer needed
            dLossdXBuffer = cell(this.NumActivations,1);
            function dLossdW = efficientBackProp(currentLayer)
                
                % Preparation
                thisLayer = this.Layers{currentLayer};
                bufferInputIndices = this.ListOfBufferInputIndices{currentLayer};
                bufferOutputIndices = this.ListOfBufferOutputIndices{currentLayer};
                learnablesThisLayer = thisLayer.LearnableParameters;
                dLossdW = cell(size(learnablesThisLayer));
                
                % Output layers
                if any(currentLayer == this.OutputLayerIndices)
                    % Perform backpropagation for an output layer
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices);
                    if hasPaged && iIsACustomLayer(thisLayer)
                        ZForThisLayer = iMoveToGpu(ZForThisLayer);
                    end
                    [~, currentInputLayer] = find(this.OutputLayerIndices == currentLayer);
                    
                    if windows.Enable == 1  
                        [max_value, max_posi] = max(squeeze(ZForThisLayer));
                        ZForThisLayer = zeros(size(ZForThisLayer));
                        for ii = 1:size(ZForThisLayer, 4)
                            sequence_list_pre = max_posi(ii)-windows.HalfLength;
                            sequence_list_post = max_posi(ii)+windows.HalfLength;
                            if (max_posi(ii)-windows.HalfLength) <= windows.MinValue/windows.Step
                                sequence_list_pre = windows.MinValue/windows.Step;
                            end
                            if (max_posi(ii)+windows.HalfLength) >= windows.MaxValue/windows.Step
                                sequence_list_post = windows.MaxValue/windows.Step;
                            end
                            if (sequence_list_pre-1)<=0
                                sequence_list_pre = 1;
                                ZForThisLayer(:,:,(sequence_list_post+1):end,ii) = 0;
                            end
                            if (sequence_list_pre+1)<size(squeeze(ZForThisLayer), 1)
                                sequence_list_post = size(squeeze(ZForThisLayer), 1);
                                ZForThisLayer(:,:,1:(sequence_list_pre-1),ii) = 0;
                            end
                            ZForThisLayer(:,:,sequence_list_pre:sequence_list_post,ii) = ZForThisLayer(:,:,sequence_list_pre:sequence_list_post,ii)/sum(ZForThisLayer(:,:,sequence_list_pre:sequence_list_post,ii));
                        end
                    end
                    if useInputAdjoints
                        % If the user has specified adjoints, use these
                        % instead of calculating the loss gradient.
                        dLossdX = dLossdOutput{currentInputLayer};
                    else
                        TForThisLayer = Y{currentInputLayer};
                        [max_value, max_posi] = max(squeeze(TForThisLayer));                      
                        
                        for ii = 1:size(TForThisLayer, 4)
                            sequence_list_pre = max_posi(ii)-windows.HalfLength;
                            sequence_list_post = max_posi(ii)+windows.HalfLength;
                            if (max_posi(ii)-windows.HalfLength) <= windows.MinValue/windows.Step
                                sequence_list_pre = windows.MinValue/windows.Step;
                            end
                            if (max_posi(ii)+windows.HalfLength) >= windows.MaxValue/windows.Step
                                sequence_list_post = windows.MaxValue/windows.Step;
                            end
                            if (sequence_list_pre-1)<=0
                                sequence_list_pre = 1;
                            end
                            if (sequence_list_pre+1)<size(squeeze(ZForThisLayer), 1)
                                sequence_list_post = size(squeeze(ZForThisLayer), 1);
                            end
                            
                            x = [sequence_list_pre:sequence_list_post];
                            TForThisLayer(:,:,sequence_list_pre:sequence_list_post,ii) = normpdf(x,max_posi,1);
                        end

                        dLossdX = thisLayer.backwardLoss( ...
                            ZForThisLayer, TForThisLayer);
                    end
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX);
                    
                % Input layers
                elseif any(currentLayer == this.InputLayerIndices)
                    % Do nothing
                    
                % Other layers
                else
                    % Perform backpropagation for some other kind of
                    % layer
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferInputIndices);
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices);
                    dLossdZ = iGetTheseActivationsFromBuffer( ...
                        dLossdXBuffer, bufferOutputIndices);
                    memory = iGetTheseActivationsFromBuffer( ...
                        memoryBuffer, currentLayer);
                    
                    % Compute either all gradients or only the activations
                    % gradients depending on whether this layer is learning
                    backwardArgs = ...
                        { XForThisLayer, ZForThisLayer, dLossdZ, memory };
                    if hasPaged && iIsACustomLayer(thisLayer)
                        backwardArgs = iMoveToGpu(backwardArgs);
                    end
                    if layerIsLearning(currentLayer)
                        [dLossdX, dLossdW] = thisLayer.backward( backwardArgs{:} );
                    else
                        dLossdX = thisLayer.backward( backwardArgs{:} );
                    end
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX );
                end
                
                % Delete data that is no longer needed
                indicesToClear = this.ListOfBufferIndicesForClearingBackward{currentLayer};
                if needToCalculateInputGradients
                    % If we need to return input gradients, we need to make
                    % sure we don't clear them.
                    indicesToClear = setdiff(indicesToClear, indicesForInputActivations);
                end
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, indicesToClear );
                memoryBuffer = iClearActivationsFromBuffer( ...
                    memoryBuffer, currentLayer );
                dLossdXBuffer = iClearActivationsFromBuffer( ...
                    dLossdXBuffer, indicesToClear );
            end
            
            % We can recover GPU memory by gathering the current
            % intermediate activations back to the host.
            function gatherActivations()
                hasPaged = true;
                activationsBuffer = iGatherGPUCell(activationsBuffer);
            end
            recoveryStrategies = {@gatherActivations};
            %
            % We could also recover the memory and backward loss buffers
            function gatherBuffers()
                hasPaged = true;
                memoryBuffer = iGatherGPUCell(memoryBuffer);
                dLossdXBuffer = iGatherGPUCell(dLossdXBuffer);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherBuffers} ];
            %
            % We could also return gradients on the host instead of the GPU
            gradients = {};
            function gatherGradients()
                hasPaged = true;
                gradients = iGatherGPUCell(gradients);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherGradients} ];
            
            % To optimize away unnecessary backpropagation, determine
            % the earliest layer that needs its weight gradients computed
            if needToCalculateInputGradients
                earliestLearningLayer = 1;
            else
                earliestLearningLayer = find( layerIsLearning, 1, 'first' );
            end
                
            % Propagate loss and gradient back through the network
            for i = this.NumLayers:-1:1
                if i >= earliestLearningLayer
                    theseGradients = iExecuteWithStagedGPUOOMRecovery( ...
                        @() efficientBackProp(i), ...
                        1, recoveryStrategies, i );
                else
                    % Pad output even if propagation has stopped
                    theseGradients = cell(1, numel(this.Layers{i}.LearnableParameters) );
                end
                gradients = [theseGradients gradients]; %#ok<AGROW>
            end
            
            % Predict
            predictions = cell(1, this.NumOutputLayers);
            for i = 1:this.NumOutputLayers
                outputLayerBufferIndex = this.ListOfBufferOutputIndices{this.OutputLayerIndices(i)};
                predictions{i} = activationsBuffer{outputLayerBufferIndex};
            end
            if this.NumOutputLayers == 1
                predictions = predictions{1};
            end
            
            if hasPaged
                predictions = iMoveToGpu(predictions);
            end
            
            % Assign the input gradients if the caller has requested them
            % (4th output argument).
            if needToCalculateInputGradients
                inputGradients = dLossdXBuffer(indicesForInputActivations);
            end
        end
        
        function loss = loss(this, Y, T)
            % Wrap Y and T in cell if needed
            Y = iWrapInCell(Y);
            T = iWrapInCell(T);
            
            % loss   Calculate the network loss
            loss = [];
            for i = 1:this.NumOutputLayers
                loss = [loss this.Layers{this.OutputLayerIndices(i)}.forwardLoss(Y{i}, T{i})]; %#ok<AGROW>
            end
            loss = sum(loss);
        end
        
        function this = setLearnableParameterValues(this, values)
            % setLearnableParameterValues   Set the value of each learnable
            % parameter
            parameterIndex = 1;
            for layerIndex = 1:this.NumLayers
                for layerParamIndex = 1:numel(this.Layers{layerIndex}.LearnableParameters)
                    this.Layers{layerIndex}. ...
                        LearnableParameters(layerParamIndex).Value = ...
                        values{parameterIndex};
                    parameterIndex = parameterIndex + 1;
                end
            end
        end
        
        function this = updateLearnableParameters(this, deltas)
            % updateLearnableParameters   Update each learnable parameter
            % value by subtracting a delta from it
            currentDelta = 1;
            for el = 1:this.NumLayers
                thisLayer = this.Layers{el};
                learnableParameters = thisLayer.LearnableParameters;
                numLearnables = numel(learnableParameters);
                if numLearnables > 0
                    this.Layers{el} = thisLayer.updateLearnableParameters( deltas(currentDelta:currentDelta+numLearnables-1) );
                    currentDelta = currentDelta + numLearnables;
                end
            end
        end
        
        function this = updateNetworkState(this, states, varargin)
            % updateNetworkState   Update network using state information
            % computed during gradient computation
            %
            % Inputs
            %   states                - cell array of state information
            %                           needed to update layer states after
            %                           gradient computation
            %   layerIndex (optional) - index of the final layer through
            %                           which we update the network state 
            % Output
            %   this                  - network with updated state
            if nargin > 2
                maxLayer = varargin{1};
                maxLayer = this.NetworkOptimizer.mapFromOriginal(maxLayer);
            else
                maxLayer = this.NumLayers - 1;
            end
            for currentLayer = 1:maxLayer
                if any(currentLayer == this.StatefulIdx)
                    this.Layers{currentLayer} = this.Layers{currentLayer}.updateState( states{currentLayer} );
                end
            end
        end
        
        function this = resetNetworkState(this)
            % resetNetworkState   Reset the stateful layers of the network
            % to their initial states
            %
            % Output
            %   this                  - network in initial state
            
            for currentLayer = 1:this.NumLayers-1
                if any(currentLayer == this.StatefulIdx)
                    initialState = this.Layers{currentLayer}.computeState([], [], [], false);
                    this.Layers{currentLayer} = this.Layers{currentLayer}.updateState( initialState );
                end
            end
        end
        
        function this = prepareNetworkForTraining(this, executionSettings)
            % prepareNetworkForTraining   Convert the network into a format
            % suitable for training

            % Prepare the layers
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.prepareForTraining();
            end
            
            % Determine whether training should occur on host or GPU
            if ismember( executionSettings.executionEnvironment, {'gpu'} )
                % Don't move data if training in parallel, allow this to
                % happen as training progresses. This ensures we can
                % support clients without GPUs when the cluster has GPUs.
                delayMove = executionSettings.useParallel || executionSettings.useStateless;
                this = this.setupNetworkForGPUTraining(delayMove);
            else
                this = this.setupNetworkForHostTraining();
            end
        end
        
        function this = prepareNetworkForPrediction(this)
            % prepareNetworkForPrediction   Convert the network into a
            % format suitable for prediction

            % Prepare the layers
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.prepareForPrediction();
            end
        end
        
        function this = optimizeNetworkForPrediction(this, optimizer)
            layerGraph = nnet.internal.cnn.LayerGraph( this.SortedLayers, this.SortedConnections );
            optimizedLayerGraph = optimizer.optimizeForPrediction( layerGraph );
            this = buildFromLayerGraphAnalysis( this, optimizedLayerGraph );
            
            this.NetworkOptimizer = optimizer;
        end
        
        function this = optimizeNetworkForTraining(this, optimizer)
            layerGraph = nnet.internal.cnn.LayerGraph( this.SortedLayers, this.SortedConnections );
            optimizedLayerGraph = optimizer.optimizeForTraining( layerGraph );
            this = buildFromLayerGraphAnalysis( this, optimizedLayerGraph );
            
            this.NetworkOptimizer = optimizer;
        end
        
        function this = setupNetworkForHostPrediction(this)
            % setupNetworkForHostPrediction   Setup the network to perform
            % prediction on the host

            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostPrediction();
            end
            this.UseGpu = false;
        end
        
        function this = setupNetworkForGPUPrediction(this)
            % setupNetworkForGPUPrediction   Setup the network to perform
            % prediction on the GPU
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForGPUPrediction();
            end
            this.UseGpu = true;
        end
        
        function this = setupNetworkForHostTraining(this)
            % setupNetworkForHostTraining   Setup the network to train on
            % the host
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostTraining();
                this.Layers{el} = this.Layers{el}.moveToHost();
            end
            this.UseGpu = false;
        end
        
        function this = setupNetworkForGPUTraining(this, deferMove)
            % setupNetworkForGPUTraining   Setup the network to train on
            % the GPU. deferMove allows the actual move of data to the GPU
            % to be deferred to happen as training progresses instead of in
            % advance.
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForGPUTraining();
                if ~deferMove
                    this.Layers{el} = this.Layers{el}.moveToGPU();
                end
            end
            this.UseGpu = true;
        end
        
        function indices = namesToIndices(this, stringArray)
            % namesToIndices   Convert a string array of layer names into
            % layer indices
            numLayersToMatch = numel(stringArray);
            indices = zeros(numLayersToMatch,1);
            layerNames = nnet.internal.cnn.layer.Layer.getLayerNames(this.Layers);
            for i = 1:numLayersToMatch
                indices(i) = find(strcmp(stringArray(i), layerNames));
            end
        end
        
        function layerIndex = getOriginalLayerIndex(this, layerIndex, offset)
            % getOriginalLayerIndex  Convert an internal layer index and
            % offset into an index into the Original layer graph. Use this
            % for error or other reporting so that indices match the user
            % input. The offset is needed for FusedLayers, to refer to an
            % underlying layer that has been fused.
            if nargin < 3
                offset = 1;
            end
            indices = this.NetworkOptimizer.mapToOriginal(layerIndex);
            layerIndex = this.TopologicalOrder(indices(offset));
        end
        
        function this = finalizeNetwork(this, X)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            
            for i = 1:this.NumLayers
                thisLayer = this.Layers{i};
                inputLayerMask = (i == this.InputLayerIndices);
                if any(inputLayerMask)
                    [Z, memory] = thisLayer.forward(X{inputLayerMask});
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        this.ListOfBufferInputIndices{i});
                    
                    [Z, memory] = thisLayer.forward(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    Z);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferIndicesForClearingForward{i});
                
                if isa( thisLayer, 'nnet.internal.cnn.layer.Finalizable' ) && ...
                        thisLayer.NeedsFinalize
                    thisLayer = finalize(thisLayer, XForThisLayer, Z, memory);
                end
                this.Layers{i} = thisLayer;
            end
        end
        
        function this = resetNetworkInitialization(this)
            for i = this.InputLayerIndices
                this.Layers{i} = reset(this.Layers{i});
            end
        end
        
        function this = initializeNetwork(this, statistics)
            for i = 1:this.NumInputLayers
                idx = this.InputLayerIndices(i);
                this.Layers{idx} = initialize(this.Layers{idx}, statistics{i});
            end
        end
        
        function tf = hasEmptyInputStatistics(this)
            tf = false;
            for i = this.InputLayerIndices
                tf = tf || needsInitialization(this.Layers{i});
            end
        end
        
        function this = inferSizes(this)
            % inferSizes   Infer layer output sizes
            
            sortedInternalLayers = this.Layers;
            numActivations = this.NumActivations;
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            
            this.Sizes = cell(numActivations,1);
            numLayers = numel(sortedInternalLayers);
            this.LayerOutputSizes = cell(numLayers,1);
            
            for i = 1:numLayers
                if any(i == this.InputLayerIndices)
                    inputSizesForThisLayer = sortedInternalLayers{i}.InputSize;
                else
                    inputSizesForThisLayer = iGetInputsFromBuffer( ...
                        this.Sizes, listOfBufferInputIndices{i});
                end
                
                sortedInternalLayers{i} = iInferSize( ...
                    sortedInternalLayers{i}, ...
                    inputSizesForThisLayer, ...
                    i);
                this.Layers{i} = sortedInternalLayers{i};
                
                outputSizesForThisLayer = sortedInternalLayers{i}.forwardPropagateSize( ...
                    inputSizesForThisLayer);
                this.Sizes = iAssignOutputsToBuffer( ...
                    this.Sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
                this.LayerOutputSizes{i} = outputSizesForThisLayer;
            end
        end
        
        function layerOutputSizes = inferOutputSizesGivenInputSizes(this, inputSizes, layerIndices)
            % inferOutputSizesGivenInputSizes   Infer output size from all
            % or a given layer given new input sizes for input layers.
            %
            % Suppose this internal DAG network has N layers which have
            % been topologically sorted and numbered from 1 to N. Suppose
            % the network has M input layers and they appear in positions
            % i_1, i_2, ..., i_M in the topologically sorted list.
            %
            % inputSizes       - is a length M cell array specifying the
            %                    input sizes for layers i_1, i_2, ..., i_M
            %                    in that order.
            %
            % layerIndices    -  the indices into the topologically
            %                    sorted, non-optimized list of layers
            %                    originally used to construct this
            %                    network.
            %
            % layerOutputSizes - a cell array of length
            %                    numel(layerIndices), where
            %                    layerOutputSizes{i} gives the output size
            %                    for layer layerIndices(i). If that layer
            %                    has multiple outputs then
            %                    layerOutputSizes{i} is a cell array of
            %                    output sizes for this layer.
            
            % Return all sizes if no layers specified
            numLayers = numel(this.Layers);
            if nargin < 3
                layerIndices = 1:numLayers;
            end
            numOutputs = numel(layerIndices);
            
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;

            % Convert layerIndices into indices into the optimized layers
            % plus offsets
            [layerIndices, layerOffsets] = this.NetworkOptimizer.mapFromOriginal(layerIndices);
            
            % Preallocate the size buffers
            %  sizes: Size of every output activation
            sizes = cell(this.NumActivations, 1);
            %  layerOutputSizes: The sizes requested
            layerOutputSizes = cell(numOutputs, 1);
            
            % Propagate sizes through the layers, filling the sizes buffer
            maxLayerIndex = max(layerIndices);
            for i = 1:maxLayerIndex
                thisLayer = this.Layers{i};
                inputLayerMask = (i == this.InputLayerIndices);
                isAnInputLayer = any(inputLayerMask);
                isAFusedLayer = iIsAFusedLayer(thisLayer);
                
                % Avoid propagating unnecessarily through a final
                % FusedLayer since we only need its input size
                if i == maxLayerIndex && isAFusedLayer
                    break;
                end
                
                % Get the input sizes for this layer
                if isAnInputLayer
                    inputSizesForThisLayer = inputSizes{inputLayerMask};
                else
                    inputSizesForThisLayer = iGetInputsFromBuffer( ...
                        sizes, listOfBufferInputIndices{i});
                end
                
                % Propagate through this layer, except for non-fused input
                % layers, which output the same size as the input
                if isAnInputLayer && ~isAFusedLayer
                    outputSizesForThisLayer = inputSizesForThisLayer;
                else
                    outputSizesForThisLayer = thisLayer.forwardPropagateSize( ...
                        inputSizesForThisLayer);
                end
                
                sizes = iAssignOutputsToBuffer( ...
                    sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
            end
            
            % Copy sizes from the buffer to the output. Where a layer is a
            % FusedLayer, propagate through that layer to the internal
            % layer requested.
            for i = 1:numOutputs
                thisLayerIndex = layerIndices(i);
                thisLayer = this.Layers{thisLayerIndex};
                if iIsAFusedLayer(thisLayer)
                    inputLayerMask = (thisLayerIndex == this.InputLayerIndices);
                    if any(inputLayerMask)
                        inputSizesForThisLayer = inputSizes{inputLayerMask};
                    else
                        inputSizesForThisLayer = iGetInputsFromBuffer( ...
                            sizes, listOfBufferInputIndices{thisLayerIndex});
                    end
                    outputSizesForThisLayer = thisLayer.forwardPropagateSize( ...
                        inputSizesForThisLayer, layerOffsets(i) );
                else
                    outputSizesForThisLayer = iGetInputsFromBuffer( ...
                        sizes, listOfBufferOutputIndices{thisLayerIndex} );
                end
                layerOutputSizes{i} = outputSizesForThisLayer;
            end
        end
        
        function layerHasSequenceOutput = inferSequenceOutput(this)
            % inferSequenceOutput   Determine whether each layer in the
            % layer graph has a sequence output. This method requires each
            % layer to have had "inferSize" called
            
            numLayers = this.NumLayers;
            layerHasSequenceOutput = false(numLayers, 1);
            
            % If the network is not an RNN, we expect no layers to have
            % sequence output and we can return early
            if ~this.IsRNN
                return
            end
            
            sortedInternalLayers = this.Layers;
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            seqLens = cell(numLayers, 1);
            
            for i = 1:numLayers
                if any(i == this.InputLayerIndices)
                    inputForThisLayer = iArbitrarySeqLen();
                else
                    inputForThisLayer = iGetInputsFromBuffer( ...
                        seqLens, listOfBufferInputIndices{i} );
                end
                inputForThisLayer = iWrapInCell(inputForThisLayer);
                
                outputForThisLayer = sortedInternalLayers{i}.forwardPropagateSequenceLength( ...
                    inputForThisLayer, this.Sizes(i) );
                seqLens = iAssignOutputsToBuffer( ...
                    seqLens, listOfBufferOutputIndices{i}, outputForThisLayer );
                layerHasSequenceOutput(i) = ...
                    any( cellfun(@(s)~isequal(s,1), outputForThisLayer) );
            end
        end
        
        function layerGraph = makeTrainedLayerGraph(this)
            % makeTrainedLayerGraph - makes an internal Layer graph
            % with most recent values of learnable parameters
            layerGraph = iMakeInternalLayerGraph(this.OriginalLayers, this.OriginalConnections);
        end
        
        function this = storeResponseData(this, responseMetaData)
            % Store any classification labels or response names in the
            % appropriate output layers.
            for i = 1:this.NumOutputs
                % Assert that the response meta data is compatible with the
                % output layer. This check complements those of
                % NetworkDataValidator.ValidateDataForProblem.
                try
                    this.OutputLayers{i} = ...
                        this.OutputLayers{i}.storeResponseMetaData( ...
                        responseMetaData(i) );
                catch exception
                    if isa(this.OutputLayers{i}, iClassificationLayer)...
                            && ~isprop(responseMetaData(i), 'Categories')
                        iThrowYNotCategoricalError(this);
                    elseif isa(this.OutputLayers{i}, iRegressionLayer)...
                            && ~isprop(responseMetaData(i), 'ResponseNames')
                        iThrowYNotValidResponseError(this);
                    else
                        throw(exception)
                    end
                end
            end
        end
        
        function networkInfo = computeNetworkInfo(this)
            isDAG = true;
            networkInfo = nnet.internal.cnn.util.ComputeNetworkInfo(...
                isDAG, this.SortedLayers);
            networkInfo = networkInfo.setNetworkSize(this);
            networkInfo = networkInfo.setLayerHasSequenceOutput(this);
        end
    end
    
    methods( Access = private )
        
        function this = buildFromLayerGraphAnalysis( this, optimizedLayerGraph )

            layers = optimizedLayerGraph.Layers;
            this.Layers = layers;
            this.LayerGraphExecutionInfo = nnet.internal.cnn.util.LayerGraphExecutionInfo(optimizedLayerGraph);
            
            layerInputConnections = this.LayerGraphExecutionInfo.LayerInputConnections;
            layerOutputConnections = this.LayerGraphExecutionInfo.LayerOutputConnections;
            this.NumInputLayers = iCountInputLayers(layerInputConnections);
            this.NumOutputLayers = iCountOutputLayers(layerOutputConnections);
            this.InputLayerIndices = iGetInputLayerIndices(layerInputConnections);
            this.OutputLayerIndices = iGetOutputLayerIndices(layerOutputConnections);
            this.FinalizableLayerIndices = iGetFinalizableLayerIndices(layers);
            
            this = inferSizes(this);
            this.InputSizes = iGetInputSizes(layers(this.InputLayerIndices));
            this.OutputSizes = iGetOutputSizes(this.LayerOutputSizes, ...
                this.OutputLayerIndices);
            
            % Save the internal connections. A layer graph with the most
            % recent values of learnable parameters can be accessed using
            % the LayerGraph property.
            this.Connections = optimizedLayerGraph.Connections;
            
            % Determine stateful layers
            this.StatefulIdx = find( nnet.internal.cnn.util.isStatefulLayer(layers) );
            
            % Determine if network outputs are sequences
            layersHaveSeqOutput = this.inferSequenceOutput();
            this.HasSequenceOutput = layersHaveSeqOutput(this.OutputLayerIndices);
        end
        
    end
    
end

function layerGraph = iMakeInternalLayerGraph(layers, connections)
layerGraph = nnet.internal.cnn.LayerGraph(layers, connections);
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end

function numInputLayers = iCountInputLayers(layerInputConnections)
numInputLayers = sum( cellfun(@iIsAnInputLayer, layerInputConnections) );
end

function numOutputLayers = iCountOutputLayers(layerOutputConnections)
numOutputLayers = sum( cellfun(@iIsAnOutputLayer, layerOutputConnections) );
end

function inputLayerIndices = iGetInputLayerIndices(layerInputConnections)
inputLayerIndices = find( cellfun(@iIsAnInputLayer, layerInputConnections) )';
end

function outputLayerIndices = iGetOutputLayerIndices(layerOutputConnections)
outputLayerIndices = find( cellfun(@iIsAnOutputLayer, layerOutputConnections) )';
end

function FinalizableLayerIndices = iGetFinalizableLayerIndices(layers)
FinalizableLayerIndices = find( cellfun(@iFinalizableLayer, layers) )';
end

function inputSizes = iGetInputSizes(inputLayers)
numInputLayers = numel(inputLayers);
inputSizes = cell(1, numInputLayers);
for i = 1:numInputLayers
    inputSizes{i} = inputLayers{i}.InputSize;
end
end

function outputSizes = iGetOutputSizes(sizes, outputLayerIndices)
numOutputLayers = numel(outputLayerIndices);
outputSizes = cell(1, numOutputLayers);
for i = 1:numOutputLayers
    currentLayer = outputLayerIndices(i);
    outputSizes{i} = sizes{currentLayer};
end
end

function tf = isEmptyOrHasEmpty(C)
tf = isempty(C) || any( cellfun(@isempty, C) );
end

function tf = iIsAnInputLayer(inputConnections)
tf = isEmptyOrHasEmpty(inputConnections);
end

function tf = iIsAnOutputLayer(outputConnections)
tf = isempty(outputConnections);
end

function tf = iFinalizableLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.Finalizable');
end

function tf = iIsAFusedLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.FusedLayer');
end

function activationsBuffer = iClearActivationsFromBuffer(activationsBuffer, indicesToClear)
activationsBuffer = nnet.internal.cnn.util.LayerGraphExecutionInfo.clearActivationsFromBuffer( ...
    activationsBuffer, indicesToClear);
end

function XForThisLayer = iGetTheseActivationsFromBuffer(activationsBuffer, inputIndices)
XForThisLayer = activationsBuffer(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end

function memoryBuffer = iAssignMemoryToBuffer(...
    memoryBuffer, ...
    bufferIndices, ...
    memory)
% FYI Batch norm and custom layers store their memory as a cell.
for i = 1:numel(bufferIndices)
    memoryBuffer{bufferIndices(i)} = memory;
end
end

function activationsBuffer = iAssignActivationsToBuffer( ...
    activationsBuffer, ...
    bufferIndices, ...
    activations)
if iscell(activations)
    activationsBuffer(bufferIndices) = activations;
else
    activationsBuffer{bufferIndices} = activations;
end
end

function activationsBuffer = iIncrementActivationsInBuffer(activationsBuffer, bufferIndices, activations)

numActivationsFromLayer = numel(bufferIndices);
if ~iscell(activations)
    if isempty(activationsBuffer{bufferIndices})
        activationsBuffer{bufferIndices} = activations;
    else
        activationsBuffer{bufferIndices} = activationsBuffer{bufferIndices} + activations;
    end
else
    for i = 1:numActivationsFromLayer
        if isempty(activationsBuffer{bufferIndices(i)})
            activationsBuffer{bufferIndices(i)} = activations{i};
        else
            activationsBuffer{bufferIndices(i)} = activationsBuffer{bufferIndices(i)}+ activations{i};
        end
    end
end
end

function internalLayer = iInferSize(internalLayer, inputSize, index)
if(~internalLayer.HasSizeDetermined)
    % Infer layer size if its size is not determined
    try
        internalLayer = internalLayer.inferSize(inputSize);
    catch e
        throwWrongLayerSizeException( e, index );
    end
else
    % Otherwise make sure the size of the layer is correct
    iAssertCorrectSize( internalLayer, index, inputSize );
end
end

function activationsBuffer = iAssignOutputsToBuffer( ...
    activationsBuffer, ...
    outputIndices, ...
    outputActivations)

numOutputsFromLayer = numel(outputIndices);
if ~iscell(outputActivations)
    activationsBuffer{outputIndices} = outputActivations;
else
    for i = 1:numOutputsFromLayer
        activationsBuffer{outputIndices(i)} = outputActivations{i};
    end
end
end

function iAssertCorrectSize( internalLayer, index, inputSize )
% iAssertCorrectSize   Check that layer size matches the input size,
% otherwise the architecture would be inconsistent.
if ~internalLayer.isValidInputSize( inputSize )
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception);
end
end

function throwWrongLayerSizeException(e, index)
% throwWrongLayerSizeException   Throws a getReshapeDims:notSameNumel exception as
% a WrongLayerSize exception
if (strcmp(e.identifier,'MATLAB:getReshapeDims:notSameNumel'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(message(errorID, varargin{:}));
end

function XForThisLayer = iGetInputsFromBuffer(layerOutputs, inputIndices)
XForThisLayer = layerOutputs(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end

function seq = iArbitrarySeqLen()
% Token to describe an arbitrary sequence length
seq = nnet.internal.cnn.util.arbitrarySequenceLengthToken();
end

function cellOrArray = iGatherGPUCell(cellOrArray)
if iscell(cellOrArray)
    cellOrArray = cellfun(@iGatherGPUCell, cellOrArray, 'UniformOutput', false);
elseif isa(cellOrArray, 'gpuArray')
    cellOrArray = gather(cellOrArray);
end
end

function varargout = iExecuteWithStagedGPUOOMRecovery(varargin)
[varargout{1:nargout}] = nnet.internal.cnn.util.executeWithStagedGPUOOMRecovery(varargin{:});
end

function X = iMoveToGpu(X)
if iscell(X)
    X = cellfun(@iMoveToGpu, X, 'UniformOutput', false);
elseif isnumeric(X)
    X = gpuArray(X);
end
end

function iThrowYNotCategoricalError(internalNetwork)
if ~iIsRecurrent(internalNetwork)
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:YIsNotCategoricalResponseVector'))
else
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:YIsNotValidSequenceCategorical'))    
end    
end

function iThrowYNotValidResponseError(internalNetwork)
if ~iIsRecurrent(internalNetwork)
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:YIsNotValidResponseArray'))
else
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:YIsNotValidSequenceResponse'))    
end    
end

function tf = iIsRecurrent(internalNetwork)
tf = nnet.internal.cnn.util.isRNN( internalNetwork.Layers );
end

function class = iClassificationLayer()
class = 'nnet.internal.cnn.layer.ClassificationLayer';
end

function class = iRegressionLayer()
class = 'nnet.internal.cnn.layer.RegressionLayer';
end

function tf = iIsACustomLayer(layer)
tf = isa(layer, 'nnet.internal.cnn.layer.CustomLayer') ...
    || isa(layer, 'nnet.internal.cnn.layer.CustomClassificationLayer') ...
    || isa(layer, 'nnet.internal.cnn.layer.CustomRegressionLayer');
end

function tf = iHasANonZeroLearnRate(layerLearnables)
learnRatesCell = { layerLearnables.LearnRateFactor };
hasNonZeroLearnRates = cellfun( @(lr)any(lr,'all'), learnRatesCell );
tf = any( hasNonZeroLearnRates );
end