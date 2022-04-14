
function lgraph_1 = fuzzy_resnet50(numClasses)

% numClasses = 100;

%  Load the resnet50 module
net = resnet50;
layerTransfor = net.Layers(1:end-3);

%construct the connection between two layers
layers = [
    layerTransfor
    fullyConnectedLayer(numClasses,'Name',strcat('fc',num2str(numClasses)),'WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
    softmaxLayer('Name',strcat('fc',num2str(numClasses),'_softmax'))
    classificationLayer('Name',strcat('ClassificationLayer_fc',num2str(numClasses)))
    ];  
lgraph = layerGraph(layers);
 
%change the connection between two layers
lgraph = removeLayers(lgraph,'res2a_branch1');
lgraph = removeLayers(lgraph,'bn2a_branch1');
lgraph = removeLayers(lgraph,'res3a_branch1');
lgraph = removeLayers(lgraph,'bn3a_branch1');
lgraph = removeLayers(lgraph,'res4a_branch1');
lgraph = removeLayers(lgraph,'bn4a_branch1');
lgraph = removeLayers(lgraph,'res5a_branch1');
lgraph = removeLayers(lgraph,'bn5a_branch1');

layers_1=lgraph.Layers;
lgraph_1 = layerGraph(layers_1);
 
%add layers

res2a_branch1 = convolution2dLayer(1,256,'Name','res2a_branch1','Stride',1);
bn2a_branch1 = batchNormalizationLayer('Name','bn2a_branch1');
% -----------------------------------------------------------------------------------------------------------------------------------------------
res3a_branch1 = convolution2dLayer(1,512,'Name','res3a_branch1','Stride',2);
bn3a_branch1 = batchNormalizationLayer('Name','bn3a_branch1');
% -------------------------------------------------------------------------------------------------------------------------------------------------
res4a_branch1 = convolution2dLayer(1,1024,'Name','res4a_branch1','Stride',2);
bn4a_branch1 = batchNormalizationLayer('Name','bn4a_branch1');
% -------------------------------------------------------------------------------------------------------------------------------------------------
res5a_branch1 = convolution2dLayer(1,2048,'Name','res5a_branch1','Stride',2);
bn5a_branch1 = batchNormalizationLayer('Name','bn5a_branch1');
 
 
lgraph_1 = addLayers(lgraph_1,res2a_branch1);
lgraph_1 = addLayers(lgraph_1,bn2a_branch1);
lgraph_1 = addLayers(lgraph_1,res3a_branch1);
lgraph_1 = addLayers(lgraph_1,bn3a_branch1);
lgraph_1 = addLayers(lgraph_1,res4a_branch1);
lgraph_1 = addLayers(lgraph_1,bn4a_branch1);
lgraph_1 = addLayers(lgraph_1,res5a_branch1);
lgraph_1 = addLayers(lgraph_1,bn5a_branch1);

%change conections
 
 lgraph_1 = connectLayers(lgraph_1,'max_pooling2d_1','res2a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'res2a_branch1','bn2a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'bn2a_branch1','add_1/in2');
%  ------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_4_relu','add_2/in2');
% -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_7_relu','add_3/in2');
%  ----------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_10_relu','res3a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'res3a_branch1','bn3a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'bn3a_branch1','add_4/in2');
%  ------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_13_relu','add_5/in2');
% -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_16_relu','add_6/in2');
 % -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_19_relu','add_7/in2');
 %  ----------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_22_relu','res4a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'res4a_branch1','bn4a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'bn4a_branch1','add_8/in2');
%  ------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_25_relu','add_9/in2');
% -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_28_relu','add_10/in2');
 % -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_31_relu','add_11/in2');
 % -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_34_relu','add_12/in2');
 % -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_37_relu','add_13/in2');
  % -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_40_relu','res5a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'res5a_branch1','bn5a_branch1');
 lgraph_1 = connectLayers(lgraph_1,'bn5a_branch1','add_14/in2');
%  ------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_43_relu','add_15/in2');
% -------------------------------------------------------------------------------------
 lgraph_1 = connectLayers(lgraph_1,'activation_46_relu','add_16/in2');

%  analyzeNetwork(lgraph_1);
 
 
 