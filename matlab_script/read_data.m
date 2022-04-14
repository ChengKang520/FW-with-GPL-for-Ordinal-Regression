


clear all;
close all;clc;
%--------------------------------------------------------------------------
%                               Set Paths Here
%--------------------------------------------------------------------------
%% SET THESE PATHS FIRST!
Data_length = 6300;
FaceData.Face = uint8(zeros(224, 224, 3, Data_length));
FaceData.Age = zeros(1, Data_length);
Sample_num = 0;
CollectionDirectory = './FaceAgeData/wiki/';

AllCollectionsOfDDSM = struct2cell(dir(CollectionDirectory));
numOfFolders1 = size(AllCollectionsOfDDSM,2);

Data_part = 0;
for t = 1:numOfFolders1
    % DIR command also outputs "." and "..", so we need to skip those
    if(~strcmp(AllCollectionsOfDDSM{1,t}, '.') && ...
            ~strcmp(AllCollectionsOfDDSM{1,t}, '..') && ...
            AllCollectionsOfDDSM{5,t} == 1)

        %% Lets start.
        a = struct2cell(dir(strcat(CollectionDirectory, AllCollectionsOfDDSM{1,t})));
        numOfFolders = size(a,2);
        
        for i = 1:numOfFolders
            % DIR command also outputs "." and "..", so we need to skip those
            if(~strcmp(a{1,i}, '.') && ~strcmp(a{1,i}, '..')) %  && a{5,i} == 1
                
                % Get path of first folder.
                CaseDirectory = [CollectionDirectory, AllCollectionsOfDDSM{1,t}, '/'];
                img_file_name = a{1,i};
                pathToCaseFile = strcat([CaseDirectory, a{1,i}]);
                
                face_age = str2num(img_file_name(end-7:end-4)) - str2num(img_file_name(end-18:end-15));
                if isempty(face_age)
                    continue;
                end
                
                face_image = imread(pathToCaseFile);
                if length(size(face_image)) < 3
                    face_image_temp = uint8(zeros([size(face_image), 3]));
                    face_image_temp(:,:,1) = face_image;
                    face_image_temp(:,:,2) = face_image;
                    face_image_temp(:,:,3) = face_image;
                    face_image = face_image_temp;
                end
                face_image = imresize(face_image, [224 224]);
                Sample_num = Sample_num + 1;
                
                FaceData.Face(:,:,:,Sample_num) = face_image;
                FaceData.Age(Sample_num) = face_age;
                
                if Sample_num > Data_length
                    Data_part = Data_part + 1;
                    save_name = strcat('WIKI_FaceData_Part', num2str(Data_part));
                    save([CollectionDirectory save_name], 'FaceData');
                    
                    % initial them again
                    FaceData.Face = uint8(zeros(224, 224, 3, Data_length));
                    FaceData.Age = zeros(1, Data_length);
                    Sample_num = 0;
                end
            end
        end
    end
end



