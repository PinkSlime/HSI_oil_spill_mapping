% Random Forest Classifier based on randomforest-matlab routine.
% http://code.google.com/p/randomforest-matlab/
% INPUT:
% 1- InputImage - MANDATORY
% 2- Train - MANDATORY
% 3- Test - MANDATORY
% 4- ImageIM - OPTIONAL
% 5- WriteOutput - OPTIONAL (default = true)
% InputImage: ENVI or TIF image of the scene.
% Train: Training image of the scene in GRAY SCALE (Classes denoted using numbers 0,1,2,3...).
% Test: Test image of the scene in GRAY SCALE (Classes denoted using numbers 0,1,2,3...).
% ImageIM: A single image RGB of the scene
% 
% The main goal of these functions is to provide a classification map of the image by using RF (ClassRF) classifier.
% The ImageIM is a RGB image of the scene, and if it is available, the test set is superimposed in order to provide to the user
% a better overview of the test set and of the goodness of the classification.
% A latex file with all the accuracies is written, if the WriteOutput flag is not 'false' (default=true);
% 
% EXAMPLE 1:
% ClassRF('PaviaPCA','Roi_Pavia_image_TRAINING.tif','Roi_Pavia_image_TEST.tif'); // ENVI CLASSIFICATION FILE 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXAMPLE 2:
% ClassRF('PaviaPCA','Roi_Pavia_image_TRAINING.tif','Roi_Pavia_image_TEST.tif','expPavia.png');// TIFF FILE 
%
% Mattia Pedergnana
% mattia@mett.it
% 14/07/2011
%%%%%%%


function [acc, TimeTook, FinalClass, Bands] = ClassRF(varargin)
% InputImage, TR, TE, ImageIM, WriteOutput
WBar = waitbar(0,'Wait.. performing a RF classification..');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Check the inputs %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
    help ClassRF_help
    error('You must specify at least three inputs');
end
%Format
% 0 = tiff
% 1 = envi
% 2 = both

%ImageReference
% 0 == no, 1 == auto, 2 == manual
if nargin == 3
    InputImage = varargin{1};
    Train = varargin{2};
    Test = varargin{3};
    WriteOutput = true;
    Format = 0;
    ImageIM = 0;
elseif nargin == 4
    InputImage = varargin{1};
    Train = varargin{2};
    Test = varargin{3};
    WriteOutput = varargin{4};
    Format = 0;
	ImageIM = 0;
elseif nargin == 5
    InputImage = varargin{1};
    Train = varargin{2};
    Test = varargin{3};
    WriteOutput = varargin{4};
    Format = varargin{5};
    ImageIM = 0;
elseif nargin == 6
    InputImage = varargin{1};
    Train = varargin{2};
    Test = varargin{3};
    WriteOutput = varargin{4};
    Format = varargin{5};
	ImageIM = varargin{6};
elseif nargin == 6
    InputImage = varargin{1};
    Train = varargin{2};
    Test = varargin{3};
    WriteOutput = varargin{4};
    Format = varargin{5};
    ImageIM = varargin{6}; 
end

if ischar(InputImage)
    D = enviread(InputImage);
    fprintf('\n\t Starting RANDOM FOREST CLASSIFY.. file: %s\n', InputImage);
else
    D = InputImage;
    InputImage=0; % clear
    fprintf('\n\t Starting RANDOM FOREST CLASSIFY.....');
end

if ischar(Train)
    Train = imread(Train);
end

if ischar(Test)
    Test = imread(Test);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Preprocessing %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Train, Test, ~ ] = InitializeRoiImage(Train, Test);

[row, col, Bands] = size(D);
si=size(Train);

if 	ImageIM ==1
ImageIM = zeros(row, col, 3);
    if Bands > 3
        ImageIM(:,:,1) = int8(mat2gray(D(:,:,int8(Bands/100*7+1)))*255);
        ImageIM(:,:,2) = int8(mat2gray(D(:,:,int8(Bands/100*16+1)))*255);
        ImageIM(:,:,3) = int8(mat2gray(D(:,:,int8(Bands/100*33)+1))*255);
    else
        disp('Warning, too few bands to make an automatic reference map');
        ImageIM = 0;
    end
end
D = double(D);
train_label=reshape(Train,1,si(:,1)*si(:,2))';
tridx = find(Train>0);
train_labels=double(train_label(tridx));
Data=reshape(D,row*col,Bands);

%  for i=1:Bands
%      Data(i,:)=double(mat2gray(Data(i,:)));
%  end
X=Data(train_label>0,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Classification %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start = tic;
model = classRF_train(X,train_labels,200);
waitbar(0.5);
res = classRF_predict(Data,model,200);
waitbar(0.9);
TimeTook = toc(start);
disp(['Elapsed Time: '  num2str(TimeTook) ' seconds']);

ClassifiedImage=reshape(res,si(:,1),si(:,2));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Accuracies %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[acc, FinalClass] = WriteClassifiedAndAccuracyFile(InputImage,...
                                    ClassifiedImage,...
                                    Test,...
                                    '_Class_RF',...
                                    ImageIM,...
                                    Bands,...
                                    WriteOutput,...
                                    Format);
close(WBar);
end

