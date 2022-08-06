% Support Vector Machines (SVMs) classifier based on 
% LibSVM v3.0.1 
% http://www.csie.ntu.edu.tw/~cjlin/libsvm/

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
% The main goal of these functions is to provide a classification map of the image by using SVM (ClassSVM) classifier.
% The ImageIM is a RGB image of the scene, and if it is available, the test set is superimposed in order to provide to the user
% a better overview of the test set and of the goodness of the classification.
% A latex file with all the accuracies is written, if the WriteOutput flag is not 'false' (default=true);
%
% EXAMPLE 1:
% ClassSVM('PaviaPCA','Roi_Pavia_image_TRAINING.tif','Roi_Pavia_image_TEST.tif'); // ENVI CLASSIFICATION FILE 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXAMPLE 2:
% ClassSVM('PaviaPCA','Roi_Pavia_image_TRAINING.tif','Roi_Pavia_image_TEST.tif','expPavia.png');// TIFF FILE 
%
% Mattia Pedergnana
% mattia@mett.it
% 14/07/2011
%%%%%%%

function [OA,TimeTook,FinalClass,Bands] = ClassSVM(varargin)
% InputImage, Train, Test, ImageIM, WriteOutput
WBar = waitbar(0,'Wait.. performing a SVM classification..');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Check the inputs %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
    help ClassSVM_help
    error('You must specify at least three inputs');
end

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
    fprintf('\n\t Starting SVM CLASSIFY.. file: %s\n', InputImage);
    Img = enviread(InputImage);
else
    Img = InputImage;
	fprintf('\n\t Starting SVM CLASSIFY.....');
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
[Train, Test, ~] = InitializeRoiImage(Train, Test);
[~, ~, Bands] = size(Img);

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Classification %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tstart = tic;
waitbar(0.3);
ClassifiedImage = classify_svm(Img, Train);
waitbar(0.9);
TimeTook = toc(tstart);
disp(['Elapsed Time: '  num2str(TimeTook) ' seconds']);

fprintf('\n\t Number of Features: %i\n\n', Bands);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Accuracies %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[OA, FinalClass] = WriteClassifiedAndAccuracyFile(InputImage,...
                                    ClassifiedImage,...
                                    Test,...
                                    '_Class_SVM',...
                                    ImageIM,...
                                    Bands,...
                                    WriteOutput,...
                                    Format);
close(WBar);
end

