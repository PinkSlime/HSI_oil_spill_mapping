function [ OA  AA  kappa   CA  Result ] =ERW_OIL( img,GT,Tr,Te )

addpath('graphAnalysisToolbox-1.0');
addpath('EMPtoolbox');
addpath('ERW');
addpath('ERW\drtoolbox');
addpath('ERW\drtoolbox\techniques')
addpath('libsvm-3.14\matlab');

dim=max(unique(Te));
[no_lines, no_rows, no_bands] = size(img);
img = ToVector(img);
img = img';%200*21025
classes = dim ;
% classes       = 42;
GroundT= matricetotwo(GT)';
trainall =GroundT';
% train = trainall(:,indexes);%2*118
% % the test set
% test = trainall;%2*10249
%%
%%% get the training-test indexes
train = matricetotwo(Tr);
test = matricetotwo(Te);


%%% get the training-test samples and labels
% arrange the image struct
img = struct('im',img,'s',[no_lines no_rows no_bands classes]);

%size=img.sizep

% arrange the AL_sampling struct
AL_method='RS';  % active learning method, 
                   %    RS  -- random selection
                   %    MI  -- mutual information
                   %    BT  -- breaking ties
                   %    MBT -- mordified breaking ties
candidate =test;   % candidate set
U =  500;          %  total size of actively selected samples
u =  10;          %  new samples per iteration
AL_sampling = struct('AL_method',AL_method,'candidate',candidate,'U',U,'u',u);

% implement the supervised segmentation algorithm by using LORSAL and MLL
% spatial prior with active learnign
[ seg_result ] = ERW_AL(img,train,test,[]);
AA= seg_result.AA;
OA=seg_result.OA;
CA=seg_result.CA';
kappa =seg_result.kappa;
result=seg_result.map;

Result =reshape(result,[no_lines no_rows]);

end

