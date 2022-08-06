close all
clear all
clc
addpath('graphAnalysisToolbox-1.0');
addpath('EMPtoolbox');
addpath('ERW');
addpath('ERW\drtoolbox');
addpath('ERW\drtoolbox\techniques')
addpath('libsvm-3.14\matlab');
% no_sslsamples=5;
% no_ssl=ceil(no_sslsamples/16);
% R1 = ['ASSL_PU_MBI_ssl_',num2str(no_sslsamples)];
% R2 = ['SOASSL_PU_MBI_ssl_',num2str(no_sslsamples)];
% eval([R1,'=','[]']);
% eval([R2,'=','[]']);
ERW_IN_118=[];
tic
% ASSL_PU_45=[];
for i=1:10
% load image and ground truth
load IndiaP.mat;

% load training data
path='';
tt = 'IndiaP_118_';
location_tt = [path,tt,num2str(i)];
load (location_tt);

% 
[rows,cols,bands] = size(img);
img = ToVector(img);
img = img';%200*21025
classes       = 16;
% classes       = 42;
trainall =GroundT';
train = trainall(:,indexes);%2*118
% the test set
test = trainall;%2*10249
% arrange the image struct
img = struct('im',img,'s',[rows cols bands classes]);

%size=img.sizep

% arrange the AL_sampling struct
AL_method='RS';  % active learning method, 
                   %    RS  -- random selection
                   %    MI  -- mutual information
                   %    BT  -- breaking ties
                   %    MBT -- mordified breaking ties
candidate =test;   % candidate set
U =  600;          %  total size of actively selected samples
u =  10;          %  new samples per iteration
AL_sampling = struct('AL_method',AL_method,'candidate',candidate,'U',U,'u',u);

% implement the supervised segmentation algorithm by using LORSAL and MLL
% spatial prior with active learnign
[seg_results] = ERW_AL(img,train,test,[]);
ERW_IN_118  =[ERW_IN_118 seg_results];
% ERW_PU_45=[SOASSL_PU_45 seg_results];
i
end
toc
save ERW_IN_118 ERW_IN_118
