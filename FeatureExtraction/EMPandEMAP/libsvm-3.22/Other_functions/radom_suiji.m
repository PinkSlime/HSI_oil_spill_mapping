clc,clear all,close all
%% 加载indian_pines数据
load Indian_pines_corrected.mat 
load Indian_pines_gt.mat
img3D=indian_pines_corrected;
img_gt=indian_pines_gt; 


%% 加载PaviaU数据
% load PaviaU
% load PaviaU_gt
% img3D=paviaU;
% img_gt=paviaU_gt;


%% 加载Salinas数据
% load Salinas_corrected.mat 
% load Salinas_gt.mat
% img3D=salinas_corrected;
% img_gt=salinas_gt; 

%% 特征提取

 img2=average_fusion(indian_pines_corrected,20);
[no_lines, no_rows, no_bands] = size(img2);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
 fimg=spatial_feature(fimg,200,0.3);
[no_lines, no_rows, no_bands] = size(fimg);
fimg=reshape(fimg,[no_lines*no_rows no_bands]);
img2D=fimg';





% [i_row, i_col,bands] = size(img3D); 
[I J] = find( img_gt+1 );
cord = [I J];
N = size(cord,1);
% D = size(img3D,3);
labels = zeros(N,1);
% img2D = zeros(D,N);
for i = 1:N
%     img2D(:,i) = reshape( img3D(I(i),J(i),:), D, 1);
    labels(i) = img_gt( I(i),J(i) );
end
% [img2D] = scale_func(img2D);
class = max(labels) - min(labels);
%% %%%%样本选择
im_gt_1d = labels';
index = [];
label = [];
num_class = [];
for i = 0:1: class,
   index_t =  find(im_gt_1d == i); 
   index = [index index_t];  
   
   label_t = ones(1,length(index_t))*i; 
   label = [label label_t];
end
num_tr = [5 143 83 24 48 73 3 48 2 97 246 59 21 127 39 9]; 
trainSamples=[];
testSamples=[];
trainLabels=[];
testLabels=[];
train_index=[];
test_index=[];
for i = 1:1:class,
   label_c = find(label == i); 
   random_index = label_c(randperm(length(label_c)));
   temp = index(random_index(1:num_tr(i))); 
   trainSamples_i = img2D(:,temp);
   trainSamples = [trainSamples trainSamples_i];                              
   temp = index(random_index(1:num_tr(i))); 
   trainLabels_i = ones(1,length(temp))*i;         
   trainLabels = [trainLabels trainLabels_i];   
   train_index = [train_index temp]; 
   
   %%%%%%%%%%一一对应的关系
   
   temp = index(random_index(num_tr(i)+1:end));
   testSamples_i = img2D(:,temp);                   
   testSamples = [testSamples testSamples_i]; 
   testLabels_i = ones(1,length(temp))*i;
   testLabels = [testLabels testLabels_i]; 
   test_index = [test_index temp]; 
end
%% 
[Ccv Gcv cv cv_t]=cross_validation_svm(trainLabels',trainSamples');
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(trainLabels',trainSamples',parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg,model); 
% Evaluation
% GroudTest = double(test_labels(:,1));
ResultTest = Result(test_index,:);
[OA,AA,kappa,CA]=confusion(testLabels',ResultTest)