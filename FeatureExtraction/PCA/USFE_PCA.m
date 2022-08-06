function [ OA  AA  kappa   CA  Result ]  = USFE_PCA (img, GroundT,Tr,Te,dim)
[no_lines, no_rows, no_bands] = size(img);
data=reshape(img,no_lines*no_rows, no_bands);
clear img
%% matlab PCA from the stats toolbox (C:\Program Files\MATLAB\R2018b\toolbox\stats\stats\pca.m)
[code] = pca(data');

fimg=reshape(code(:,1:dim),no_lines,no_rows,dim);


GroundT=GroundT';
OA=[];AA=[];kappa=[];CA=[];




%% SVM classification
fimg = ToVector(fimg);
fimg = fimg';
fimg=double(fimg);


%%% traing and test samples
train_SL = matricetotwo(Tr);
train_samples = fimg(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
%
test_SL = matricetotwo(Te);
% test_SL(:,indexes) = [];
test_samples = fimg(:,test_SL(1,:))';
test_labels = test_SL(2,:)';

[train_samples,M,m] = scale_func(train_samples);
[fimg11 ] = scale_func(fimg',M,m);
% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg11,model); 
% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA,AA,kappa,CA]=confusion(GroudTest,ResultTest);
Result = reshape(Result,no_lines,no_rows);
% VClassMap=label2colord(Result,'india');
% figure,imshow(VClassMap);



end





