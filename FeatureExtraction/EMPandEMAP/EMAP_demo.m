clc

%

%

load(['oil.mat'       ]);

load('GT.mat');

load(['Tr.mat'       ]);

load(['Te.mat'       ]);



%% size of image
[no_lines, no_rows, no_bands] = size(img);
tic;

%     img=Normalization(img);
%
%     mask=GT;
%     mask(find(GT~=0))=1;
%     img=img.*double(mask);
%


fimg=EMAP(img,'', true, '', 'a', [  200 500 1000],'s',[2.5 5 7.5 10]);
% fimg=double(fimg);


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

% Normalizing Training and original img
[train_samples,M,m] = scale_func(train_samples);
[fimg ] = scale_func(fimg',M,m);
% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg,model);
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA_i,AA_i,kappa_i,CA_i]=confusion(GroudTest,ResultTest)
EMAPresult = reshape(Result,no_lines,no_rows);

