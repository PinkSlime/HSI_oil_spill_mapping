function [Result, out ] = pre_classification( img,Tr,Te )
[no_lines, no_rows, no_bands] = size(img);
%% feature extraction
% load fea
% fimg_fea=fea;
fimg_fea=SDSE(img);
% load Gabor3D
% fimg_fea=img_gabor3D;
% load Gabor3D
% fimg_fea=img_gabor3D;
%  fimg_fea=kpca(fimg_sub, 1000,18, 'Gaussian',20);
fimg2 = ToVector(fimg_fea);
fimg2 = fimg2';
fimg2=double(fimg2);
%% Training and testing samples
train_SL=matricetotwo(Tr);
train_samples = fimg2(:,train_SL(1,:))';
train_labels= train_SL(2,:)';

test_SL=matricetotwo(Te);
test_samples = fimg2(:,test_SL(1,:))';
test_labels = test_SL(2,:)';

%% 
% no_classes=max(Te(:));
% labels=Te+Tr;
% train_number = ones(1,no_classes)*2;
% [train_SL,test_SL,test_number]= GenerateSample(labels,train_number,no_classes);
% train_samples = fimg2(:,train_SL(1,:))';
% train_labels= train_SL(2,:)';
% test_samples = fimg2(:,test_SL(1,:))';
% test_labels = test_SL(2,:)';
%% Spectral classifier
% Normalizing Training and original img 
[train_samples,M,m] = scale_func(train_samples);
[fimg3 ] = scale_func(fimg2',M,m);
% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
%give the parameters of the SVM (Thanks Pedram for providing the
% parameters of the SVM)
parameter=sprintf('-s 0 -t 2 -c %f -g %f -m 500 -b 1',Ccv,Gcv); 
%%% Train the SVM
model=svmtrain(train_labels,train_samples,parameter);
[Result, accuracy,prob] = svmpredict(ones(no_lines*no_rows,1) ,fimg3,model,'-b 1');  
prob=reshape(prob,[no_lines no_rows max(Result(:))]);
out=prob;

end

