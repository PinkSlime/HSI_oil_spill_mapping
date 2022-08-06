function [ OA  AA  kappa   CA  Result ] =wzh_EPF( img,GroundT,Tr,Te )


%%%% give the parameters
no_classes=max(unique(Te));

%%%% estimate the size of the input image
[no_lines, no_rows, no_bands] = size(img);

%%%% vectorization
img = ToVector(img);
img = img';
GroundT= matricetotwo(GroundT)';

%%%% construct training and test datasets

%%% random selection


%%% get the training-test indexes
train_SL = matricetotwo(Tr);
test_SL = matricetotwo(Te);


%%% get the training-test samples and labels
train_samples = img(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
test_samples = img(:,test_SL(1,:))';
test_labels = test_SL(2,:)';

%%%% Normalize the training set and original image
[train_samples,M,m] = scale_func(train_samples);
[img ] = scale_func(img',M,m);

%%%% Select the paramter for SVM with five-fold cross validation
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);

%%%% Training using a Gaussian RBF kernel
%%% give the parameters of the SVM (Thanks Pedram for providing the
%%% parameters of the SVM)
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 


%%% Train the SVM
model=svmtrain(train_labels,train_samples,parameter);

%%%% SVM Classification
SVMresult = svmpredict(ones(no_lines*no_rows,1),img,model); 

%%%% Evaluation the performance of the SVM
GroudTest = double(test_labels(:,1));
SVMResultTest = SVMresult(test_SL(1,:),:);
[SVMOA,SVMAA,SVMkappa,SVMCA]=confusion(GroudTest,SVMResultTest)
%%%% Display the result of SVM 
SVMresult = reshape(SVMresult,no_lines,no_rows);
% SVMmap = label2color(SVMresult,'india');
% figure,imshow(SVMmap);

%%%% EPF based spectral-spatial classification

EPFresult = EPF(3,1,img,SVMresult);

%%% shows the computing time of EPF
EPFresult =reshape(EPFresult,[no_rows*no_lines 1]);
EPFresulttest = EPFresult(test_SL(1,:),:);
%%%% Evaluation the performance of the EPF
[OA,AA,kappa,CA]=confusion(GroudTest,EPFresulttest);
Result =reshape(EPFresult,[no_lines no_rows]);
%%%% Display the result of EPF 
% EPFmap=label2color(EPFresult,'india');
% figure,imshow(EPFmap);




end

