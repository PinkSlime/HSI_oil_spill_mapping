
clc
clear

%
load(['oil.mat'       ]);

load('GT.mat');

load(['Tr.mat'       ]);

load(['Te.mat'       ]);


img=oil;



tic


[no_lines, no_rows, no_bands] = size(img);
%%% size of image
Nonzero_map = zeros(no_rows,no_lines);
Nonzero_index =  find(GT ~= 0);
Nonzero_map(Nonzero_index)=1;


img3D=img;
%% %%%%%% EMP
InitialSizeOfSE=2;
Step=2;
NumberOfOpeningClosing=4;
I = img3D;
PCs= PCA_img(I,3);
[row, col, Bands] = size(PCs);
for i=1:Bands
    %Normalization
    TempRE = reshape(PCs(:,:,i), row*col, 1 );
    TempRE = ((TempRE-mean(TempRE))/std(TempRE)+3)*1000/6;
    PCs(:,:,i) = reshape(TempRE, row, col, 1);
    tempBand = PCs(:,:,i);
    [MP DMP] = morpho_multiN(tempBand, InitialSizeOfSE, Step, NumberOfOpeningClosing);
    if(i ~= 1)
        OutEMP = cat(3, OutEMP, MP);
        OutDMP = cat(3, OutDMP, DMP);
    else
        OutEMP = MP;
        OutDMP = DMP;
    end
end
fimg=OutEMP;
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
EMPresult = reshape(Result,no_lines,no_rows);

CM=EMPresult;
CM=uint8(CM);
CM(find(GT==0))=0;
color_map=[
    0,0,0;
    120,152,225;
    118,218,145;
    248,149,136;
    248,203,127;
    124,214,207;
    250,109,29;
    153,135,206;
    145,146,171;
    118,80,5;
    ];
color_map=color_map/255;
gt=reshape(GT,1,no_lines*no_rows);
cm=reshape(CM,1,no_lines*no_rows);

[aa oa ua pa K confu]=new_confusion(GroudTest,ResultTest);

out_put=[aa,oa,K,pa'];

