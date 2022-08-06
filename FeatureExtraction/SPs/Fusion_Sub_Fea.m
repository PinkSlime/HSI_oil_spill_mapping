clc;
clear;
close all;

addpath('SDSE');
addpath('Subfeature')
addpath('Datasets');
addpath('Compositekernel')
addpath('MSTV')
addpath('SVM')
addpath('ERW')
% addpath('SVM');
%% input your HS data
load Houston18
load Houston_Te_Tr
gt=Te+Tr;
GroundT=matricetotwo(gt);
load XX
indexes=XX(:,1);
Train_SL=GroundT(:,indexes);
Tr=zeros(601,2384);
Tr(Train_SL(1,:))=Train_SL(2,:);
Test_SL=GroundT;
Test_SL(:,indexes)=[];
Te=zeros(601,2384);
Te(Test_SL(1,:))=Test_SL(2,:);
%%
img=Normalization(img);
%% size of image 
[no_lines, no_rows, no_bands] = size(img);
tic;
% load Houston_Te_Tr
test_SL=matricetotwo(Te);
test_labels = test_SL(2,:)';
[Pre_re Pre_pro ] = pre_classification( img,Tr,Te );
[ Pos_re Pos_pro] = post_classification( img,Tr,Te );
t=0.5;
Fuse_pro=t.*Pre_pro+(1-t).*Pos_pro;
[Class_pro,Fuse_Result]=max(Fuse_pro,[],3);
Result=reshape(Fuse_Result,[no_lines*no_rows 1]);
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);

ResultPre = Pre_re(test_SL(1,:),:);
ResultPos = Pos_re(test_SL(1,:),:);
% ResultTest1 = Sresult(test_SL(1,:),:);
[Pre_OA,Pre_AA,Pre_Kappa,Pre_CA]=confusion(GroudTest,ResultPre);

[Pos_OA,Pos_AA,Pos_Kappa,Pos_CA]=confusion(GroudTest,ResultPos);
[SVM_OA,SVM_AA,SVM_Kappa,SVM_CA]=confusion(GroudTest,ResultTest);
time=toc
Result = reshape(Result,no_lines,no_rows);
VClassMap=label2colord(Result,'hu');
figure,imshow(VClassMap);
disp('%%%%%%%%%%%%%%%%%%% Classification Results of SDSE Method %%%%%%%%%%%%%%%%')
disp(['OA',' = ',num2str(SVM_OA),' ||  ','AA',' = ',num2str(SVM_AA),'  ||  ','Kappa',' = ',num2str(SVM_Kappa)])
