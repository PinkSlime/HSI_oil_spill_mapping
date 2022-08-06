close all;
clear;
clc;
% load('Indiana200.mat');
load('oil_1.mat')
load('GT_1_0.05.mat')
GT=double(GT);
[AA BB ]=size(GT);

Indiana_data=reshape(oil,[AA*BB,270]);
Indiana_labels=reshape(GT,[AA*BB,1]);


lyl=[1,2,3,4,5,6,7,8,9];
data_ori =[];
labels=[];
C=9;
for j=1:C
data_ori =[data_ori,Indiana_data(find(Indiana_labels == lyl(j)), :)' ];
labels = [labels ,repmat(j,1,length(Indiana_labels(find(Indiana_labels == lyl(j)), :)'))];
end
%数据归一化
data = normalize_data(data_ori);

%对原始数据进行LFDA映射降到30维，用于后面的cdKNN
[T,ZZ]=lda(data_ori', labels', 30);   %T是映射矩阵，Z是降维后的数据T'*X
Z=ZZ';
zuizhong=[];
S=6;%稀疏度
K=20;%原子数目
%选取训练样本和测试样本
%select_train_data.m用来按比例选择训练样本
%select_train_data1.m用来按个数选择训练样本
percent = 0.1; %每类样本中训练样本比例
%N = 50;   %每类取N个作为训练样本
[train_index, test_index] = select_train_data(labels,percent);
%[train_index, test_index] = select_train_data1(labels, 50);

%用于cdOMP的归一化后数据
train_data = data(:, train_index);
train_label = labels(train_index);
test_data = data(:, test_index);
test_label = labels(test_index);
traincsma=tabulate(train_label(:));
mlstr=traincsma(:,2)';
testcsma=tabulate(test_label(:));
mlste=testcsma(:,2)';

%用于cdKNN的降维数据
train_data_ori = Z(:, train_index);
test_data_ori = Z(:, test_index);

X = train_data;
c = max(labels);

[nn,PP]=size(test_data);
%%新字典
X1=cell(1,PP);
XML=cell(max(train_label),PP);
A=cell(max(train_label),PP);
residual=zeros(max(train_label),PP);
tralabel=cell(1,PP);
tic
%cdKNN程序，得到【类别数*测试样本个数】大小的矩阵distance
[distance,I] = cdKNN(train_data_ori, test_data_ori, train_label, K);

for ii=1:PP
     XN=X(:, cell2mat(I(1,ii)));
     LABELN=train_label(:, cell2mat(I(1,ii)));
     for j=1:C
         X1{1,ii} =[cell2mat(X1(1,ii)),XN(:,find(LABELN == j)) ];
         tralabel{1,ii} = [cell2mat(tralabel(1,ii)) ,repmat(j,1,length(LABELN(:,find(LABELN == j))))];
         XML{j,ii}=XN(:,find(LABELN == j));
     end
end

cgtest=[];%重构
re=[];
result = zeros(1, length(test_label));
%cdOMP迭代程序，得到【类别数*测试样本个数】大小的矩阵residual
for j=1:PP
    for i=1:max(train_label) 
        ss=size(cell2mat(XML(i,j)),2);
        if ss==0
            residual(i, j) = 100;
        elseif (ss>0)&&(ss<S)
            A{i,j}=OMP(cell2mat(XML(i,j)), test_data(:,j), ss);
            nor = sqrt(sum((cell2mat(XML(i,j)) *cell2mat(A(i,j)) - test_data(:,j)).^2));
            residual(i, j) = nor;
        else
            A{i,j}=OMP(cell2mat(XML(i,j)), test_data(:,j), S);
            nor = sqrt(sum((cell2mat(XML(i,j)) *cell2mat(A(i,j)) - test_data(:,j)).^2));
            residual(i, j) = nor;
        end    
    end
    re=residual(:,j);
    [zd,result(j)] =min(re);   
end

[OA,kappa,AA,CA] = calcError(test_label-1, result-1, 1: C);
toc;

