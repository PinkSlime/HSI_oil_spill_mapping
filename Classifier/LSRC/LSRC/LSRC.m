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
%���ݹ�һ��
data = normalize_data(data_ori);

%��ԭʼ���ݽ���LFDAӳ�併��30ά�����ں����cdKNN
[T,ZZ]=lda(data_ori', labels', 30);   %T��ӳ�����Z�ǽ�ά�������T'*X
Z=ZZ';
zuizhong=[];
S=6;%ϡ���
K=20;%ԭ����Ŀ
%ѡȡѵ�������Ͳ�������
%select_train_data.m����������ѡ��ѵ������
%select_train_data1.m����������ѡ��ѵ������
percent = 0.1; %ÿ��������ѵ����������
%N = 50;   %ÿ��ȡN����Ϊѵ������
[train_index, test_index] = select_train_data(labels,percent);
%[train_index, test_index] = select_train_data1(labels, 50);

%����cdOMP�Ĺ�һ��������
train_data = data(:, train_index);
train_label = labels(train_index);
test_data = data(:, test_index);
test_label = labels(test_index);
traincsma=tabulate(train_label(:));
mlstr=traincsma(:,2)';
testcsma=tabulate(test_label(:));
mlste=testcsma(:,2)';

%����cdKNN�Ľ�ά����
train_data_ori = Z(:, train_index);
test_data_ori = Z(:, test_index);

X = train_data;
c = max(labels);

[nn,PP]=size(test_data);
%%���ֵ�
X1=cell(1,PP);
XML=cell(max(train_label),PP);
A=cell(max(train_label),PP);
residual=zeros(max(train_label),PP);
tralabel=cell(1,PP);
tic
%cdKNN���򣬵õ��������*����������������С�ľ���distance
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

cgtest=[];%�ع�
re=[];
result = zeros(1, length(test_label));
%cdOMP�������򣬵õ��������*����������������С�ľ���residual
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

