function  [ OA,AA,kappa,CA ,result_temp ,TT] = wzh_LSRC(oil,GT,Tr,Te)

[row,line,band]=size(oil);
%
Tr_1=Tr(:);
Te_1=GT(:);

TT=0;
tic

% AAAA=find(GT~=0);
% BBBB=find(Tr~=0);
% 
% CCCC=setdiff(AAAA,BBBB);
% 
% Tr_1(CCCC)=255;


% Tr_1(find(Tr_1==0))=[];
% Te_1(find(Te_1==0))=[];

index_1=(find(GT~=0))'; 


% Indiana_data=fea';
Indiana_data=reshape(oil,[row*line,band]);
% Indiana_data=Indiana_data';
% Indiana_labels=labels';
Indiana_labels=reshape(GT,[row*line,1]);


lyl=[1,2,3,4];
data_ori =[];
labels=[];
C=4;
train_index=[];
test_index=[];
train_label=[];
test_label=[];
index_2=[];



for j=1:C
    data_ori =[data_ori,Indiana_data(find(Indiana_labels == lyl(j)), :)' ];
    labels = [labels ,repmat(j,1,length(Indiana_labels(       find(Indiana_labels == lyl(j))   , :)'))  ];
    
    
    train_index_temp  =   Tr_1 (  find(Indiana_labels == lyl(j))  ) ;
    test_index_temp   =   Te_1 ( find(Indiana_labels == lyl(j)) );
    
    
    train_index=[train_index  train_index_temp' ];
    test_index=[test_index   test_index_temp'];
        
    
    index_temp=find(GT ==  lyl(j) );
    
    index_2=[index_2 , index_temp'];
    
    
end

train_index=find((train_index~=0))';
test_index=find((test_index~=0) )';
%���ݹ�һ��
data = normalize_data(data_ori);
train_data=[];
test_data=[];
% for j=1:C
%     train_data=[ train_data,  data(:,  train_index  )   ];
%     test_data=[ test_data,  data( :,  test_index    )   ];
% 
% end


%��ԭʼ���ݽ���LFDAӳ�併��30ά�����ں����cdKNN
[T,ZZ]=lda(data_ori', labels', 30);   %T��ӳ�����Z�ǽ�ά�������T'*X
Z=ZZ';
zuizhong=[];
S=6;%ϡ���
K=20;%ԭ����Ŀ
%ѡȡѵ�������Ͳ�������
%select_train_data.m����������ѡ��ѵ������
%select_train_data1.m����������ѡ��ѵ������
% percent = 0.1; %ÿ��������ѵ����������
%N = 50;   %ÿ��ȡN����Ϊѵ������\\
% train_index=(find(Tr~=0))';
% test_index=(find(Te~=0))';

% [train_index, test_index] = select_train_data(labels,percent);
%[train_index, test_index] = select_train_data1(labels, 50);

%����cdOMP�Ĺ�һ��������
train_data = data(:, train_index);
train_label = labels(train_index);
% train_label=

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
toc; 
TT=toc;

[OA,kappa,AA,CA] = calcError(test_label-1, result-1, 1: C);
result_temp=zeros(row,line);

for ii=1:length(result)
    result_temp(  index_2 (ii) )=result(ii);
end

% 
% [OA,AA,Ka,PA]=confusion(GT,result_temp);

