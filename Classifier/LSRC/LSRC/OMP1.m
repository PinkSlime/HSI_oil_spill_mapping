%A-ϡ��ϵ������
%D-�ֵ�/����������֪��
%X-����ֵ������֪��
%L-ϡ���

function A=OMP(D,X,L,i)

[n,P]=size(X);
A=cell(1,P);
%[n,K]=size(cell2mat(D(1,1)));
for k=1:1:P
    [n,K]=size(cell2mat(D(i,k)));
    a=[];
    x=X(:,k);%��k����������
    residual=x;%�в�
    %r = zeros(1, P);
    indx=zeros(L,1);%������
    DD=cell2mat(D(i,k));
    for j=1:1:L
        proj=DD'*residual;%Dת����residual��ˣ��õ���residual��Dÿһ�е��ڻ�ֵ
        pos=find(abs(proj)==max(abs(proj)));%�ҵ��ڻ����ֵ��λ��
        pos=pos(1);%�����ֵ��ֹһ����ȡ��һ��
        indx(j)=pos;%�����λ�ô����������ĵ�j��ֵ
        a=pinv(DD(:,indx(1:j)))*x;%indx(1:j)��ʾ��һ��ǰj��Ԫ��
        residual=x-DD(:,indx(1:j))*a;
    end
    %r(k) = norm(residual);
    temp=zeros(K,1);
    temp(indx)=a;
    A{1,k}=temp;%ֻ��ʾ����ֵ����λ��
end
return
