%A-ϡ��ϵ������
%D-�ֵ�/����������֪��
%X-����ֵ������֪��
%L-ϡ���
function A=OMP(D,X,L)
[n,P]=size(X);
[n,K]=size(D);
for k=1:1:P
    a=[];
    x=X(:,k);
    residual=x;%�в�
    %r = zeros(1, P);
    indx=zeros(L,1);%������
    for j=1:1:L
        proj=D'*residual;%Dת����residual��ˣ��õ���residual��Dÿһ�е��ڻ�ֵ
        pos=find(abs(proj)==max(abs(proj)));%�ҵ��ڻ����ֵ��λ��
        pos=pos(1);%�����ֵ��ֹһ����ȡ��һ��
        indx(j)=pos;%�����λ�ô����������ĵ�j��ֵ
        a=pinv(D(:,indx(1:j)))*x;%indx(1:j)��ʾ��һ��ǰj��Ԫ��
        residual=x-D(:,indx(1:j))*a;
    end
    %r(k) = norm(residual);
    temp=zeros(K,1);
    temp(indx)=a;
    A(:,k)=temp;%ֻ��ʾ����ֵ����λ��
end