%function distance = cdKNN(X, y, train_label, K)
function [distance,I] = cdKNN(train_data_ori, test_data_ori, train_label, K)

%inputs:
%           train_data_ori: �ܵ�ԭʼѵ����������
%           test_data_ori: ԭʼ����������
%           train_label: ѵ��������Ӧ�����
%outputs:
%           distance: y��ÿ���K���ھ���
[m, n] = size(test_data_ori);
distance = zeros(1, n);
I=cell(1,n);
for p = 1:n
    y = test_data_ori(:, p);
    d = sqrt(sum((repmat(y, 1, size(train_data_ori, 2)) - train_data_ori).^2));
    [B,I{1,p}] = mink(d,K);
        
        %��ȡǰK����С���벢ȡ��ֵ
        %sort_d = sort(d, 'ascend'); %d����������
        %distance(i, p) = sum(sort_d(1:K)) / K; %K����С����ľ�ֵ��Ϊy�Ե�i��ľ�����Ϣ����
end

end

    