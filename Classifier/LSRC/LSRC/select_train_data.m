function [train_index, test_index] = select_train_data(labels, percent)
%���룺
%           labels: �������ݶ�Ӧ�����
%           percent: ÿ��ѡȡ��ѵ����������
%�����
%           train_index: ѡȡ��ѵ�����ݵ��±�
%           test_index: ʣ�µ���Ϊ���������±�

num_of_classes = max(labels);
train_index = [];
test_index = [];
for i = 1:num_of_classes
    
    %���ѡȡһ��������ѵ������
    ind = find(labels == i);
    L = length(ind);
    p = randperm(L);
    num = round(L*percent);
    train_index = [train_index, ind(p(1:num))];
    test_index = [test_index, ind(p(num+1:end))];
    
%     %ѡȡ��ǰ��һ��������ѵ������
%     ind = find(labels == i);
%     L = length(ind);
%     num = round(L * percent);
%     train_index = [train_index, ind(1:num)];
%     test_index = [test_index, ind(num+1:end)];
end

end