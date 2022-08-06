function [alpha, r] = gradient_descent(X2, y, alpha, a, num_iters)
%inputs:
%           X1: ѵ������
%           y: ��������
%           alpha: ��������
%           a: 
%           num_iters: ����������
%outputs:
%           alpha: ����
%           r: ��ֵ����

m = length(y);
%A = [];
for i = 1:num_iters
    alpha = alpha - a / m * X2' * (X2 * alpha - y);
    r = y - X2 * alpha;
    %A(i) = norm(r);
end
%plot(A);

end