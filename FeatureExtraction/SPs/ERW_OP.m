        function [ ERWresult,probability ] = ERW_OP( img,train,prob )

%%%%%���ӵ��λ�ú����ӵı��
seeds = [train(1,:)];
labels= [train(2,:)];  
gamma=0.16^5; % or 
 %gamma = 710 ;
 %beta = 0.1^5; 
 beta=5200;
[ERWresult,probability] = RWOptimize(img,seeds,labels,beta,prob,gamma,1); 


end

