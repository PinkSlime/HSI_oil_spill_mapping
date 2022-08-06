function [oa, aa, K, ca]=calcError(true_label,estim_label)
% This function is used to calculated the OA, AA  and the Kappa coefficient.


l=length(true_label);
nb_c=max(true_label);

%confu=zeros(nb_c,nb_c);
ErrorMatrix=zeros(16,16);
for i=1:l
  ErrorMatrix(estim_label(i),true_label(i))= ErrorMatrix(estim_label(i),true_label(i))+1;
end

oa=trace(ErrorMatrix)/sum(ErrorMatrix(:)); %overall accuracy
ca=diag(ErrorMatrix)./sum(ErrorMatrix,1)';  %class accuracy
ca(isnan(ca))=0;
number=size(ca,1);

aa=sum(ca)/number;


Po=oa;
Pe=(sum(ErrorMatrix)*sum(ErrorMatrix,2))/(sum(ErrorMatrix(:))^2);

K=(Po-Pe)/(1-Pe);%kappa coefficient