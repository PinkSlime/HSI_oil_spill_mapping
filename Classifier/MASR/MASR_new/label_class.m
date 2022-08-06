function class_value =  label_class (tr_dat, tt_dat,s,NumClass,classids,tr_lab)
%=================================================================================
%This function is used to calculate residual for single scale 
%input arguments:  tr_dat       : training data
%                  tt_dat       : testing data of single scale
%                  s            : sparse matrix corresponding to single scale                      
%                  tr_lab       : label of training samples
%output arguments: class_value  : residual vector 
%=================================================================================
%Initialize residual for each class
gap     = zeros(1,NumClass);
%Calculate residual of all class
for j   =  1:NumClass
    temp_s =  zeros(size(s));
    class  =  classids(j);
    Index = find(class==tr_lab);
    temp_s(Index,:)  =  s(Index,:);
    zz     = tt_dat-tr_dat*temp_s;
    gap(j) =  zz(:)'*zz(:); 
end
class_value =gap  ;
