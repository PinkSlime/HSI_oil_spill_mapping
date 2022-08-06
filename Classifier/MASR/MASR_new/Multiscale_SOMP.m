function [A]=Multiscale_Dynamic_SOMP(D,Multiscale_Y,L, scale_num,tr_lab); 
%=============================================
% Modified SOMP algorithm to sparse representation multiscale pixels
% based on a given structure dictionary and specified number of atoms to use. 
% input arguments: D            : the dictionary
%                  Multiscale_Y : multiscale training samples matrix
%                  L            : sparsity level
%                  scale_num    : number of scale
%                  tr_lab       : labels of training samples
% output arguments: A: sparse coefficient matrix 
%=============================================

[n,K]=size(D);
classids=unique(tr_lab);
% Initialize sparse matrix 
A = {};
% Initialize residual error matrix
Multiscale_residual = {};
% Generate residual error matrix for each scale
for iii = 1: scale_num
 Multiscale_residual{iii} =  Multiscale_Y{iii}; 
 A{iii} = zeros(size(D,2),size(Multiscale_Y{iii},2));
end

indx = [];
a = {};
%% Select a most appropriate atoms from the dictionary for each scale and pixel at each iteration
for l=1:1:L,
    proj_t_whole = zeros(K,scale_num);
    for ijj = 1: scale_num
       proj = [];
       proj=D'*Multiscale_residual{ijj};
       proj_t =[];
       proj_t = max(abs(proj),[],2);
       proj_t_whole(:,ijj) =  proj_t;
    end
    
    max_value = [];
    max_index = [];
    for classi = 1: length(classids)
     class_label = find( tr_lab == classi);   
     Class_proj_temp = proj_t_whole(class_label,:);   
     [max_value_temp, max_index_temp] = max(Class_proj_temp,[],1);       
     max_value(classi) = sum(max_value_temp);
     max_index(classi,:) = max_index_temp - 1 + class_label(1);
    end
    
    [max_m, index_m] = max(max_value);
    max_final = max_index(index_m,:);
    indx(l,:) = max_final;
    
    for nscale = 1: scale_num
       a{nscale}=pinv(D(:,indx(1:l,nscale)))*Multiscale_Y{nscale};
       Multiscale_residual{nscale}=Multiscale_Y{nscale}-D(:,indx(1:l,nscale))*a{nscale};    
    end
end
if (length(indx)>0)
    for nnscale = 1: scale_num
    A{nnscale}(indx(:,nnscale),:) = a{nnscale};        
    
    end
end
return;

