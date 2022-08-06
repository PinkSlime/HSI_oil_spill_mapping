function [reflectance shading] = IntrinsicImage_GS_new(img,wd,iterNum,rho)
% intrinsic images using optimization without user interaction

epsilon = 1/10000;
img=double(img);
n = size(img,1); 
m = size(img,2);
number_bands=size(img,3);
img=reshape(img,[n*m number_bands]);
rgbIm = scale_new(img)+ epsilon;
rgbIm=reshape(rgbIm,[n m number_bands]);
ntscIm = mean(rgbIm,3);

imgSize = n*m;

%% calcutlate local weight
display('Compute Weight matrix ...');


% normalize RGB triplets
rgbIm2 = sum(rgbIm.^2,3);
rgbIm_c = rgbIm ./ (repmat(rgbIm2.^(1/2), [1 1 number_bands]));

gvals = zeros(1,(2*wd+1)^2);        % luminlances in a local window
t_cvals = zeros((2*wd+1)^2,number_bands);      % angles in a local window

num_W = (2*wd+1)^2-1;                  
W = zeros(imgSize,num_W);
n_idx = ones(imgSize,num_W);
pixelIdx = reshape(1:imgSize,n,m);
  
len = 0;
for j=1:m
    for i=1:n      
        len=len+1;
        tlen=0;            
        for ii=max(1,i-wd):min(i+wd,n)
            for jj=max(1,j-wd):min(j+wd,m)           
                if (ii~=i)||(jj~=j)
                    tlen=tlen+1;
                    gvals(tlen)=ntscIm(ii,jj,1); 
                    t_cvals(tlen,:) = reshape(rgbIm_c(ii,jj,:),1,number_bands);
                    n_idx(len,tlen) = pixelIdx(ii,jj);                  
                end
            end
        end

        %����Yֵ�ķ���
        t_val=ntscIm(i,j,1);            
        gvals(tlen+1)=t_val;                    
        c_var=mean((gvals(1:tlen+1)-mean(gvals(1:tlen+1))).^2);                              %change                   
        csig=c_var*0.6;         

        mgv=min((gvals(1:tlen)-t_val).^2);
        if (csig<(-mgv/log(0.01)))
            csig=-mgv/log(0.01);
        end
        if (csig<0.000002)
            csig=0.000002;
        end        

        %���ڼнǵķ���
        t_cval = reshape(rgbIm_c(i,j,:),1,number_bands);        
        cvals = sum(t_cvals(1:tlen,:) .* repmat(t_cval,[tlen 1]),2);
        inds = cvals > 1;
        cvals(inds) = 1;
        cvals = acos(cvals);
        c_var_cvals = mean((cvals - mean(cvals)).^2); 
        csig_cvals = c_var_cvals*0.6;             
        mgv_cvals=min((cvals-1).^2);                              %change
        if (csig_cvals<(-mgv_cvals/log(0.01)))                              %change
            csig_cvals=-mgv_cvals/log(0.01);                              %change
        end                                                     %change
        if (csig_cvals<0.000002)                              %change
            csig_cvals=0.000002;                              %change
        end  

        gvals(1:tlen)=exp(-(((gvals(1:tlen)-t_val).^2/csig)+(cvals'.^2/ csig_cvals.^2)));
        tmp_sum = sum(gvals(1:tlen));
        gvals(1:tlen)=gvals(1:tlen)/tmp_sum;        

        W(len,1:tlen) = gvals(1:tlen); 
    end
end

if len ~= imgSize
    display('there are erros in creating the W matrix!');
    return;
end


%% iteration
R = 0.5*ones(imgSize,number_bands);
inv_S = 2*ones(imgSize,1);
rgbIm = reshape(rgbIm,imgSize,number_bands);
rgbIm2 = reshape(rgbIm2,imgSize,1);

for iter = 1:iterNum

    sI = repmat(inv_S, [1 number_bands]) .* rgbIm;
    len = 0;     
    for j = 1:m
       for i = 1:n
           len = len + 1;       
           sumR = W(len,:)*R(n_idx(len,:),:) + sI(len,:);
           R(len,:) = (1-rho)*R(len,:) + 0.5*rho*sumR;
           R(len,:) = max(epsilon, min(1, R(len,:)));
       end
    end 

    inv_S = (1-rho)*inv_S + rho*(sum(rgbIm.*R,2)./rgbIm2);
    inv_S = max(1, inv_S);

end

%%
reflectance = reshape(R, [n m number_bands]);
shading = reshape(1./inv_S, n, m);
