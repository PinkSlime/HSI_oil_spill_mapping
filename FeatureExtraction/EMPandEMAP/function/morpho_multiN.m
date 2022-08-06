function [MP DMP]=morpho_multi(im,si,st,nb)
%
% function [MP DMP]=morpho(im,si,st,nb)
%
% This function compute the morphological profile and its
% derivative using a circular SE (disk) and return a solution in
% multispectral format: [h w nb_b]
%
% INPUT
%
% im: the images to processed
% si: the size of the se
% st: the step of the size
% nb the number of opening/closing
%
% OUTPUT
%
% MP: the morphological profile
% DMP: the Derivative of the MP


[H,W]=size(im);
imc=imcomplement(im);  % 对图像数据进行取反运算
MP=zeros(H,W,2*nb +1,'double');
i=1;
MP(:,:,nb+1)=im;  %retaining PC
% figure,subplot(1,9,nb+1),imshow(MP(:,:,nb+1),[]);title(sprintf('PCA'));  %显示进行MP前的图像也就是PCA图像
%Computing MP
for j=1:nb
 se=strel('disk',si,0);
%  se=strel('line',S,-45);
 tempO=imerode(im,se);
 tempC=imerode(imc,se);
 tempO=imreconstruct(tempO,im);
 tempC=imreconstruct(tempC,imc);
 MP(:,:,nb+1+i)=tempO;  %OP
%  subplot(1,9,nb+1+i),imshow(MP(:,:,nb+1+i),[]);title(sprintf('OP%d',nb+1+i));
 MP(:,:,nb+1-i)=imcomplement(tempC);  %CP
%  subplot(1,9,nb+1-i),imshow(MP(:,:,nb+1-i),[]); title(sprintf('CP%d',nb+1-i));
 si=si+st;
 i=i+1;
end
% % name=sprintf('indian_MP_multi%d.mat',S);
% name=sprintf('paviaU_MP_multi%d.mat',si);
% save(name,'MP');

if nargout >1
 %Computing DMP
%  DMP=zeros(H,W,2*nb,'single');
DMP=zeros(H,W,2*nb,'double');
 for i=2:(2*nb)
   DMP(:,:,i-1)=imabsdiff(MP(:,:,i+1),MP(:,:,i));
%    figure,imshow(DMP(:,:,i-1),[]); title(sprintf(' DMP%d',i-1));
 end
% % name=sprintf('indian_DMP_multi%d.mat',si);
% name=sprintf('paviaU_DMP_multi%d.mat',si);
% save(name,'DMP');
end