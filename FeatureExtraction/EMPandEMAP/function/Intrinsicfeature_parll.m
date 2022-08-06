
function [fimage X] = Intrinsicfeature_parll(img,k)
%SPATIAL_FEATURE Summary of this function goes here
%   Detailed explanation goes here
[r c bands]=size(img);
if floor(bands/k)==0
    [fimage x]=IntrinsicImage_GS_new(img,3,100,1.9);
    X=x;
else
parfor i=1:floor(bands/k)
[y x]=IntrinsicImage_GS_new(img(:,:,(i-1)*k+1:i*k),3,100,1.9);
X(:,:,i)=x;
y=reshape(y,[r,c*k]);
Y(:,:,i)=y;
end
fimage=reshape(Y,[r c floor(bands/k)*k]);
if floor(bands/k)<bands/k
   [fimage(:,:,floor(bands/k)*k+1:bands) x]=IntrinsicImage_GS_new(img(:,:,floor(bands/k)*k+1:bands),3,100,1.9);
   X(:,:,ceil(bands/k))=x;
end
end


% if floor(bands/k)==0
%     [fimage x]=IntrinsicImage_GS_new(img,3,100,1.9);
%     X=x;
% else
% for i=1:floor(bands/k)
% [fimage(:,:,(i-1)*k+1:i*k) x]=IntrinsicImage_GS_new(img(:,:,(i-1)*k+1:i*k),3,100,1.9);
% X(:,:,i)=x;
% end
% if floor(bands/k)<bands/k
%    [fimage(:,:,i*k+1:bands) x]=IntrinsicImage_GS_new(img(:,:,i*k+1:bands),3,100,1.9);
%    X(:,:,ceil(bands/k))=x;
% end
% end
% parfor i=1:floor(bands/k)
% [y x]=IntrinsicImage_GS_new(img(:,:,(i-1)*k+1:i*k),3,100,1.9);
% X(:,:,i)=x;
% y=reshape(y,[r,c*k]);
% Y(:,:,i)=y;
% end
% Y=reshape(Y,[r c bands]);