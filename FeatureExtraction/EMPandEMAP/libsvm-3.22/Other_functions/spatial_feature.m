function [fimage] = spatial_feature( img,r,eps)
%SPATIAL_FEATURE Summary of this function goes here
%   Detailed explanation goes here
bands=size(img,3);
for i=1:bands
    fimage(:,:,i)=RF(img(:,:,i),r,eps,3,img(:,:,i));
   
end
