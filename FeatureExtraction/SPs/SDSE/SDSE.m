function [ out,fimg1 ] = SDSE( img )
 img2=average_fusion(img,35);
% img2=img;
 [no_lines, no_rows, no_bands] = size(img);
%%% normalization

no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
fimg1 = satv(fimg,1.6,[],[],'L2');
% out=fun_MyMNF(fimg1,16);
% out =kpca(fimg1, 2000,25, 'Gaussian',40);
out =kpca(fimg1, 1000,3, 'Gaussian',40);
end

