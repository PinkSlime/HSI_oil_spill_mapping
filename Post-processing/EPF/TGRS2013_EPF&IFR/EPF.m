function [ ClasMap ] = EPF(n,m,img,SVMMap)
% n = 1, 2 or 3 1:Bilateral filter based EPF;2:IC filter based EPF 3:Guided
% filter based EPF
% m = 1 or 2 1:one band of PCA 2:three bands of PCA
[r,c]=size(SVMMap);
bands=size(img,2);
img=reshape(img,[r,c,bands]);
switch m
    case 1
    GDimg = GDConA(img);
    ClasMap = CostF(SVMMap, GDimg,n);
    case 2
    GDimg = GDConB(img);
    ClasMap = CostFC(SVMMap, GDimg,n);
end
function [GDimg] = GDConB(img)
[r,c,b]=size(img);
x=reshape(img,[r*c b]);
x=compute_mapping(x,'PCA',3);
x=mat2gray(x);
x=reshape(x,[r,c,3]);
GDimg=x;
function [GDimg] = GDConA(img)
[r,c,b]=size(img);
x=reshape(img,[r*c b]);
x=compute_mapping(x,'PCA',1);
x = mat2gray(x);
x =reshape(x,[r,c,1]);
GDimg=x;

function [ClassMap] = CostF(SVMMap, GDimg,n)
L = max(SVMMap(:));
for i=1:L
    p = zeros(size(SVMMap));
    p(SVMMap==i) = 1;
    switch n
    case 1
    c(:,:,i) = bilateralFilter(p,GDimg,0,1,3,0.1);
    case 2
    c(:,:,i) = IC(p,3,0.1,3,GDimg); 
    case 3
    c(:,:,i) = guidedfilter(GDimg,p,3,0.1^3); 
    end
end
[unused,ClassMap] = max(c,[],3);
function [ClassMap] = CostFC( SVMMap, GDimg,n)
L = max(SVMMap(:));
for i=1:L
    p = zeros(size(SVMMap));
    p(SVMMap==i) = 1;
    switch n
    case 1
    c(:,:,i) = jbfilter2(p,GDimg,4,[4 0.1]);
    case 2
    c(:,:,i) = IC(p,3,0.1,3,GDimg); 
    case 3
    c(:,:,i) = guidedfilter_color(GDimg,p,4,0.1^3);
    end
end
[unused,ClassMap] = max(c,[],3);