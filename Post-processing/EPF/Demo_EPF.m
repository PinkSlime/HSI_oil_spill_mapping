
%%
clear all
close all


load oil.mat
%
load(['GT.mat'          ]);
%
load(['TR.mat'       ]);
%
GT=uint8(GT );
Tr=   TR;
[no_lines, no_rows, no_bands] = size(img);
Te=GT-Tr;
%time
T = 0;
tic;
%% compute the EMAP features
%%
oil=img;
[ OA,AA,K,PA ,result ] = wzh_EPF(oil,GT,Tr,Te);

T = T + toc;

%%
CM=uint8(result );

color_map=[
    0,0,0;
    255,255,255;
    0,0,0;
    163,163,163;
    224,224,224;
    ];

color_map=color_map/255;

gt=reshape(GT,1,no_lines*no_rows);
cm=reshape(CM,1,no_lines*no_rows);
[aa oa ua pa K confu]=new_confusion(gt,cm);




