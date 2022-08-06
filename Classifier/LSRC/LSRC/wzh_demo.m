clear all
close all

load (['oil.mat'  ]);                   %%load HSI data

load (['GT.mat'  ]);                    %%load GT

load (['Tr.mat'  ]);                   %%load Train set

Te=GT-Tr;
GT=uint8(GT);
Tr=uint8(Tr);
Te=uint8(Te);
%

 

%time
T = 0;
tic;
%% compute the EMAP features
%%
[no_lines, no_rows, no_bands] = size(img);
%
oil=img;

%time
%         T = 0;
%         tic;
%% compute the LSRC features
%%

[ OA,AA,K,PA ,result,T ] = wzh_LSRC(oil,GT,Tr,Te);

T = T + toc;

%%
CM=uint8(result );



color_map=[
    0,0,0;
    95,69,255;
    
    172,165,199;
    255,243,219;
    
    ];
color_map=color_map/255;

gt=reshape(GT,1,no_lines*no_rows);
cm=reshape(CM,1,no_lines*no_rows);

[aa oa ua pa K confu]=new_confusion(gt,cm);
out_put=[aa,oa,K,pa'];
