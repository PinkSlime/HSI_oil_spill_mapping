clear all
close all



load (['oil.mat'  ]);                   %%load HSI data

load (['GT.mat'  ]);                    %%load GT

Tr = imread(['Tr.tif'  ]);              %%load Train set


Te=GT-Tr;
GT=uint8(GT);
Tr=uint8(Tr);
Te=uint8(Te);
[no_lines, no_rows, no_bands] = size(img);
%time
T = 0;
tic;
dim=max(unique(Te));
%% compute the PCA features
%%
[ OA,AA,K,PA ,result ] = USFE_PCA(img,GT, Tr, Te, dim);

T = T + toc;
%%
CM=uint8(result );



save ( [ 'PCA_',   num2str(oil_index), '.mat'] , 'CM'  )


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


