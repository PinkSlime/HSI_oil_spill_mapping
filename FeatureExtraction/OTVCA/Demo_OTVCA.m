clear all
close all



load (['oil.mat'  ]);                   %%load HSI data

load (['GT.mat'  ]);                    %%load GT

load (['Tr.mat'  ]);                   %%load Train set





Te=GT-Tr;
GT=uint8(GT);
Tr=uint8(Tr);
Te=uint8(Te);


%% size of image
[no_lines, no_rows, no_bands] = size(img);
dim=max(unique(Te)); % Number of Features is set to the number of classes
Trees = 200;

%time
T = 0;
tic;
%% compute the OTVCA features
%%
[acc_Mean , CM] = USFE_OTVCA(img, Tr, Te, dim, Trees);
T = T + toc;
PA = acc_Mean(1:dim,1);
OA = acc_Mean(dim+2,1);
K = acc_Mean(dim+3,1);
AA = mean(PA);

%%
CM=uint8(CM );






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



