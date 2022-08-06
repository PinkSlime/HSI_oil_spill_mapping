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
Te=GT-Tr;

%time
T = 0;
tic;
%% compute the EMAP features
%%
oil=img;
 
[no_lines, no_rows, no_bands] = size(img);
GT=GT;
Tr=Tr;
Te=Te;
[ OA,AA,K,PA ,result ] = ERW_OIL(oil,GT,Tr,Te);

T = T + toc;

%%
CM=uint8(result );
CM(find(GT==0))=0;
 

 
color_map=[
    0,0,0;
    255,255,255;
    0,0,0;
    163,163,163;
    224,224,224;
    ];
 
color_map=color_map/255;
out_put=[AA,OA,K,PA',T];

gt=reshape(GT,1,no_lines*no_rows);
cm=reshape(CM,1,no_lines*no_rows);
[aa oa ua pa K confu]=new_confusion(gt,cm);


%         xlswrite('D:\oil\add\PengLai\ploil_data\add.xlsx',out_put , 'Sheet1',['B23']  )
%         imwrite(CM ,color_map, ['D:\oil\add\PengLai\ploil_data\ERW',  '.png'])
%

