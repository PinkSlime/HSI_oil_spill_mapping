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

%%
[no_lines, no_rows, no_bands] = size(img);
%%
%         img=oil;
img=Normalization(img);
[no_lines, no_rows, no_bands] = size(img);
test_SL=matricetotwo(Te);
test_labels = test_SL(2,:)';
[Pre_re Pre_pro ] = pre_classification( img,Tr,Te );
[ Pos_re Pos_pro] = post_classification( img,Tr,Te );
t=0.5;
Fuse_pro=t.*Pre_pro+(1-t).*Pos_pro;
[Class_pro,Fuse_Result]=max(Fuse_pro,[],3);
Result=reshape(Fuse_Result,[no_lines*no_rows 1]);
T=toc
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);

ResultPre = Pre_re(test_SL(1,:),:);
ResultPos = Pos_re(test_SL(1,:),:);
% ResultTest1 = Sresult(test_SL(1,:),:);
[Pre_OA,Pre_AA,Pre_Kappa,Pre_CA]=confusion(GroudTest,ResultPre);

[Pos_OA,Pos_AA,Pos_Kappa,Pos_CA]=confusion(GroudTest,ResultPos);
[OA,AA,K,PA]=confusion(GroudTest,ResultTest);

Result = reshape(Result,no_lines,no_rows);
CM=Result;


%         CM=uint8(result );
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

