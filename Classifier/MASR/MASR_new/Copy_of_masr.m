    function [OA,AA,Ka,PA ,result,T] = Copy_of_masr(oil,GT,Tr,Te)

% Define the size of each scale
Multi_scale_Patchsize = [3 5 7 9 11 13 15 ];
scale_num = length(Multi_scale_Patchsize);


Tr=Tr(:);
Te=Te(:);
% 
% Tr(find(Tr==0))=[];
% Te(find(Te==0))=[];

Multi_scale_nblk = [];
Multi_scale_PS_H = [];
for i = 1:scale_num
    Multi_scale_nblk(i) = Multi_scale_Patchsize(i)*Multi_scale_Patchsize(i);
    Multi_scale_PS_H(i) = floor(Multi_scale_Patchsize(i)/2);
end
multiscalemap = 1: Multi_scale_Patchsize(end)*Multi_scale_Patchsize(end);
multiscalemap = reshape(multiscalemap,[Multi_scale_Patchsize(end) Multi_scale_Patchsize(end) ]);
xx = Multi_scale_PS_H(end)+1;
yy = Multi_scale_PS_H(end)+1;

%Gerate the neighbors for each scale
scale_index = {};
for ss = 1:scale_num
    index_temp = multiscalemap((xx-Multi_scale_PS_H(ss)):(xx+Multi_scale_PS_H(ss)),(yy-Multi_scale_PS_H(ss)):(yy+Multi_scale_PS_H(ss)));
    index_vec = index_temp(:)';
    scale_index{ss} = index_vec;
end

%Number of class
no_classes       = 3;
%Number of training samples
%no_train = 1043;
%Load Indian Pines dataset
% load Indian_pines_corrected
% im = indian_pines_corrected;
im=oil;

Multi_scale_im_Ex = {};
Multi_scale_im_Ex = padarray(im,[Multi_scale_PS_H(end) Multi_scale_PS_H(end)],'symmetric'  );

[I_row,I_line,I_high] = size(im);
im1 = reshape(im,[I_row*I_line,I_high]);
im1 = im1';

%load groundtruth
% load Indian_pines_gt
indian_pines_gt=GT;
K = no_classes;
% OA_class=[];
% kappa_class=[];
% AA_class=[];
% CA_class=[];
%for k=1:1
%    load (strcat('IndianP_',num2str(k)))

Train_Label = [];
Train_index = [];
for ii = 1: no_classes
    index_ii =  find(indian_pines_gt == ii);
    class_ii = ones(length(index_ii),1)* ii;
    Train_Label = [Train_Label class_ii'];
    Train_index = [Train_index index_ii'];
end
% Train_Label = uint16(Train_Label );
% Train_index = uint16(Train_index);

K = max(Train_Label);
% Select the number of training samples for each class
% RandSampled_Num = [5 143 83 24 48 73 3 48 2 97 246 59 21 127 39 9];
tr_lab = [];
tt_lab = [];
tr_dat = [];

% Create the Training and Testing set with randomly sampling 3-D Dataset and its correponding index
Index_train = {};
Index_test = {};
for i = 1: K
%     W_Class_Index = find(Train_Label == i);
%     Random_num = randperm(length(W_Class_Index));
%     Random_Index = W_Class_Index(Random_num);
%     Tr_Index = Random_Index(1:RandSampled_Num(i));
    AAA=(find(Tr==i))';
    [~,Tr_Index] = ismember(AAA,Train_index);
        Tr_Index(find(Tr_Index==0))=[];
    
    Index_train{i} = Train_index(Tr_Index);
    
%     Tt_Index{i} = Random_Index(RandSampled_Num(i)+1 :end);
    BBB=(find(Te==i))';
    [~,Tt_Index{i}] =ismember(BBB,Train_index);
    Tt_Index{i}(find(Tt_Index{i}==0))=[];

    Index_test{i} = Train_index(Tt_Index{i});
    
%     tr_ltemp = ones(RandSampled_Num(i),1)'* i;
    tr_ltemp = ones(length( Tr_Index ),1)'* i;
    
    tr_lab = [tr_lab tr_ltemp];
    tr_Class_DAT = im1(:,Train_index(Tr_Index));
%     tr_Class_DAT = im1(:,Tr_Index);
    tr_dat = cat(2,tr_dat,tr_Class_DAT);
end

% Normalizing the training data with 2-norm
tr_dat        =    tr_dat./ repmat(sqrt(sum(tr_dat.*tr_dat)),[size(tr_dat,1) 1]);

classids       =    unique(tr_lab);
NumClass       =    length(classids);
tt_ID          =    [];
tic
for i = 1: K
    tt_ltemp = ones(length(Tt_Index{i}),1)'* i;
    tt_lab = [tt_lab tt_ltemp];
    tt_dat = {};
    tt_ID_temp = [];
    
    for j = 1:length(Tt_Index{i})
        X = mod(Train_index(Tt_Index{i}(j)),I_row);
        Y = ceil(Train_index(Tt_Index{i}(j))/I_row);
        X_new = X+Multi_scale_PS_H(end);
        Y_new = Y+Multi_scale_PS_H(end);
        X_range = [X_new-Multi_scale_PS_H(end) : X_new+Multi_scale_PS_H(end)];
        Y_range = [Y_new-Multi_scale_PS_H(end) : Y_new+Multi_scale_PS_H(end)];
        if(ismember(0,X_range))
            X_range(find(X_range==0))=1;
        end
        tt_Class_DAT_temp = Multi_scale_im_Ex( X_range , Y_range ,:);
        [r,l,h]=size(tt_Class_DAT_temp);
        tt_Class_DAT_temp = reshape(tt_Class_DAT_temp,[r*l,h]);
        tt_Class_DAT = tt_Class_DAT_temp';
        tt_Class_DAT =  tt_Class_DAT./ repmat(sqrt(sum(tt_Class_DAT.*tt_Class_DAT)),[size(tt_Class_DAT,1) 1]);
        
        for si = 1: scale_num
            tt_dat{si} = tt_Class_DAT(:,scale_index{si});
        end
        %%   Calculate the multiscale sparse matix
        Multiscale_s = Multiscale_SOMP(tr_dat,tt_dat,2, scale_num,tr_lab );
        %%   Determined the label of a test sample
        tt_ID1 = Multiscale_label_new(NumClass, scale_num,Multiscale_s , tr_dat, tt_dat, classids, tr_lab);
        tt_ID_temp  =  [tt_ID_temp tt_ID1];
    end
    tt_ID = [tt_ID tt_ID_temp];
end

%%  Calculate the error based on predict label and truth label of testing samples%
%[OA_class,kappa_class,AA_class,CA_class] = calcError( tt_lab-1, tt_ID-1, 1: NumClass);
[OA,Ka,AA,PA] = confusion(tt_lab,tt_ID);


%end

T=toc;

resultmap = zeros(I_row,I_line);
% Testing set mapping
id = 1;
%for i = 1: size(Index_test,2)
for i=1:K
    for j = 1: length(Index_test{i})
        if mod(Index_test{i}(j),I_row) == 0
            
            X = I_row;
            Y = ceil(Index_test{i}(j)/I_row);
        else
            X = mod(Index_test{i}(j),I_row);
            Y = ceil(Index_test{i}(j)/I_row);
        end
        
        resultmap(X,Y) = tt_ID(id);
        id = id+1;
    end
end
% Training set mapping
id = 1;
%for i = 1: size(Index_train,2)
% % % for i=1:K
% % %     for j = 1: length(Index_train{i})
% % %         
% % %         if mod(Index_train{i}(j),I_row) == 0
% % %             
% % %             X = I_row;
% % %             Y = ceil(Index_train{i}(j)/I_row);
% % %         else
% % %             X = mod(Index_train{i}(j),I_row);
% % %             Y = ceil(Index_train{i}(j)/I_row);
% % %         end
% % % 
% % %         resultmap(X,Y) = i;
% % %         id = id+1;
% % %     end
% % % end
result=resultmap;
% 
% [xx,yy]=size(GT);
% result=imresize(tt_ID,[xx,yy]);
% MASRmap=label2color(resultmap,'india');
% imwrite(MASRmap,'MASR.tif');
% figure,imshow(MASRmap);

