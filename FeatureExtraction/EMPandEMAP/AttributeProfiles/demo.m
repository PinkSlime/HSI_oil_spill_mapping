
clc,clear
tic
no_classes       = 16;
no_train         = round(1083);
path='.\Dataset\';
inputs = 'IndiaP';%145*145*200/10249/16
% inputs = 'PaviaU';%610*340*115/42776/9
% inputs = 'Salinas';%512*217*224/54129/16
location = [path,inputs];
load (location);


%%% size of image 
[no_lines, no_rows, no_bands] = size(img);
GroundT=GroundT';


indexes=train_test_random_new(GroundT(2,:),fix(no_train/no_classes),no_train);
fimg = NormalizeData(img);

b1=fimg(:,:,1);
r=EMAP(b1,'',false, '', 'a', 25);