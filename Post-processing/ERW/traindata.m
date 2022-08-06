
load IndiaP.mat;
trainall =GroundT';
number_train=118;
% randomly select the training set
for i=1:10
indexes=train_test_random_new(trainall(2,:),round(number_train/16),number_train);  
tt = 'IndiaP_118_';
location_tt = [tt,num2str(i)];
save ([location_tt], 'indexes') ;
end