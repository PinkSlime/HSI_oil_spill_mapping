function [acc] = ComputeClassificationAccuracy(varargin)
%  ComputeClassificationAccuracy
%  
%  [Kappa OverallAccuracy AveAccuracy Kvar Zstatistic UserAccuracy ProducerAccuracy ConfusionMatrix] =
%  ComputeClassificationAccuracy(ClassMap, RefMap)
%  
%  Input
%      - 1 input -
%      Confusion Matrix in number of samples. If the confusion matrix is
%      available in percentage and the number of samples of the reference
%      map are also available see at the bottom of this page for the
%      conversion routine.
%
%      - 2 input -
%      ClassMap - Classification map to be evaluated (the value of each
%      pixel refers to the class where the pixel belongs to - values equal
%      to zero are not allowed).
%      RefMap - Reference map used for computing the accuracy (pixels
%      having value 0 are considered unclassified and thus the accuracy
%      will be not conputed on that pixels).
%  
%  Output
%      Kappa - Kappa accuracy
%      OverallAccuracy - Overall Accuracy
%      AveAccuracy - Average Accuracy
%      UserAccuracy - User Accuracy (array)
%      ProducerAccuracy - Producer Accuracy (array)
%      ConfusionMatrix - Confusion Matrix
%      Kvar - Estimation of the Variance of Kappa 
%      Zstatistic - Z statistic (test for assessing the significance of the
%      confusion matrix).
%  
%   Function that computes the Kappa Accuracy, Overall Accuracy,
%   User Accuracy, Producer Accuracy and Confusion Matrix of the classification map 
%   ClassMap taking as reference RefMap. The variance of Kappa and Z
%   statistic is computed.
%  
%           WARNING:
%           Unclass: pixel value ->  0
%           Class i: pixel value ->  i
%  
%   Reference: [Congalton, Green "Assessing the Accuracy of Remotely Sensed
%   Data: Principles and Practices", Chapt. 5]
%
%  -----------------
%  Mauro Dalla Mura
%  28 Nov 2007


switch (nargin)
    case 1
        ConfusionMatrix = varargin{1};
        nclasses = size(ConfusionMatrix,1);
    case 2
        ClassMap = varargin{1};
        RefMap = varargin{2};

        % Check inputs
        if range(RefMap(:)) == 0
            error('The reference map has range 0');
        end

        [row col] = size(RefMap);
        %nclasses = range(RefMap(:));        % number of classes

        if islogical(RefMap)    % the value 0 is considered the label of a class
            RefMap = double(RefMap) + 1;
            ClassMap = double(ClassMap) + 1;
        end

        % handle the case of non sequential class labels (e.g., 1,2,4...)
        % non consider zero as a label.
        lab_m = unique(ClassMap);
        lab_t = unique(RefMap);   
        lab_m = lab_m(lab_m~=0);
        lab_t = lab_t(lab_t~=0);
        
        % assign new sequential labels (which will be the indexes of the
        % confusion matrix) to the real labels.
        new_lab_m = zeros(1,max(lab_m));
        new_lab_m(lab_m) = 1:length(lab_m);
        new_lab_t = zeros(1,max(lab_t));
        new_lab_t(lab_t) = 1:length(lab_t);
        
        
        nclasses = length(unique([lab_m;lab_t]));   % take all the labels
        ConfusionMatrix = zeros(nclasses);  % preallocation of the confusion matrix

        % Compute the Confusion Matrix (rows refers to pixels)
        for i=1:row
            for j=1:col
                if (RefMap(i,j)~=0 && ClassMap(i,j) ~= 0)   % neither of the values of the two maps have to be zero.
                    ConfusionMatrix(new_lab_m(ClassMap(i,j)),new_lab_t(RefMap(i,j))) = ConfusionMatrix(new_lab_m(ClassMap(i,j)),new_lab_t(RefMap(i,j)))+1;
                end
            end
        end
        
        acc.ConfusionMatrix = ConfusionMatrix;
end


% trial
% nclasses = 4;
% ConfusionMatrix = [45     4    12    24
%      6    91     5     8
%      0     8    55     9
%      4     7     3    55];


r = sum(ConfusionMatrix,2);       % sum the rows of the confusion matrix
c = sum(ConfusionMatrix,1);       % sum the cols of the confusion matrix
UserAccuracy = zeros([1 nclasses]);
ProducerAccuracy = zeros([1 nclasses]);

for i=1:nclasses  
    UserAccuracy(i) = ConfusionMatrix(i,i)*100/r(i);
    ProducerAccuracy(i) = ConfusionMatrix(i,i)*100/c(i);
end

AveAccuracy = sum(ProducerAccuracy)/nclasses;


a = sum(diag(ConfusionMatrix));
b = c*r;
n = sum(r);
a1 = a/n;
b1 = b/n^2;

OverallAccuracy = a1*100;
Kappa = (a1 - b1)/(1 - b1);

% Compute the variance of Kappa by using the Delta method
t1 = a1;
t2 = b1;
t3 = (diag(ConfusionMatrix)'*(r+c'))/n^2;
t4 = 0;
for i=1:nclasses
    for j=1:nclasses
        t4 = t4 + ConfusionMatrix(i,j)*(r(j)+c(i))^2;       % check correct form
        %t4 = t4 + ConfusionMatrix(i,j)*(r(i)+c(j))^2;      
    end
end
t4 = t4/n^3;

Kvar = (1/n)*( t1*(1-t1)/(1-t2)^2 + 2*(1-t1)*(2*t1*t2-t3)/(1-t2)^3 + (t4-4*t2^2)*(1-t1)^2/(1-t2)^4);
Zstatistic = Kappa/sqrt(Kvar);

acc.UserAccuracy = UserAccuracy;
acc.ProducerAccuracy = ProducerAccuracy;
acc.AveAccuracy = AveAccuracy;
acc.OverallAccuracy = OverallAccuracy;
acc.Kappa = Kappa;
acc.Kvar = Kvar;
acc.Zstatistic = Zstatistic;

% -------------------------------------------------------------------------
% % Routine for converting a confusion matrix expressed in percentage to a
% % matrix expressed in number of samples. The number of samples per class of
% % the reference map has to be available.
% ConfusionMatrix_samp = ConfusionMatrix;
% for i=1:nclasses
%     ConfusionMatrix_samp(i,:) = round(ConfusionMatrix(i,:)*refSamples(i)/100);
% end