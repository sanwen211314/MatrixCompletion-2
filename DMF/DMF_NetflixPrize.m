% Thomas Merkh, tmerkh@g.ucla.edu
% This code works depends on .mat files created by a python program Create_Data_Subset.py
% This python code in turn depends on the Netflix Prize dataset, which was provided at https://www.kaggle.com/netflix-inc/netflix-prize-data

% The program at hand reads in the already cleaned data found in 'X_???by???.mat' files
% This incomplete matrix X is all of the data available.  Of interest is predicting the 0 values in this matrix.
% In order to measure performance, this code further splits the known data of X up into a training set and a test set. 

% pkg load statistics
clc
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% User defind parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('./Netflix_data/X_50by50.mat') 
savefile = 'nooo.txt';                  % For saving the output.
% X = X(1:30, 1:30);                              % If a subset of the full dataset is desired.

verbose = false;
r = 10;                                            % Similar to the estimated matrix rank
layer_sizes = [r 50 size(X,1)];                 % Ordered: input, hidden1, ..., last hidden, output (this is m)
options.activation_func = {'tanh_opt','linear'}; % 'tanh_opt','softmax', 'sigm', 'linear'
options.Wp = 0.5;                                % Weight decay penalty
options.Zp = 0.2;                                % Latent variable penalty
options.maxiter = 30000;                         % 5000-30000is usually enough
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X0 = X;    
% Preserve the full data known to us as X0
% X is m by n, where m is the number of movies and n is the number of customers.
m = size(X,1);
n = size(X,2);
mask_rate = 0.2;  % 0.2 seems like a reasonable amount of data to remove for testing

M = ones(m,n);
for i = 1:n
    nulls = randperm(m, ceil(m*mask_rate));
    M(nulls,i) = 0;
end
X = X.*M;  
Test = X0 - X;
Observed_Training = X > 0;
% X is now the training data
% Test=X0-X is the test data
% X0 is the full data set
% Observed_Training tells us which training entries are nonzero

if(verbose && n > 19 && m > 19)
  disp('X(1:20,1:20) is...')
  disp(X(1:20,1:20))
  disp('Test(1:20,1:20) is...')
  disp(Test(1:20,1:20))
endif
tic
% Initialize the Deep Network and Training
[X_predicted, Network_Details] = MC_DMF(X', Observed_Training', layer_sizes, options);
toc
% Frobenious norm here includes the sqrt. 
% Recovery Error between the predictions and the known test values. 
recovery_error = norm((Test - X_predicted').*(Test > 0), 'fro')/norm(Test, 'fro');
% Recovery Error for random guessing:
Randomly_Guessing = randi(5,m,n);
guessing_error = norm((Test - Randomly_Guessing).*(Test > 0), 'fro')/norm(Test, 'fro');
disp(' ')
disp('The relative recovery error for the network is:')
recovery_error % Ranges from (0.37570, 0.38807) for 3 hidden layers, (0.37952, 0.41842, 0.42109, 0.53211) for 2 layers ,(0.40166, 0.44224, 0.44763) for 1 hidden - best for large weight decay param
% It seems like Wp between 0.01 and 0.1 do well.  When there are more layers, smaller weight decay is necessary, for few layers, they prefer large weight decay. 
disp(' ')
disp('The relative recovery error when randomly guessing is:')
guessing_error % Ranges from (0.58564, 0.64005, 0.65575, 0.69539, 0.70951, 0.74138, 0.74411, 0.75948, 0.76554,  0.80911, 0.82598, 0.82712, 0.86823)

if(verbose)
  disp('The residual matrix (abs) for the test set is:')
  disp( abs(Test - round(X_predicted.*(Test > 0) ) ) )
endif

% To do : Create a way to record the parameters of a test, and the results. 
fileID = fopen(savefile,'a');
s = dir(savefile);
if(s.bytes == 0) % check if file is empty
  meta = {'Data = 50 by 50',' maxiter = 30000'};
  fprintf(fileID,'%s,%s\n', meta{:});
  fields = {'Recovery_error',' Wp',' Zp',' Layer Sizes',' Layer Activations'};
  fprintf(fileID,'%s,%s,%s,%s,%s\n', fields{:});
endif
formatSpec = '%0.3f, %0.3f, %0.3f'; 
fields = {recovery_error, options.Wp, options.Zp};
fprintf(fileID, formatSpec, fields{:});

for ii = 1:size(layer_sizes,2)
    if(ii == 1)
      fprintf(fileID,', [%i',layer_sizes(ii));
    elseif(ii == size(layer_sizes,2))
      fprintf(fileID,';%i]',layer_sizes(ii));
    else
      fprintf(fileID,';%i',layer_sizes(ii));
    endif
endfor

if(size(layer_sizes,2) == 3)
  fields = {options.activation_func{1},options.activation_func{2}};
  fprintf(fileID, ', [%s;%s]\n', fields{:});  
elseif(size(layer_sizes,2) == 4)
  fields = {options.activation_func{1},options.activation_func{2},options.activation_func{3}};
  fprintf(fileID, ', [%s;%s;%s]\n', fields{:});  
elseif(size(layer_sizes,2) == 5)
  fields = {options.activation_func{1},options.activation_func{2},options.activation_func{3},options.activation_func{4}};
  fprintf(fileID, ', [%s;%s;%s;%s]\n', fields{:});  
elseif(size(layer_sizes,2) == 6)
  fields = {options.activation_func{1},options.activation_func{2},options.activation_func{3},options.activation_func{4},options.activation_func{5}};
  fprintf(fileID, ', [%s;%s;%s;%s;%s]\n', fields{:});  
endif
%fprintf(fileID, '\n');  
fclose(fileID);