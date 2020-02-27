%{
    Original Author: Stephen Becker, March 2009  srbecker@caltech.edu

    This code has been altered to do matrix completion for the netflix prize data and to compare to the DMF method and others
    Thomas Merkh, tmerkh@ucla.edu, Last Update: 2/12/2020
    Converted this MATLAB compatible code to Octave 5.1.0 Compatible.  Not using mex. 
%}

addpath(fullfile(pwd,'SVD_utilities'));
pkg load statistics
warning('off','SVT:NotUsingMex')
if ispc
    warning('off','PROPACK:NotUsingMex');
endif

clear;
%%%%%%%%%%%%%%%%%%% Parameters you can play with %%%%%%%%%%%%%%%%%%%
%% Load Data:
load('../DMF_code/Netflix_data/X_1000by100.mat') 
X0 = X;           % Original Data
mask_rate = 0.2;  % 0.2 seems like a reasonable amount of data to remove for testing
maxiter = 3000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = size(X,1);
n2 = size(X,2); 

M = ones(n1,n2);
for i = 1:n2
    nulls = randperm(n1, ceil(n1*mask_rate));
    M(nulls,i) = 0;
end
X = X.*M;         % Training Set
Test = X0 - X;    % Test Set
Observed_Training = X > 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[I,J,data] = find(X); % i,j are indices, and v is the value at that non-zero location. 
nonzers = size(data,1);
data = data';
% I is the row index, J is the column.
% Below turns 2D indices --> 1D indices
% Omega should go down first column, then move to the second. 
% [Omega', data'] is basically keys and values
for i = 1:nonzers
  ii = I(i);
  jj = J(i);
  %Omega(i) = n2*(ii-1)+jj;
  Omega(i) = n1*(jj-1)+ii;
endfor
% Percentage of observations
p = nonzers/(n1*n2);

fprintf('Matrix completion: %d x %d matrix, %.1f%% observations\n', n1, n2, 100*p);
    
%%%%%%%%%%%%%%%%%%%%% Set parameters and solve %%%%%%%%%%%%%%%%%%%
tau = 10*sqrt(n1*n2); 
delta = 0.01;     
tol = 1e-4;
%{
tau should probably be bigger than 5*sqrt(n1*n2) in general
increase tau to increase accuracy
delta smaller will give better accuracy
Approximate minimum nuclear norm solution by SVT algorithm
%}

fprintf('\nSolving by SVT...\n');
tic
[U, S, V, numiter] = SVT([n1 n2], Omega, data, tau, delta, maxiter, tol);
toc
    
X_recovered = U*S*V';
for i=1:n1
  for j=1:n2
    X_recovered(i,j) = max([0,min([X_recovered(i,j),5])]);
  endfor
endfor
    
% Show results

fprintf('The recovered rank is %d\n', length(diag(S)));
fprintf('The relative error on Omega (i.e. Training error) is: %d\n', norm(data-X_recovered(Omega))/norm(data))
fprintf('The relative test recovery error is: %d\n', norm((Test - X_recovered).*(Test > 0),'fro')/norm(Test,'fro'))
fprintf('The relative recovery error (i.e. Entire dataset) is: %d\n', norm(X0-X_recovered.*(X0 > 0),'fro')/norm(X0,'fro'))
%fprintf('The relative recovery in the spectral norm is: %d\n', norm(X0-X_recovered)/norm(X0))


return 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Approximate minimum nuclear norm solution by FPC algorithm
%  This version of FPC uses PROPACK for the multiplies
%  It is not optimized, and the parameters have not been tested
%  The version in the FPC paper by Shiqian Ma, Donald Goldfarb and Lifeng Chen
%  uses an approximate SVD that will have different properties;
%  That code may be found at http://www.columbia.edu/~sm2756/FPCA.htm

mu_final = .01; tol = 1e-3;

fprintf('\nSolving by FPC...\n');
tic
[UU,SS,VV,numiter] = FPC([n1 n2],Omega,data,mu_final,maxiter,tol);
toc
   
X_recovered2 = UU*SS*VV';

% Show results
fprintf('The recovered rank is %d\n', length(diag(SS)));
fprintf('The relative error on Omega is: %d\n', norm(data-X_recovered2(Omega))/norm(data))
fprintf('The relative recovery error is: %d\n', norm(X0-X_recovered2.*(X0 > 0),'fro')/norm(X0,'fro'))
%fprintf('The relative recovery in the spectral norm is: %d\n', norm(X0-X_recovered2)/norm(X0))