%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
% This is a demo for the LWEA and LWGP algorithms. If you find this %
% code useful for your research, please cite the paper below.       %
%                                                                   %
% Dong Huang, Chang-Dong Wang, and Jian-Huang Lai.                  %
% "Locally weighted ensemble clustering."                           %
% IEEE Transactions on Cybernetics, accepted, 2017.                 %
% DOI: 10.1109/TCYB.2017.2702343                                    %
%                                                                   %
% The code has been tested in Matlab R2014a and Matlab R2015a on a  %
% workstation with Windows Server 2008 R2 64-bit.                   %
%                                                                   %
% https://www.researchgate.net/publication/316681928                %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demo_LWEA_and_LWGP()
%% A demo for the LWEA and LWGP algorithms.

clear all;
close all;
clc;


%% Load the base clustering pool.
% We have generated a pool of 100 candidate base clusterings for each dataset.

% Please uncomment the dataset that you want to use and comment the other ones.
% dataName = 'VS';
dataName = 'Semeion';
% dataName = 'SPF';
% dataName = 'MF';
% dataName = 'IS';
% dataName = 'Caltech20';
% dataName = 'FCT';
% dataName = 'MNIST';
% dataName = 'Texture';
% dataName = 'ODR';
% dataName = 'LS';
% dataName = 'ISOLET';
% dataName = 'PD';
% dataName = 'USPS';
% dataName = 'LR';

members = [];
gt = [];
load(fullfile('..','data',['baseClsPool_',dataName,'.mat']),'members','gt');

[N, poolSize] = size(members);

%% Parameter
para_theta = 0.4;

%% Settings
% Ensemble size M
M = 10;
% How many times the LWEA and LWGP algorithms will be run.
cntTimes = 20;
% You can set cntTimes to a greater (or smaller) integer if you want to run
% the algorithms more (or less) times.

% For each run, M base clusterings will be randomly drawn from the pool.
% Each row in bcIdx corresponds to an ensemble of M base clusterings.
bcIdx = zeros(cntTimes, M);
for i = 1:cntTimes
    tmp = randperm(poolSize);
    bcIdx(i,:) = tmp(1:M);
end

%% Run LWEA and LWGP repeatedly.
% The numbers of clusters.
clsNums = [2:30];
% Scores
outDir = fullfile('..','results');
mkdir(outDir);
nmiScoresBestK_LWEA = zeros(cntTimes, 1);
nmiScoresTrueK_LWEA = zeros(cntTimes, 1);
nmiScoresBestK_LWGP = zeros(cntTimes, 1);
nmiScoresTrueK_LWGP = zeros(cntTimes, 1);
for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    
    %% Construct the ensemble of M base clusterings
    % baseCls is an N x M matrix, each row being a base clustering.
    baseCls = members(:,bcIdx(runIdx,:));
    
    %% Get all clusters in the ensemble
    [bcs, baseClsSegs] = getAllSegs(baseCls);
    
    %% Compute ECI
    disp('Compute ECI ... '); 
    tic; 
    ECI = computeECI(bcs, baseClsSegs, para_theta);
    toc;
    
    %% Compute LWCA
    LWCA= computeLWCA(baseClsSegs, ECI, M);
    
    %% Perform LWGP 
    disp('Run the LWGP algorithm ... '); 
    resultsLWGP = runLWGP(bcs, baseClsSegs, ECI, clsNums);     
    disp('--------------------------------------------------------------'); 

    %% Perform LWEA
    disp('Run the LWEA algorithm ... '); 
    resultsLWEA = runLWEA(LWCA, clsNums);
    % The i-th column in resultsLWEA represents the consensus clustering 
    % with clsNums(i) clusters by LWEA.
    disp('--------------------------------------------------------------');
    
    
    %% Display the clustering results.    
    disp('##############################################################'); 
    scoresLWGP = computeNMI(resultsLWGP,gt);
    scoresLWEA = computeNMI(resultsLWEA,gt);
    
    nmiScoresBestK_LWEA(runIdx) = max(scoresLWEA);
    trueK = numel(unique(gt));
    nmiScoresTrueK_LWEA(runIdx) = scoresLWEA(clsNums==trueK);
    
    nmiScoresBestK_LWGP(runIdx) = max(scoresLWGP);
    nmiScoresTrueK_LWGP(runIdx) = scoresLWGP(clsNums==trueK);
    
    disp(['The Scores at Run ',num2str(runIdx)]);
    disp('    ---------- The NMI scores w.r.t. best-k: ----------    ');
    disp(['LWGP : ',num2str(nmiScoresBestK_LWGP(runIdx))]);
    disp(['LWEA : ',num2str(nmiScoresBestK_LWEA(runIdx))]);
    
    disp('    ---------- The NMI scores w.r.t. true-k: ----------    ');
    disp(['LWGP : ',num2str(nmiScoresTrueK_LWGP(runIdx))]);
    disp(['LWEA : ',num2str(nmiScoresTrueK_LWEA(runIdx))]);
    
    disp('##############################################################'); 
    
    %% Save results
    save(fullfile(outDir,['results_',dataName,'.mat']),'bcIdx','nmiScoresBestK_LWEA','nmiScoresTrueK_LWEA','nmiScoresBestK_LWGP','nmiScoresTrueK_LWGP');  
end

disp('**************************************************************');
disp(['** Average Performance over ',num2str(cntTimes),' runs on the ',dataName,' dataset **']);
disp(['Data size: ', num2str(N)]);
disp(['Ensemble size: ', num2str(M)]);
disp('   ---------- Average NMI scores w.r.t. best-k: ----------   ');
disp(['LWGP   : ',num2str(mean(nmiScoresBestK_LWGP))]);
disp(['LWEA   : ',num2str(mean(nmiScoresBestK_LWEA))]);
disp('   ---------- Average NMI scores w.r.t. true-k: ----------   ');
disp(['LWGP   : ',num2str(mean(nmiScoresTrueK_LWGP))]);
disp(['LWEA   : ',num2str(mean(nmiScoresTrueK_LWEA))]);
disp('**************************************************************');
disp('**************************************************************');
