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

function labels = runLWGP(bcs,baseClsSegs, ECI, clsNum)

lwB = getBipartiteGraph_LWBGP(baseClsSegs, ECI);
labels = zeros(size(bcs,1),numel(clsNum));
for i = 1:numel(clsNum); % clustering number. 
    disp(['Obtain ',num2str(clsNum(i)),' clusters by LWGP.']); tic;
    labels(:,i) = Tcut_for_clustering_ensemble_New_I(lwB,clsNum(i));toc;
end 

function B = getBipartiteGraph_LWBGP(bcsSegs, ECI)

[nCls,n] = size(bcsSegs);

W_Y = eye(nCls);

W_XY = bcsSegs';
W_XY = W_XY.*repmat(ECI',n,1);

B = [W_XY;W_Y];



function labels = Tcut_for_clustering_ensemble_New_I(B,Nseg)

% B - |X|-by-|Y|, cross-affinity-matrix
% note that |X| = |Y| + |I|

[Nx,Ny] = size(B);
if Ny < Nseg
    error('Need more superpixels!');
end

%%% build the superpixel graph
dx = sum(B,2);
Dx = sparse(1:Nx,1:Nx,1./dx);
Wy = B'*Dx*B;

%%% compute Ncut eigenvectors
% normalized affinity matrix
d = sum(Wy,2);
D = sparse(1:Ny,1:Ny,1./sqrt(d));
nWy = D*Wy*D;
nWy = (nWy+nWy')/2;

% computer eigenvectors
[evec,eval] = eig(full(nWy)); % use eigs for large superpixel graphs  
[~,idx] = sort(diag(eval),'descend');
Ncut_evec = D*evec(:,idx(1:Nseg));

%%% compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
evec = Dx * B * Ncut_evec;

%%% k-means clustering
% extract spectral representations for pixels
% evec = evec(1:prod(img_size),:);
evec = evec(1:(Nx-Ny),:);

% normalize each row to unit norm
evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );

% k-means
labels = k_means_New_I(evec',Nseg);



function [R, M] = k_means_New_I(X, K, seed)
%KMEANS:  K-means clustering
%  idx = KMEANS(X, K) returns M with K columns, one for each mean.  Each
%      column of X is a datapoint.  K is the number of clusters
%  [idx, mu] = KMEANS(X, K) also returns mu, a row vector, R(i) is the
%      index of the cluster datapoint X(:, i) is assigned to.
%  idx = KMEANS(X,K) returns idx where idx(i) is the index of the cluster
%      that datapoint X(:,i) is assigned to.
%  [idx,mu] = KMEANS(X,K) also returns mu, the K cluster centers.
%
%  KMEANS(X, K, SEED) uses SEED (default 1) to randomise initial assignments.

if ~exist('seed', 'var'), seed = 1; end

%
%  Initialization
%
[D,N] = size(X);
% if D>N, warning(sprintf('K-means running on %d points in %d dimensions\n',N,D)); end;

M = zeros(D, K);
Dist = zeros(N, K);
M(:, 1) = X(:,seed);
Dist(:, 1) = sum((X - repmat(M(:, 1), 1, N)).^2, 1)';
for ii = 2:K
  % maximum, minimum dist
  mindist = min(Dist(:,1:ii-1), [], 2);
  [junk, jj] = max(mindist);
  M(:, ii) = X(:, jj);
  Dist(:, ii) = sum((X - repmat(M(:, ii), 1, N)).^2, 1)';
end

% plotfig(X,M);
X2 = sum(X.^2,1)';
converged = 0;
R = zeros(N, 1);

cnt = 1;
while (~converged) && cnt<200
    cnt = cnt+1;
  distance = repmat(X2,1,K) - 2 * X' * M + repmat(sum(M.^2, 1), N, 1);
  [junk, newR] = min(distance, [], 2);
  if norm(R-newR) == 0
    converged = 1;
  else
    R = newR;
  end
  total = 0;
  for ii = 1:K
    ix = find(R == ii);
    M(:, ii) = mean(X(:, ix), 2);
    total = total + sum(distance(ix, ii));
  end
% plotfig(X,M);
%   fprintf('Distance %f\n', total);
end
% cnt
% pause; close all;
return

function plotfig(x,M),
	figure; plot(x(1,:),x(2,:),'go', 'MarkerFaceColor','g', 'LineWidth',1.5); hold on; plot(M(1,:),M(2,:),'rx','MarkerSize',12, 'LineWidth',2);
	w = 2.15; h = 2;
	for k=1:size(M,2),
		rectangle('Position',[M(1,k) M(2,k) 0 0]+w*[-1 -1 +2 +2], 'Curvature',[1 1], 'EdgeColor','r', 'LineWidth',2);
	end;
	xlim([floor(min(x(1,:))) ceil(max(x(1,:)))]);
	ylim([floor(min(x(2,:))) ceil(max(x(2,:)))]);
return



    