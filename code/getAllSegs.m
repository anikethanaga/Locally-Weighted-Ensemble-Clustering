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

function [bcs, baseClsSegs] = getAllSegs(baseCls)
%% Get all clusters in the ensemble

[N,nBC] = size(baseCls);
% n:    the number of data points.
% nBase:    the number of base clusterings.
% nCls:     the number of clusters (in all base clusterings).


bcs = baseCls;
nClsOrig = max(bcs,[],1);
C = cumsum(nClsOrig); 
bcs = bsxfun(@plus, bcs,[0 C(1:end-1)]);
nCls = nClsOrig(end)+C(end-1);
baseClsSegs = zeros(nCls,N);

for i=1:nBC 
    if i == 1
        startK = 1;
    else
        startK = (C(i-1)+1);
    end
    endK = C(i);
    searchVec = startK:endK;
    F = bsxfun(@eq,bcs(:,i),searchVec);
    baseClsSegs(searchVec,:) = F';
end






