clc;
clear;
close all;
load('/home/sunli/4T/yanzi/now/data/abu-urban-4/abu-urban-4.mat');
% parameters needed adjust
K=50;     % cluster number
num=4;  % the minimum sample numbers each cluster should include
para1=0.1;     % the distance between two clusters, less than which will be combined
para2=0.1;     % the standard deviation of samples in a cluster
tgtx = 60;
tgty = 27;
% data normalization
recover_hsi=hyperNormalize(data);
d=squeeze(recover_hsi(tgtx,tgty,:));
% background endmember extraction
[bkg,tgt,labs_sup,bmap,labs_iso,clus] = S2UE2(recover_hsi, d, K, num, para1, para2,args);
%% TSP detector
t=cputime;
[bkgsup,hdasp] = TSP(recover_hsi, bkg', tgt);
time = cputime - t;