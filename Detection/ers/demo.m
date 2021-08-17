clc;
clear;
close all;
addpath('./ers_matlab_wrapper_v0.1');
path = '/home/sunli/4T/yanzi/DASP/sdae/sandiego1-100-0.005-64-0.05-0.01-0-189-20';
files = dir(strcat(path,'/model'));
number = size(files);
load(strcat(path,'/result/recover_hsi_',num2str(number(1)-3),'.mat'))
load(strcat(path,'/result/groundtruth','.mat'));
[h,w]=size(groundtruth);
[~,b]=size(recover_hsi);
hyperspectral = reshape(recover_hsi,[h w b]);
k=50; %superpixel number
[PCA_hsi, out_param] = PCA_img(hyperspectral,1); % obtain first PC
[labels_superpixel,bmapOnImg]=suppixel(PCA_hsi,k); % obtain superpixel
figure,imagesc(labels_superpixel),axis image,axis off;
figure,imagesc(bmapOnImg),axis image,axis off;
export_fig(2,'./superpixel.png','-r800''png');
% BW2=edge(PCA_hsi,'roberts',0.16);
% figure,imagesc(BW2),axis image,axis off;