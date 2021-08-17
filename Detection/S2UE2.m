function [bkg,tgt,labels_superpixel,bmapOnImg,labels_isodata,clusters]= S2UE2(data, d, K, num, para1, para2,arg)
% input:
%       data---HSI(h*w*b)
%       K----number of superpixel
%       arg---arg=1(spatial and spectral), arg=2(spatial), arg=3(spectral)
% output:
%       bkg----extracted endmember
%       labels_superpixel----label map after ERS superpixel segmentation
%       bmapOnImg-----edge map after ERS superpixel segmentation
%       labels_isodata-----label map after isodata clustering
%       clusters----clusters after isodata.
if (arg==1)
    disp('spatial-spectral unified background extraction');
    [labels_superpixel,bmapOnImg] = spatial(data,K);
    [h,w,~]=size(data);
%     data_ers = zeros(size(data));
    meanspec = [];
    for i=1:K
        tmp = labels_superpixel+1;
        tmp(tmp~=i) = 0;
        tmp(tmp==i) = 1;
        meanspec = [meanspec squeeze(sum(sum(data.*tmp,1),2)/sum(tmp(:)))];
%         meanspec = squeeze(sum(sum(data.*tmp,1),2)/sum(tmp(:)));
%         for j=1:h
%             for k=1:w
%                 if tmp(j,k)==1
%                     data_ers(j,k,:) = meanspec;
%                 end
%             end
%         end
    end
    t2 = cputime;
%     [labels_isodata,clusters] = spectral(data,meanspec,d,K,num,para1,para2);
    [labels_iso,clusters] = spectral(meanspec,K,num,para1,para2);
    fprintf(1,'Use %f sec. \n',cputime-t2);
    fprintf(1,'\t to obtain %d clusters. \n',size(clusters,1));
    labels_isodata = zeros(size(labels_superpixel));
    for i=1:size(clusters,1)
        loc = find(labels_iso==i);
        for j=1:size(loc,2)
            labels_isodata((labels_superpixel+1)==loc(j))=i;
        end
    end
%     labels_isodata = reshape(labels_isodata,[h w]);
elseif (arg==2)
    disp('spatial background extraction');
    [labels_superpixel,bmapOnImg] = spatial(data,K);
    clusters = [];
    for i=1:K
        tmp = labels_superpixel+1;
        tmp(tmp~=i) = 0;
        tmp(tmp==i) = 1;
        clusters = [clusters; (squeeze(sum(sum(data.*tmp,1),2)/sum(tmp(:))))'];
    end
    labels_isodata = labels_superpixel;
else
    disp('spectral background extraction');
    [h,w,~]=size(data);
    t2 = cputime;
    [labels_isodata,clusters] = spectral(data,K,num,para1,para2);
    fprintf(1,'Use %f sec. \n',cputime-t2);
    fprintf(1,'\t to obtain %d clusters. \n',size(clusters,1));
    labels_isodata = reshape(labels_isodata,[h w]);
    labels_superpixel = zeros(size(labels_isodata));
    bmapOnImg = zeros(size(labels_isodata));
end

radians = (hyperSam(clusters',d,0))';
% exclude the potential target spectra in U
indexbkg = [];
indexd = [];
for i=1:size(radians,1)
    if (radians(i) >0.98)
        indexbkg=[indexbkg i];
    end
    if (radians(i) >0.99)
        indexd=[indexd i];
    end
end
bkg = clusters;
tgt = d;
if size(indexbkg,2) ~= size(radians,1)
    bkg(indexbkg(:),:)=[];
end
if size(indexd,2) ~= size(radians,1)
    tgt=[tgt (clusters(indexd(:),:))'];
end

function [labels_superpixel,bmapOnImg] = spatial(hsi,K)
addpath('./ers');
addpath('./ers/ers_matlab_wrapper_v0.1');
[PCA_hsi, ~] = PCA_img(hsi,1); % obtain first PC
[labels_superpixel,bmapOnImg]=suppixel(PCA_hsi,K); % obtain superpixel

function [labels_isodata,clusters] = spectral(data,K,num,para1,para2)
% % initial clustering center
% data_ers2D = hyperConvert2d(data_ers);
% [D,~]=hyperAtgp(data,5);
% % final clustering center
% [labels_isodata,clusters] = isodata(data_ers2D', [D d]', 2, 30, num, para1, para2);
% initial clustering center
if size(size(data),2)==3
    data = hyperConvert2d(data);
end
% final clustering center
[labels_isodata,clusters] = isodata(data', 20, 2, 30, num, para1, para2);
