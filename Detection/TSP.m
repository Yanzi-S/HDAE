function [dataU3D,detmap] = TSP(data, bkg, d)
% input:
%       data---HSI(h*w*b)
%       bkg----extracted background endmember (b*K, where K is the number of background)
%       d----desired target (b*1)
% output:
%       detmap----detection map
[h,w,b]=size(data);
data2D = hyperConvert2d(data);
P_U = eye(b) - bkg * (inv(bkg'*bkg+0.0001*eye(size(bkg,2)))*bkg');
dataU = P_U * data2D;
dU = P_U * d;
dataU3D=reshape(dataU',[h w b]);
% ��Ŀ���ӿռ�ͶӰ
dataD = [];
for i=1:size(dU,2)
    S_d = inv(dU(:,i)' * dU(:,i)+0.0001*eye(size(dU(:,i),2)))*dU(:,i)';
    dataD = [dataD; S_d*(dataU)];    
end
% for i=1:size(dU,2)
%     dataD = [dataD; hyperSam(dataU,dU(:,i),0)];    
% end
result = max(dataD,[],1);
detmap = reshape(result,h,w);
% 
% S_d = inv(dU' * dU+0.00001*eye(size(dU,2)))*dU';
% dataD = S_d*dataU;
% detmap = reshape(dataD,h,w);
