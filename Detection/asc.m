function [cluster, Z] = asc(X, k, L, I, ON, OC, OS)

%%%%%%%%%%%%�������?%%%%%%%%%%%%%
% input: X---data(N x p);
%        clusterZ--��ʼ��������(k x p)��
%        L---��һ�ε�������п��Ժϲ��ľ������ĵ�������?
%        I---����������
%        ON--ÿһ����������������Ŀ��
%        OC--������������֮�����С���룬С�ڸþ���������������кϲ���
%        OS--һ�������������ֲ��ı�׼�
%        NO--
% output: cluster ---- ÿ�����ص����?1 x N����
%         Z ---- ��������(* x p),����*��ʾ���յľ�����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ����������?  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=hyperNormalize(X);
s=size(X,1);
cluster=zeros(1,s); % ��ʱ�ռ���ڲ����
iter=0;
final=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  1.��ʼ������  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% k=size(clusterZ,1);  %        k---Ԥ�ڵľ���������
Z=inicializa_centros(X,k);
% Z = clusterZ;



%%%%%%%%%%%%%%%%%%%%%%%%
%  ��Ҫ����  %
%%%%%%%%%%%%%%%%%%%%%%%%

while final==0   %����ѭ�����?
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%  3.����ͷ���ÿ��������ģ�X��Y��  %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for i=1:s
        cluster(i)=cercano(X(i,:), Z);
    end
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%  4.ɾ��Ⱥ��  %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% ������Լ������ǣ������Ҫ���ľ���������Ŀ�ͷ��֡�
    [Z, cluster]=eliminar(cluster, Z, X, ON);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%  5.����Ⱥ������  %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	Z=recalcula(cluster, X, Z);
    
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%  6.��󣬷��ѻ�ϲ�  %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (iter==I)
        final=1;
        next=0;
	else
        next=decide78(iter, k, size(Z,1));
    end


	%%%%%%%%%%%%%%%
	%  7.����  %
	%%%%%%%%%%%%%%%
	if next==1
        next=2;  %һ��������ѣ���ô����һ�κ�ͷ��صڶ�������û�з�����ѣ���ֱ�ӽ���ϲ����??
        hubo_division=0;
        A=size(Z,1);
        % ����ÿ���������ĵ�ƽ����룬�ܵ�ƽ����룬��׼ƫ��
        
        [Di, D, STM]= dispersion(X, Z, cluster);
	
        % ���������ĸ�����з���?
		i=0;    % �������������Ѻ͵��?                         
		while (hubo_division==0) && (i < A)      % �������ĳ��?...                                     
		    i=i+1;                                                                                      
		    index=find(cluster==i); % index��ʾȺ����ֵ����      
		    sindex=size(index,2);                                                                                                                                                   
		    if  (STM(i)>OS) && (((Di(i)>D) && (sindex>(2*(ON+1)))) || (A<=(k/2)) ) 
		        hubo_division=1;                                                                        
		        next=1;                                                                                 
		        [Z, cluster]=dividir(STM, cluster, Z, i, X);    % ����.                                                       
		        iter=iter+1;                                                                            
            end                                                                                         
        end                                                                                            
    end

    
	%%%%%%%%%%%%%%%
	%  8.�ϲ�  %
	%%%%%%%%%%%%%%%
	if  next==2
        %��������������L֮�����̾��롣
        [orden, Dij]= distancia_centros(Z, OC, L);   %���? orden==0 ��û��ʲô�ϲ���.
        % �ϲ��ļ�Ⱥ��
        if  orden(1) > 0
            [cluster, Z]=union(orden, cluster, Z, Dij);
            % ������
            Z=recalcula(cluster, X, Z);
        end
    end
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%  9.��ֹ������ѭ��  %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if (next==2)
        if (iter==I)
            final=1;
        end
        iter=iter+1;
%         [iter,final,vuelve3]= termina_o_itera(iter, I, NO);
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ��ֹѭ����������ѭ���꣩  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  �㲻�����С����?  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for j=1:s
%     temp=0;
%     P=X(j,:);
%     for i=1:A
%         if (distancia(P,Z(i,:)) > min)
%             temp=temp+1;
%         end
%     end
%     if  temp==A
%         cluster(j)=0;
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  K��ֱ��������ģʽK %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xcluster=0;
% Ycluster=0;
% for m=1:k
%     inedx=0;
%     index=find(cluster==m);
%     s2=size(index,2);
%     for n=1:s2
%         Xcluster(1,n,m)= X(index(n));
%         Ycluster(1,n,m)= Y(index(n));
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     ��������                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %              ������ʹ�õ��Ӻ���                 %%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  �˺���������ÿ����ӽ����ĵĵ�?  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m] = cercano(data, Z)
    dtemp=0;
    for j=1:size(Z,1)
        d=distancia(Z(j,:), data); % �����ĵ������?
        if j<2
            m=j;    % ��һ������ʼ������Ч�ġ�
            dtemp=d;
        elseif d < dtemp
            m=j;    %��������Ӧ������ 
            dtemp=d;  % ������С
        end
    end  
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  �˺�������������  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Zout1] = recalcula(cluster, data, Z) %Ϊÿ����ƽ����������
    Zout1=zeros(size(Z));
	for m=1:size(Z,1)
        index=find(cluster==m);
        if isempty(index)==0
            sindex=size(index,2);
            Zout1(m,:)=(sum(data(index,:),1)) / sindex;      
        else
            Zout1(m,:)=Z(m,:);
        end
    end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ����ÿ�����������ĵ�ƽ����룬�ܵ�ƽ����룬��׼ƫ�����ķ��� %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ditemp, Dtemp, STMAX] = dispersion(data, Z, cluster)
    A=size(Z,1);
    Ditemp=zeros(A,1);
    Dtemp=zeros(1,1);
    STMAX=zeros(1,A);
    for i=1:A
        suma=0;
        index=find(cluster==i);
        sindex=size(index,2);
        for j=index
            P=data(j,:);
            d=distancia(Z(i,:), P);% �������� Xi��Zi�ľ���.
            suma=suma + d;   % sumax
        end
        % ��ɢ��Ⱥ��
        Ditemp(i,:)=suma / sindex;
        % ȫ����ʱ��ɢ
        Dtemp=Dtemp + (Ditemp(i,:) * sindex);% �ܺ� Ni*Di
        %��ɢ�ı���
        ST=std(data(index,:));
        % ��ɢ�������?
        STMAX(i)=max(ST);
    end
    % ȫ�ַ�ɢ���?
    Dtemp=Dtemp / size(data,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   ����������Ѱ�Ҿ�������  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ztemp, clustertemp]=eliminar(cluster, Z, data, ON)
	%����
    %λ�ƣ������������ģ��������?
    A=size(Z,1);
	desplazamiento=zeros(1,A);  % ����ܵ�ֵ��?: -1, (=0) �� (>0).
	for i=1:A                  % Si -1: ɾ�����顣���? 0: ����ı�?
        cont=find(cluster==i);  % Si >0: ��ȥ�����ܶ��λ����ʾ���?.
        scont=size(cont,2);        
        if scont < ON
            desplazamiento(i)=-1;
            if  i < A
                for j=(i+1):A
                    desplazamiento(j)=desplazamiento(j)+1;
                end
            end   
        end
    end
	%����
    [Ztemp, clustertemp]=reduce(desplazamiento, cluster, Z);
	% ���ķ��䵽��Ŀ���������?
    if isempty(Ztemp)==1  % ���ȡ������Ⱥ��?
        Ztemp=median(data,1);
    end
	vacio=find(clustertemp==0);
	if  isempty(vacio)==0
        for i=vacio
            clustertemp(i)=cercano(data(i,:), Ztemp);
        end
    end
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  �����ʵ��ľ�������ģ������������������������µ��?  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [clustertemp, Ztemp]=union(orden, cluster, Z, Dij)
    A=size(Z,1);
    clustertemp=cluster;
    sorden=size(orden,2);
    unidos=0;
    uindex=0;
    marca=zeros(1,A);
    for i=1:sorden
        yaunido=0;
        temp=[0 0];
        [fcnum(1),fcnum(2)]=find(Dij==orden(i)); %   fcnum(1) < fcnum(2)
        for j=1:2
            if isempty(find(unidos==fcnum(j)))==0
                yaunido=1;
            else
                temp(j)=fcnum(j);
            end
        end        
        if yaunido==0
            for h=1:2  
                unidos(uindex+1)=temp(h);
            end
            marca(fcnum(2))=-1;
            selec=find(clustertemp==fcnum(2));   % ѡ��������RESP�������顣���y
            clustertemp(selec)=fcnum(1);         % �����˸��Եļ�ֵ�����С����?.
        end
    end
    
    adicion=0;  %��������ʽ����Ϣ��������'���?'��
    for i=1:A
        if  marca(i) >= 0
            marca(i)=marca(i)+adicion;
        else
            adicion=adicion+1;
        end
    end
    % ���٣����б�Ҫ��ѧϰ��Ŀ���ҵ���Ⱥ��
    [Ztemp,clustertemp]=reduce(marca, clustertemp, Z);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  �������ĺ�����  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ztemp, clustertemp]=reduce(desplazamiento, cluster, Z)
    clustertemp=cluster;
    Ztemp=find(size(Z,1)==999999);    
    for i=1:size(Z,1)
        if  (desplazamiento(i) < 0)
            selec=find(cluster==i);
            if isempty(selec)==0
                clustertemp(selec)=0;
            end
        else
            Ztemp( (i-desplazamiento(i)), :)= Z(i,:);
            selec=find(clustertemp==i);
            clustertemp(selec)=clustertemp(selec)-desplazamiento(i);
        end
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��������ĺ��Ե���˳����̾���%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [orden, Dij]= distancia_centros(Z, OC, L)
    A=size(Z,1);
    Dij=zeros((A-1),A);
    %����֮��ľ�����㡣
    for i=1:(A-1)
        for j=(1+i):A
            Dij(i,j)=distancia(Z(i,:), Z(j,:));
        end
    end
    
    %�����L�����OCС��
    index= find((Dij>0) & (Dij<OC))';
    if (isempty(index)==0)
        orden=sort(Dij(index));
        sorden=size(orden,2);
        if sorden>L
            orden=orden(1,1:L);
        end
    else
        orden=0;
    end

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ���Ѿ���  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ztemp, clustertemp]=dividir(ST, cluster, Z, ncentro, data)
    clustertemp=cluster;
    Ztemp=Z;
    k2=0.5;          % 0 < k2 < 1
    Yi=ST(ncentro) * k2;
    Atemp=size(Z,1)+1;
    Ztemp(Atemp,:)=Ztemp(ncentro,:);         % ��Ⱥ����
    m=find(Ztemp(ncentro,:)==max(Ztemp(ncentro,:)));    % Э��ָ��ϸ�?
    Ztemp(ncentro,m)=Ztemp(ncentro,m)+Yi;  % Z+=Z(ncentro)
    Ztemp(Atemp,m)=Ztemp(Atemp,m)-Yi;      % Z-=Z(Atemp)
    dividendo= find(clustertemp==ncentro);
    for i=(dividendo)
        P=data(i,:);
        if  (distancia( P, Ztemp(ncentro,:) )) >= (distancia(P, Ztemp(Atemp,:)))  %d(Z+) >= d(Z-)
            clustertemp(i)=Atemp;
        end
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ����������֮��ľ���?  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dist]= distancia(Z1, Z2)
%     dist=sqrt( ((Z1(1)-Z2(1))^2) + ((Z1(2)-Z2(2))^2) );% ������֮��ľ���?.
    dist = 1-(Z1*Z2')/sqrt((Z1*Z1')*(Z2*Z2'));
          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  �����Ƿ����ó�����Ļ�����ѭ����һЩ����?  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [itertemp,FINtemp,vuelve3temp]= termina_o_itera(iter, I, NO)
    itertemp=iter;
    FINtemp=0;
    vuelve3temp=1;
    if itertemp==I
        FINtemp=1;
    else
        if NO==1
            vuelve3temp=1;    
        else
            preg2=find(iter==0); % Resp. vacia
            fprintf('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n');
            fprintf('��ǰ���? %d \n',itertemp);
            while (isempty(preg2)==1) | ( (preg2~=2) & (preg2~=1) ),
                fprintf('Ҫ��ģ�Ȼ�󷵻ص���κβ��� ����= 1����= 2�� \n ��');
                preg2=input('');
            end;
            if preg2==1
                vuelve3temp=0;  % ת����2�����޸Ĳ���
            else
                vuelve3temp=1;  % ת����3��
            end;
        end;
        itertemp=itertemp+1;
    end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%  �����Ƿ�ǰ����8��7  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nexttemp]=decide78(iter, k, A)
    nexttemp=0;
    if A <= (k/2)
        nexttemp=1;     % ת������7.
    end
    if ( (A>=(2*k)) || ( (iter>0) && (((iter+1)/2)>(ceil(iter/2)))) )   % ���? a>=2k ������ȡ�?
        nexttemp=2;     % ת������ 8.
    end
    if nexttemp==0
        nexttemp=1;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ��ʼ������ %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
function [Z]=inicializa_centros(data, k)
    % ���ȷֲ�����
    dx= (max(data))-(min(data));
    dzx= dx/(k+1);    % Э������֮��ľ���? X.
    for i=1:k
        Z(i,:)=min(data)+(dzx*i);
    end