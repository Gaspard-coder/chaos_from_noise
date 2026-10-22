close all
%clear
clc
tic
%load frac_alea.mat
x = frac_alea(4,:);
%x = data_L1_segunda_vuelta;
%load blanco_gaussiano.mat
%x = rbg(1,:);
load ruido_blanco.mat
x = ruido_blanco(4,:);
%10 muestras de 10000 elementos
%parametros 
m = 6;
t = 1;
alpha = 0.001;
%imf= [];
%imf_emd=[];
%for k=1:10
%    signal=x(k,:);  
%Descomposiciones en diferentes formas
%wally(k)=struct(emd(signal));
%[imf(:,k),res,info] = emd(signal);
%end

%imf = modwtmra(modwt(x,"db2"));
%imf = imf';
%[imf,res,info] = ewt(x);

%[imf,res,info] = emd(x,Interpolation="pchip");
[imf,res,info] = emd(x,Interpolation="spline",Display=1);


%Nstd = 0.2;
%NR = 500;
%MaxIter = 5000;


%[modes its]=ceemdan(x,0.2,500,5000);


%[imf_v,res_v,info_v] = vmd(x);
%[imf_w,res_w] = ewt(x);
%mra_x = modwtmra(modwt(x,"db2"));

%=====================================================
% IMPORTANTE
%[imf,res_v,info_v] = vmd(x,NumIMFs=10);
%=====================================================

disp('Test Estadistico')
for k=1:10
    disp(k)
    test_estadistico(k)=runstest(imf(:,k),alpha);
end
test_estadistico(k+1) =runstest(x,alpha);

disp('Exponentes de Lyapunov')
for k=1:10
    disp(k)
        test_liap(k)=lyapunovExponent(imf(:,k),1);
        test_liap(k)=lyapunovExponent(imf(:,k),227);
end
test_liap(k+1) = lyapunovExponent(x,2.7e-3);


% disp('Test Caotico')
% for k=1:10
%     disp(k)
%         test_chaos(k)=chaostest(imf(:,k),'TANH',[5 6 5],alpha);
% end   

% y = imf(:,3);
% 
%  figure(4)
%  xrec = y(1:length(y)-1);
%  yrec = diff(y(1:length(y)),1);
%  %subplot(1,2,2)
%  plot(xrec,yrec,'r')
%  xlabel('xrec')
%  ylabel('yrec')
%  title('Reconstrucción de atractores sobre datos incognita 2')
%  
%  
%  figure(5)
%  xrec = y(1:length(y)-2);
%  yrec = diff(y(1:length(y)-1),1);
%  zrec = diff(y(1:length(y)),2)
%  %subplot(1,2,2)
%  plot3(xrec,yrec,zrec,'r')
%  xlabel('xrec')
%  ylabel('yrec')
%  title('Reconstrucción de atractores sobre datos incognita 2')
 entro_original = HXC_1(x',6,1);
 entro_1 = HXC_1(imf(:,1),6,1);
 entro_2 = HXC_1(imf(:,2),6,1);
 entro_3 = HXC_1(imf(:,3),6,1);
 entro_4 = HXC_1(imf(:,4),6,1);
 entro_5 = HXC_1(imf(:,5),6,1);
 entro_6 = HXC_1(imf(:,6),6,1);
 entro_7 = HXC_1(imf(:,7),6,1);
 entro_8 = HXC_1(imf(:,8),6,1);
 entro_9 = HXC_1(imf(:,9),6,1);
 entro_10 = HXC_1(imf(:,10),6,1);
  
H_original = entro_original(1,1,1);
H(1) = entro_1(1,1,1);
H(2) = entro_2(1,1,1);
H(3) = entro_3(1,1,1);
H(4) = entro_4(1,1,1);
H(5) = entro_5(1,1,1);
H(6) = entro_6(1,1,1);
H(7) = entro_7(1,1,1);
H(8) = entro_8(1,1,1);
H(9) = entro_9(1,1,1);
H(10) = entro_10(1,1,1);

C_original = entro_original(1,3,1);
C(1) = entro_1(1,3,1);
C(2) = entro_2(1,3,1);
C(3) = entro_3(1,3,1);
C(4) = entro_4(1,3,1);
C(5) = entro_5(1,3,1);
C(6) = entro_6(1,3,1);
C(7) = entro_7(1,3,1);
C(8) = entro_8(1,3,1);
C(9) = entro_9(1,3,1);
C(10) = entro_10(1,3,1);

figure(2)
plano_HXC 
hold on
original = 'S';
%title('Informational Plane Entropy Complexity','FontSize',14)
text(H_original,C_original,original','FontSize',12)

 figure(1)
 for k = 1:10
 subplot(5,2,k)
 plot(imf(:,k))
 num = num2str(k);
 title(['IMF'  num])
 end
 
toc 