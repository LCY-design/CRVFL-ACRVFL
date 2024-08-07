clc
clear all
load C
P=P1.';
T=T;
I=XP;
J=YT;
clear C;
C=2^(-6);
N=size(P,1);
num=size(I,1);
dim=size(P,2);
train_set=P;
train_label=T;
test_set=I;
test_label=J;


ActivationFunction='asinh';
Times=50;

Traintime=zeros(Times,1);
Testingtime=zeros(Times,1);
rmstraining=zeros(Times,1);
rmstesting=zeros(Times,1);
AverageTrainingTime=0;

wb=waitbar(0,'Please waiting...');
for hiddenn=5:5:100
for rnd = 1 : Times
    
   waitbar(rnd/Times,wb);
    
    %cpu_time_Train_start=cputime;
tic;
weight = 0.1*(2*rand(hiddenn,dim)-ones(hiddenn,dim))+1j*0.1*(2*rand(hiddenn,dim)-ones(hiddenn,dim));
bias = 0.1*rand(1,hiddenn)+1j*0.1*rand(1,hiddenn);
ind=ones(1,N);
BiasMatrix=bias(ind,:);

%%training 
tempH=train_set*weight.'+ BiasMatrix;
H=asinh(tempH);
%H=[real(H2),imag(H2)];
M = pinv(H'*H+(eye(size(H'*H))*C));
W=M*H'*T;
Train_time=toc;
rms_training=0;
Yout = H*W; %目标输出
E2 = train_label-Yout;%误差函数
rms_training=sqrt(E2'*E2/size(train_set,1));
%rms_training=show_arrc(E2);


%%testing
cpu_time_Test_start=cputime;
ind=ones(1,num);
BiasMatrixT=bias(ind,:);
tempH_test = test_set * weight.'+ BiasMatrixT;
H_test = asinh(tempH_test);
%H_test=[real(H_test0),imag(H_test0)];
%H_test=[real(H_test2),imag(H_test2)];

rms_testing=0;

YTest = H_test*W;
Test_time=toc-cpu_time_Test_start;

E3 = test_label-YTest; 
rms_testing=sqrt(E3'*E3/size(test_set,1));
%rms_testing=show_arrc(E3);


Traintime(rnd,1)=Train_time;
Testingtime(rnd,1)=Test_time;
rmstraining(rnd,1)=rms_training;
rmstesting(rnd,1)=rms_testing;
tic;



end

Averagermstesting=mean(rmstesting);
Averagermstraining=mean(rmstraining);
AverageTrainingTime =mean(Traintime);
AverageTestingTime=mean(Testingtime);

%AverageTrainingTime =AverageTrainingTime+AverageTrainingTime1;



fileID = fopen('resultCELM_wind.txt','a');
  fprintf(fileID,'Result for  hiddenn=%d,ActivationFunction=%s:\n', hiddenn,ActivationFunction);
   fprintf(fileID,'%12.4f',Averagermstesting);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',Averagermstraining);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTrainingTime);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_windte.txt','a');%以 a 的方式创建名为 result.txt 的文件（在文件末端添加数据）？
       fprintf(fileID,'%12.4f',Averagermstesting);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_windtre.txt','a');
       fprintf(fileID,'%12.4f',Averagermstraining);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_windtrt.txt','a');
      fprintf(fileID,'%12.4f',AverageTrainingTime);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_windtt.txt','a');
      fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);
end
close(wb);

subplot(2,2,1)
x =5:5:100;%hidden nodes
z1 =csvread('resultCELM_windte.txt');
plot(x,z1,'r');
grid on
legend('CELM')
xlabel('the number of hidden nodes')
ylabel('Testing-error')
subplot(2,2,2)
x =5:5:100;
z2=csvread('resultCELM_windtrt.txt');
plot(x,z2,'b');
grid on
legend('CELM')
xlabel('the number of hidden nodes')
ylabel('Training-time')