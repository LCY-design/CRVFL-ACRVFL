num=xlsread('qsar.csv');
X=num(:,1:8);
Y=num(:,9);
%%Download Housing Prices
len = length(Y);

index = randperm(len);%生成1~len 的随机数
train_set=X(index(1:round(len*0.7)),:);%训练样本输入
train_label=Y(index(1:round(len*0.7)),:);%训练样本输出
test_set=X(index(round(len*0.7)+1:end),:);%测试样本输入
test_label=Y(index(round(len*0.7)+1:end),:);%测试样本输出




%%产生训练集和数据集



[train_set,ps1] = mapminmax(train_set');


test_set = mapminmax('apply',test_set',ps1);

%输出样本归一化

[train_label,ps2] = mapminmax(train_label');

test_label = mapminmax('apply',test_label',ps2);
train_set=exp(1i*pi*train_set);
test_set=exp(1i*pi*test_set);
train_label=train_label.';
test_label=test_label.';
train_set=train_set.';
test_set=test_set.';


C=2^(-7);
N=size(train_set,1);
num=size(test_set,1);
dim=size(train_set,2);
ActivationFunction='asinh';
Times=50;

Traintime=zeros(Times,1);
Testingtime=zeros(Times,1);
rmstraining=zeros(Times,1);
rmstesting=zeros(Times,1);







wb=waitbar(0,'Please waiting...');
for hiddenn=10:10:200
for rnd = 1 : Times
    
    waitbar(rnd/Times,wb);
    
%     cpu_time_Train_start=cputime;
tic;
weight = 0.1*(2*rand(hiddenn,dim)-ones(hiddenn,dim));
bias = 0.1*rand(1,hiddenn);
ind=ones(1,N);
BiasMatrix=bias(ind,:);

%%training 
tempH=train_set*weight.'+ BiasMatrix;
H1=asinh(tempH);


H=[train_set,H1];
%H=[real(H2),imag(H2)];
M = pinv(H'*H+(eye(size(H'*H))*C));
W=M*H'*train_label;
Train_time=toc;
rms_training=0;
Yout = H*W; %目标输出
E2 = train_label-Yout;%误差函数
rms_training=sqrt(E2.'*E2/size(train_set,1));
%rms_training=show_arrc(E2);


%%testing
cpu_time_Test_start=cputime;
ind=ones(1,num);
BiasMatrixT=bias(ind,:);
tempH_test = test_set * weight.'+ BiasMatrixT;
H_test1 = asinh(tempH_test);

H_test=[test_set,H_test1];
%H_test=[real(H_test0),imag(H_test0)];
%H_test=[real(H_test2),imag(H_test2)];

rms_testing=0;

YTest = H_test*W;
Test_time=toc-cpu_time_Test_start;

E3 = test_label-YTest; 
rms_testing=sqrt(E3.'*E3/size(test_set,1));
%rms_testing=show_arrc(E3);


Traintime(rnd,1)=Train_time;
Testingtime(rnd,1)=Test_time;
rmstraining(rnd,1)=rms_training;
rmstesting(rnd,1)=rms_testing;
tic;



end
Averagermstesting=mean(rmstesting);
Averagermstraining=mean(rmstraining);
AverageTrainingTime=mean(Traintime);
AverageTestingTime=mean(Testingtime);


fileID = fopen('resultRVFL_Real.txt','a');
  fprintf(fileID,'Result for  hiddenn=%d,ActivationFunction=%s:\n', hiddenn,ActivationFunction);
   fprintf(fileID,'%12.4f',Averagermstesting);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',Averagermstraining);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTrainingTime);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);

Averagermstesting=mean(rmstesting);
Averagermstraining=mean(rmstraining);
AverageTrainingTime=mean(Traintime);
AverageTestingTime=mean(Testingtime);


fileID = fopen('resultRVFL_Realchannel.txt','a');
  fprintf(fileID,'Result for  hiddenn=%d,ActivationFunction=%s:\n', hiddenn,ActivationFunction);
   fprintf(fileID,'%12.4f',Averagermstesting);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',Averagermstraining);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTrainingTime);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultRVFL_Realchannelte.txt','a');%以 a 的方式创建名为 result.txt 的文件（在文件末端添加数据）？
       fprintf(fileID,'%12.4f',Averagermstesting);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultRVFL_Realchanneltre.txt','a');
       fprintf(fileID,'%12.4f',Averagermstraining);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultRVFL_Realchanneltrt.txt','a');
      fprintf(fileID,'%12.4f',AverageTrainingTime);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultRVFL_Realchanneltt.txt','a');
      fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);
end
close(wb);

subplot(2,2,1)
x =10:10:200;%hidden nodes
z1 =csvread('resultRVFL_Realchannelte.txt');
plot(x,z1,'r');
grid on
legend('RVFL_Real')
xlabel('the number of hidden nodes')
ylabel('Testing-error')
subplot(2,2,2)
x =10:10:200;
z2=csvread('resultRVFL_Realchanneltrt.txt');
plot(x,z2,'b');
grid on
legend('RVFL_Real')
xlabel('the number of hidden nodes')
ylabel('Training-time')

