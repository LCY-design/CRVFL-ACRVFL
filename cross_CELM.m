warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

dataset=xlsread('Glass.xls');
%num=table2array(num);
% X=num(:,1:12);
% Y=num(:,13);


[data_r, data_c] = size(dataset);
%将数据样本随机分割为10部分
indices = crossvalind('Kfold', data_r, 10);
for hiddenn=10:50:600
for i = 1 : 10
    % 获取第i份测试数据的索引逻辑值
    test = (indices == i);
    % 取反，获取第i份训练数据的索引逻辑值
    train = ~test;
    %1份测试，9份训练
    test_set = dataset(test, 2 : data_c-1);
    test_label =dataset(test,data_c);
    
    train_set = dataset(train, 2 : data_c-1);
    train_label = dataset(train,data_c);
    % 使用数据的代码


% len = length(Y);
% index = randperm(len);%生成1~len 的随机数
% train_set=X(index(1:round(len*0.7)),:);%训练样本输入
% train_label=Y(index(1:round(len*0.7)),:);%训练样本输出
% test_set=X(index(round(len*0.7)+1:end),:);%测试样本输入
% test_label=Y(index(round(len*0.7)+1:end),:);%测试样本输出




%%产生训练集和数据集

%P_train,t_train训练数据集和标签

[p_train,ps1] = mapminmax(train_set');


p_test = mapminmax('apply',test_set',ps1);
t_train = ind2vec(train_label');
t_test  = ind2vec(test_label' );

%输出样本归一化
% t_train = categorical(train_label).';
% t_test = categorical(test_label).';
% [train_label,ps2] = mapminmax(train_label');
% test_label = mapminmax('apply',test_label',ps2);
% train_set=train_set.';
% test_set=test_set.';
% train_label=train_label.';
% test_label=test_label.';
p_train=exp(1i*pi*p_train);
p_test=exp(1i*pi*p_test);
P_train=p_train.';
P_test=p_test.';
T_train=t_train.' ;
T_test=t_test.';

C=2^(-25);
N=size(P_train,1);
num=size(P_test,1);
dim=size(P_train,2);
m=size(train_label,1);
n=size(test_label,1);
ActivationFunction='sigmoid';
Times=10;

Traintime=zeros(Times,1);
Testingtime=zeros(Times,1);
Acctraining=zeros(Times,1);
Acctesting=zeros(Times,1);




%wb=waitbar(0,'Please waiting...');

for rnd = 1 : Times
    
%    waitbar(rnd/Times,wb);
    
%     cpu_time_Train_start=cputime;
tic;
weight = 0.1*(2*1j*rand(hiddenn,dim)-ones(hiddenn,dim))+1j*0.1*(2*rand(hiddenn,dim)-ones(hiddenn,dim));
bias = 1j*0.1*rand(1,hiddenn)+1j*0.1*rand(1,hiddenn);
ind=ones(1,N);
BiasMatrix=bias(ind,:);

%%training 
tempH=P_train*weight.'+ BiasMatrix;
H1=sigmoid(tempH);


H=H1;
%H=[real(H2),imag(H2)];
%W=((pinv(H*H.'+0.5*C*eye(N))))*T_train;
M = pinv(H'*H+(eye(size(H'*H))*C));
W=M*H'*T_train;
Train_time=toc;
rms_training=0; 
Yout1 = H*W; %目标输出
% temp_Y = zeros(size(Yout1));
% for i = 1:size(Yout1, 2)
%         [~, index] = max(Yout1(:, i));
%         temp_Y(index, i) = 1;
% end
% Yout = vec2ind(temp_Y);   
T_sim1 = vec2ind(real(Yout1'));
% [t_train, index_1] = sort(t_train);
% T_sim1 = T_sim1(index_1);
Train_Acc = (sum(train_label'==T_sim1)/m)*100;
% E2 = train_label-Yout;%误差函数
% rms_training=sqrt(E2.'*E2/size(train_set,1));
%rms_training=show_arrc(E2);


%%testing
cpu_time_Test_start=cputime;
ind=ones(1,num);
BiasMatrixT=bias(ind,:);
tempH_test = P_test * weight.'+ BiasMatrixT;
H_test1 = sigmoid(tempH_test);

H_test=H_test1;
%H_test=[real(H_test0),imag(H_test0)];
%H_test=[real(H_test2),imag(H_test2)];

%rms_testing=0;

YTest1 = H_test*W;
Test_time=toc-cpu_time_Test_start;
% temp_Y = zeros(size(YTest1));
% for i = 1:size(YTest1, 2)
%         [~, index] = max(YTest1(:, i));
%         temp_Y(index, i) = 1;
% end
%YTest = vec2ind(temp_Y);
T_sim2 = vec2ind(real(YTest1'));
% [t_test, index_1] = sort(t_test);
% T_sim2 = T_sim2(index_1);

Test_Acc = (sum(test_label'==T_sim2)/n)*100;
% E3 = test_label-YTest; 
% rms_testing=sqrt(E3.'*E3/size(test_set,1));
%rms_testing=show_arrc(E3);


Traintime(rnd,1)=Train_time;
Testingtime(rnd,1)=Test_time;
Acctraining(rnd,1)=Train_Acc;
Acctesting(rnd,1)=Test_Acc;
tic;



end
end


AverageAcctesting=mean(Acctesting);
AverageAcctraining=mean(Acctraining);
AverageTrainingTime=mean(Traintime);
AverageTestingTime=mean(Testingtime);

AverageAcctesting
fileID = fopen('resultCELM_Cross.txt','a');
  fprintf(fileID,'Result for  hiddenn=%d,ActivationFunction=%s:\n', hiddenn,ActivationFunction);
   fprintf(fileID,'%12.4f',AverageAcctesting);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageAcctraining);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTrainingTime);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);

Averagermstesting=mean(Acctesting);
Averagermstraining=mean(Acctraining);
AverageTrainingTime=mean(Traintime);
AverageTestingTime=mean(Testingtime);


fileID = fopen('resultCELM_Crossl.txt','a');
  fprintf(fileID,'Result for  hiddenn=%d,ActivationFunction=%s:\n', hiddenn,ActivationFunction);
   fprintf(fileID,'%12.4f',AverageAcctesting);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageAcctraining);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTrainingTime);%fprintf(fileID,'\n');
   fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_Crosste.txt','a');%以 a 的方式创建名为 result.txt 的文件（在文件末端添加数据）？
       fprintf(fileID,'%12.4f',AverageAcctesting);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_Crosstre.txt','a');
       fprintf(fileID,'%12.4f',AverageAcctraining);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_Crosstrt.txt','a');
      fprintf(fileID,'%12.4f',AverageTrainingTime);fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCELM_Crosstt.txt','a');
      fprintf(fileID,'%12.4f',AverageTestingTime);fprintf(fileID,'\n');
fclose(fileID);
end
%close(wb);

% subplot(2,2,1)
% x =10:50:600;%hidden nodes
% z1 =csvread('resultELM_Crosste.txt');
% plot(x,z1,'r'); 
% grid on
% legend('ELM_Cross')
% xlabel('the number of hidden nodes')
% ylabel('Testing-Accuracy') 
% subplot(2,2,2)
% x =10:50:600;
% z2=csvread('resultELM_Crosstrt.txt');
% plot(x,z2,'b');
% grid on
% legend('ELM_Cross')
% xlabel('the number of hidden nodes')
% ylabel('Training-time')