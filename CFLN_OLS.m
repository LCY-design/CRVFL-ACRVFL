clear;
n_samples = 10;

data= data_Savitha(1000);
X=data(:,1:4);
Y=data(:,5);
len = length(Y);
index = randperm(len);%生成1~len 的随机 数
train_set=X(index(1:round(len*0.7)),:);%训练样本输入
train_label=Y(index(1:round(len*0.7)),:);%训练样本输出
test_set=X(index(round(len*0.7)+1:end),:);%测试样本输入
test_label=Y(index(round(len*0.7)+1:end),:);%测试样本输出
P_train=train_set;

P_test=test_set;
T_train=train_label ;
T_test=test_label;
dim=size(P_train,2);
m=size(train_label,1);
n=size(test_label,1);
e1=0.001;
e2=0.001;
r=0;
W=[];
tau=0;
Q2=[];
Q1=[];
r=3;
  
    %生成多项式
%     if r==1
%         Train=[];
%         for k=1:m
%           T2=[1,P_train(k, :)];
%          Train(k,:) =T2; 
%       
%         end
% 
%     end
%     if r==2
%         Train=[];
% for k=1:m
%     T3=[];
%     for i= 1:dim
%         for j=i+1:dim
%              T1=P_train(k, i)*P_train(k, j);
%              T3=[T3,T1];
%         end
%     end
%     T2=[1,P_train(k, :),T3];
%     Train(k,:) =T2;
% end
%     end
% if r==3
%    Train=[];
%  for k=1:m
%     T3=[];
%     T4=[];
%     for i= 1:dim
%         for j=i+1:dim
%              T1=P_train(k, i)*P_train(k, j);
%              T3=[T3,T1];
%         end
%     end
%     for i= 1:dim
%         for j=i+1:dim
%             for t=j+1:dim
%              T5=P_train(k, i)*P_train(k, j);
%              T4=[T5,T4];
%             end
%         end
%     end
%     T2=[1,P_train(k, :),T3,T4];
%     Train(k,:) =T2;
%  end   

  
 Train=[];
%  for k=1:m
%     T3=[];
%     for i= 1:dim
%         for j=i:dim
%              T1=P_train(k, i)*P_train(k, j);
%              T3=[T3,T1];
%         end
%     end
%     T2=[1,P_train(k, :),T3];
%     Train(k,:) =T2;
% end
 for k=1:m
    T3=[];
    T4=[];
    for i= 1:dim
        for j=i:dim
             T1=P_train(k, i)*P_train(k, j);
             T3=[T3,T1];
        end
    end
    for i= 1:dim
        for j=i:dim
            for t=j:dim
             T5=P_train(k, i)*P_train(k, j)*P_train(k,t);
             T4=[T5,T4];
            end
        end
    end
    T2=[1,P_train(k, :),T3,T4];
    Train(k,:) =T2;
 end   


disp('运行第一次OLS');
[w,B,Q2,T] = OLS(W,Train,T_train,e2)
disp('结束运行第一次OLS');
b=B;
Q1=[Q1,Q2];
tau=tau+T;
 disp('运行第二次OLS');
%[w,b,q,t]=OLS([] ,Train(:,Q1),T_train,e2);
%disp('结束运行第二次OLS');
A=inv(w'*w)*w'* Train(:,Q2);
%alpha=inv(A)*b';
alpha=pinv(Train(:,Q2))*T_train;
Y_out=Train(:,Q2)*alpha;
%Y_out=w*A*alpha;
Training_mse=mean(abs(Y_out-T_train).^2);
Training_mse
r2=r;
 if r2==1
        Test=[];
        for k=1:n
          T2=[1,P_test(k, :)];
         Test(k,:) =T2; 
        end

    end
    if r2==2
        Test=[];
for k=1:n
    T3=[];
    for i= 1:dim
        for j=i:dim
             T1=P_test(k, i)*P_test(k, j);
             T3=[T3,T1];
        end
    end
    T2=[1,P_test(k, :),T3];
   Test(k,:) =T2;
end
    end
if r2==3
   Test=[];
 for k=1:n
    T3=[];
    T4=[];
    for i= 1:dim
        for j=i:dim
             T1=P_test(k, i)*P_test(k, j);
             T3=[T3,T1];
        end
    end
    for i= 1:dim
        for j=i:dim
            for t=j:dim
             T5=P_test(k, i)*P_train(k, j)*P_train(k, t);
             T4=[T5,T4];
            end
        end
    end
    T2=[1,P_test(k, :),T3,T4];
    Test(k,:) =T2;
 end   
end

Y_test=Test(:,Q2)*alpha;
Test_mse=mean(abs(Y_test-T_test).^2);
Test_mse
    