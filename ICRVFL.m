function RVFL_IR()

load A
P=P1;
T=T1;
I=X;
J=Y;
C=2^(-8);clear A;

N=size(P,1);
num=size(I,1);            
dim=size(P,2);
ActivationFunction='asinh';

for m=1:10
L=1;
MSE_expect=0.10;
L_max=1000;
j=1;

weight = 0.1*(2*rand(L,dim)-ones(L,dim))+1i*0.1*(2*rand(L,dim)-ones(L,dim));
ind=ones(1,N);
bias = 0.1*rand(1,L)+1i*0.1*rand(1,L);
BiasMatrix=bias(ind,:);
tic;
tempH=P * weight.'+ BiasMatrix;
H1=asinh(tempH);
H=[P H1];
WL=inv(H' * H + C*eye(length(H'*H)))*H';
beta=WL*T;
Time_train(m,j)=toc;
Yout = H*beta;  
E = T-Yout;
MSE_train(m,j)=sqrt(E'*E/size(P,1));

%testing
 BiasMatrixT=BiasMatrix(1:num,:);
tempH_test = I * weight.'+ BiasMatrixT;
H_test1 = asinh(tempH_test);
H_test=[I H_test1];
Yout_test = H_test*beta;
E_test = J-Yout_test; 
MSE_test(m,j)=sqrt(E_test'*E_test/size(I, 1));
while (L<L_max) && (MSE_train(m,j)>MSE_expect)
    %training
    L=L+1;
    j=j+1;
   
    weight_i= 0.1*(2*rand(1,dim)-ones(1,dim))+1i*0.1*(2*rand(1,dim)-ones(1,dim));
    bias_i=0.1*rand+1i*0.1*rand;
    bias_m=bias_i(ind,:);
    tic;
    h=asinh(imultiplication(P,weight_i.')+bias_m);
    uh=imultiplication(h',h)+C;
    ml1=imultiplication(h',H);
    ml2=imultiplication(ml1,WL);
    ml3=imultiplication(ml2,h);
    ml=(h'-ml2)/(uh-ml3);
    WL=WL-rmultiplication(imultiplication(WL,h),ml);
    WL=[WL;ml];
    beta = rmultiplication(WL,T); 
    Time_train(m,j)=Time_train(m,j-1)+toc;
    H=[H h];
    Yout=H*beta;
    E2 = T-Yout;
    MSE_train(m,j)=sqrt(E2'*E2/size(P,1));
    
    weight=[weight;weight_i];
    BiasMatrix=[BiasMatrix bias_m];
    %testing
    BiasMatrixT=BiasMatrix(1:num,:);
    tempH_test = I * weight.'+ BiasMatrixT;
    H_test1 = asinh(tempH_test);
    H_test=[I H_test1];
    Yout_test = H_test*beta;
    E_test = J-Yout_test; 
    MSE_test(m,j)=sqrt(E_test'*E_test/size(I, 1));
    
end
end
MSE_test=sum(MSE_test,1)/10;
Time_train=sum(Time_train,1)/10;
for a=2:j
fileID = fopen('resultCRVFL-IRtest.txt','a');
      fprintf(fileID,'%12.4f',MSE_test(a));fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultCRVFL-IRtraintime.txt','a');
      fprintf(fileID,'%12.4f',Time_train(a));fprintf(fileID,'\n');
fclose(fileID);
end
end