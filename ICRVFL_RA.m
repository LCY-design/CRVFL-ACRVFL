function RVFL_RA()

load A
P=P1;
T=T1;
I=X;
J=Y;
C=2^(-15);
clear A;

N=size(P,1);
num=size(I,1);
dim=size(P,2);
for m=1:2
L=2;
MSE_expect=0.1;
L_max=1018;
j=1;

weight = 0.1*(2*rand(1,dim)-ones(1,dim))+1i*0.1*(2*rand(1,dim)-ones(1,dim));
ind=ones(1,N);
bias = 0.1*rand+1i*0.1*rand;
BiasMatrix=bias(ind,:);
tic;
tempH=P * weight.'+ BiasMatrix;
H1=asinh(tempH);
H2=[P H1];
H=[real(H2) imag(H2)];
WL=inv(H' * H + C*eye(length(H'*H)))*H';
beta=WL*T;
Time_train(m,j)=toc;
Yout = H*beta;  
E = T-Yout;
MSE_train(m,j)=sqrt(E'*E/size(P,1));

BiasMatrixT=BiasMatrix(1:num,:);
tempH_test = I * weight.'+ BiasMatrixT;
H1_test = asinh(tempH_test);
H2_test=[I H1_test];
H_test=[real(H2_test) imag(H2_test)];
Yout_test = H_test*beta;
E_test = J-Yout_test; 
MSE_test(m,j)=sqrt(E_test'*E_test/size(I, 1));


while (L<L_max) && (MSE_train(m,j)>MSE_expect)
    L=size(H,2)+2;
    j=j+1;
    Q=[eye((L-2)/2) zeros((L-2)/2,1) zeros((L-2)/2) zeros((L-2)/2,1);zeros((L-2)/2) zeros((L-2)/2,1) eye((L-2)/2) zeros((L-2)/2,1);zeros(1,(L-2)/2) 1 zeros(1,(L-2)/2) 0;zeros(1,(L-2)/2) 0 zeros(1,(L-2)/2) 1 ];
    U=[eye((L-2)/2+1) 1i*eye((L-2)/2+1);eye((L-2)/2+1) -1i*eye((L-2)/2+1)];
    weight_i= 0.1*(2*rand(1,dim)-ones(1,dim))+1i*0.1*(2*rand(1,dim)-ones(1,dim));
    bias_i=0.1*rand+1i*0.1*rand;
    bias_m=bias_i(ind,:);
    tic;
    h=asinh(imultiplication(P,weight_i.')+bias_m);
    h=[real(h) imag(h)];
    uh=rmultiplication(h',h)+C*eye(2);
    ml1=rmultiplication(h',H);
    ml2=rmultiplication(ml1,WL);
    ml3=rmultiplication(ml2,h);
    ml=rmultiplication(inv((uh-ml3)),(h'-ml2));
    WL=WL-rmultiplication(rmultiplication(WL,h),ml);
    WL=[WL;ml];
    beta = imultiplication(WL,T); 
    Time_train(m,j)=Time_train(m,j-1)+toc;
    beta=Q.'*beta;
    H=[H h]*Q;
    WL=Q.'*WL;
    Yout=H*beta;
    E2 = T-Yout;
    MSE_train(m,j)=sqrt(E2'*E2/size(P,1));
    
    weight=[weight;weight_i];
    BiasMatrixT=[BiasMatrixT bias_m(1:num,:)];
    %test
    H_test1=asinh(I*weight.'+BiasMatrixT);
    H_test2=[I H_test1];
    H_test=[real(H_test2) imag(H_test2)];
    Yout_test = H_test*beta;
    E_test = J-Yout_test; 
    MSE_test(m,j)=sqrt(E_test'*E_test/size(I, 1));
    
    
end
end
MSE_test=sum(MSE_test,1)/2;
Time_train=sum(Time_train,1)/2;
for a=2:j
fileID = fopen('resultIR-RVFL-chaoticAHtest1.txt','a');
      fprintf(fileID,'%12.4f',MSE_test(a));fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('resultIR-RVFL-chaoticAHtraintime1.txt','a');
      fprintf(fileID,'%12.4f',Time_train(a));fprintf(fileID,'\n');
fclose(fileID);

end
end
