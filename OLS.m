function [ W,B,Q2,T ] = OLS( W,Train,T_train,e2 )
Q2=[];
Q1=[];
[r0,c0]=size(Train);
Train1=Train;
Train2=Train;
Y1=1:c0;
Y2=1:c0;
T=0;
B=[];
Removed=[];

returnValue1=isempty(W);
if returnValue1~=1
    A=inv(W'*W)*W'*Train;
   Train=Train-W*A;
   disp('1');
end





  R=[];
  
while sum(Y2)~=0      
   for i=1:size(Y1,2)  
    if  Y2(i)~=0    
    R(i)=norm(Train(:,i)'*T_train)^2/((Train(:,i)'*Train(:,i))*(T_train'*T_train));
    if  R(i)<e2 
       Y2(i)=0;
        R(i)=0;
    end    
    end
   end
 
     [M,k2] = max(R);   
     
     if M>0
     Q2=[Q2,k2];
     Y2(k2)=0 ;
     
W=[W,Train(:,k2)];


%计算beta
B1=Train(:,k2)'*T_train /(Train(:,k2)'*Train(:,k2));
B=[B,B1];

%计算对应的Gamma和最大的gamma对应的分量p(:,k2)
T=T+ R(:,k2);
  R(k2)=0;
p(:,k2)=Train(:,k2);


%将未删除的项保留到Y3中
Y3=[];
for i=1:c0
    if Y2(i)~=0
        Y3=[Y3,Y2(i)];
    end
end

%让Train中剩余的列向量与p(:,k2)正交
a=p(:,k2)'*Train(:,Y3)/(p(:,k2)'*p(:,k2));
Train(:,Y3)=Train(:,Y3)-p(:,k2)*a;   
     end
end



