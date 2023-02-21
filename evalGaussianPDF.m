function p= evalGaussianPDF(x,mean,cov) 
%evalute p(x) given mean and covarioance matrices

for i = 1:length(x)
p(i)= 1\sqrt(abs(2*pi*(cov(1,1)+cov(2,2))))*exp(-0.5*(x(:,i)-mean)'*inv(cov)*(x(:,i)-mean));
end 


end 


