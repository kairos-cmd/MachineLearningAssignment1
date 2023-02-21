clear all;close all;clc;
wqw=readmatrix('winequality-white.csv');
quality = wqw(:,12);
n_class=length(unique(quality));
% n_label=size(quality);
[n_sample,~]= size(wqw);
%calculate the number of each class
n_quality = zeros(1,11);
[~,n_label]=size(n_quality);
class_pos={'0','1','2','3','4','5','6','7','8','9','10'};
for i = 1:11
    n_quality(i)=sum(wqw(:,12)==i-1);
    class_pos{i}=find(quality(:,1)==(i-1));
end
%calculate the prior probability for each class
p=n_quality/n_sample;
%calculate mu for each class
mu=zeros(11,11);
for j1 = 1:11
    for j2 = 4:10
    mu(j1,j2) = sum(wqw(class_pos{j2},j1))/n_quality(j2);
    end
end
%extract each class samples into individual matrix
[c4_matrix]=extract(class_pos{4},wqw);
[c5_matrix]=extract(class_pos{5},wqw);
[c6_matrix]=extract(class_pos{6},wqw);
[c7_matrix]=extract(class_pos{7},wqw);
[c8_matrix]=extract(class_pos{8},wqw);
[c9_matrix]=extract(class_pos{9},wqw);
[c10_matrix]=extract(class_pos{10},wqw);
%calculate and regularize the covariance of each class matrix
digt=ones(1,11);
e=diag(digt);
a=0.6;
%Class 4
temp_c4=cov(c4_matrix);
sigma_c4=temp_c4+(a*trace(temp_c4)/rank(temp_c4))*e;
% pdf_c4 = mvnrnd(mu(:,4),sigma_c4,n_quality(4));
%Class 5
temp_c5=cov(c5_matrix);
sigma_c5=temp_c5+(a*trace(temp_c5)/rank(temp_c5))*e;
%Class 6
temp_c6=cov(c6_matrix);
sigma_c6=temp_c6+(a*trace(temp_c6)/rank(temp_c6))*e;
%Class 7
temp_c7=cov(c7_matrix);
sigma_c7=temp_c7+(a*trace(temp_c7)/rank(temp_c7))*e;
%Class 8
temp_c8=cov(c8_matrix);
sigma_c8=temp_c8+(a*trace(temp_c8)/rank(temp_c8))*e;
%Class 9
temp_c9=cov(c9_matrix);
sigma_c9=temp_c9+(a*trace(temp_c9)/rank(temp_c9))*e;
%Class 10
temp_c10=cov(c10_matrix);
sigma_c10=temp_c10+(a*trace(temp_c10)/rank(temp_c10))*e;
sigma = {'1','2','3','4','5','6','7'};
sigma{1}=sigma_c4;sigma{2}=sigma_c5;sigma{3}=sigma_c6;
sigma{4}=sigma_c7;sigma{5}=sigma_c8;sigma{6}=sigma_c9;
sigma{7}=sigma_c10;
%Calculate the class condition pdf
condition_pdf=zeros(7,n_sample);
px = zeros(1,n_sample);
for i=1:7
    condition_pdf(i,:)=evalGaussianPDF(wqw(:,1:11)',mu(:,i),sigma{i});
    px = px + p(i+3)*condition_pdf(i,:);
end
%calculate the posteriors probability
posterior = pos_cal(condition_pdf,p,n_sample,px);
% 
 
%use 0-1 loss matrix
loss_matrix = ones(7,7)-diag(ones(1,7));
%classify the data
expectedRisk = loss_matrix*posterior;
[~,decision] = min(expectedRisk,[],1);
 
b = wqw(:,1:11)';
%perform the PCA wo generate the data
[Q,D,xzm,yzm]=performPCA(b);
symbols='.+x^vo';
for index =1:6
    plot(yzm(1,quality'==index+2),yzm(2,quality'==index+2),symbols(index),'DisplayName',['Class' num2str(index+3)]);
hold on;
xlabel('x1');
ylabel('x2');
grid on;
title('Priciple Component Analysis Data Distribution');
%legend('Class 4','Class 5','Class 6','Class 7','Class 8','Class 9','Class 10','Location','NorthEastOutside') ;
legend('Location','NorthEastOutside') ;
% legend 'show';
end
 
col_x = length(unique(quality));
rol_x = length(unique(decision));
for index1=1:col_x
    for index2=1:rol_x
        %calculate the prior probability and confusion matrix
%       priors(1,index2) = length(find(quality' == index2+2))/length(quality');
        i1_i2 = find(decision == (index2+2) & quality' == (index1+2));
        conf_matrix(index2,index1) = length(i1_i2)/length(find(quality' ==(index1+2))); 
    end
end
 
%calculate the posterior probability function
function [classPosteriors] = pos_cal(condition_pdf,p,N,px)
    classPosteriors = condition_pdf.*repmat(p(:,4:10)',1,N)./repmat(px,7,1);
end
 
function [matrix] = extract(index,data)
    for i = 1:length(index)
        matrix(i,1:11) = data(index(i,1),1:11);
    end
 
end

function [Q,D,xzm,yzm]=performPCA(x)
% Performs PCA on real-vector-valued data.
% Receives input x with size nxN where n is the dimensionality of samples
% and N is the number of vector-valued samples.
% Returns Q, which is an orthogonal matrix that contains in its columns the
% PCA projection vectors ordered from first to last; D, which is a diagonal
% matrix that contains the variance of each principal component
% corresponding to these projection vectors. Zero-mean version of the
% samples and zero-mean principal component projections are also returned
% in xzm and yzm, respectively. 
 
[n,N]=size(x);
% Sample-based estimates of mean vector and covariance matrix
muhat = mean(x,2); % Estimate the mean vector using the samples
Sigmahat = cov(x'); % Estimate the covariance matrix using the samples
% Subtract the estimated mean vector to make the data 0-mean
xzm = x - muhat*ones(1,N); % Obtain zero-mean sample set
% Get the eigenvectors (in Q) and eigenvalues (in D) of the
% estimated covariance matrix
[Q,D] = eig(Sigmahat);
% Sort the eigenvalues from large to small, reorder eigenvectors
% accordingly as well.
[d,ind] = sort(diag(D),'descend');
Q = Q(:,ind); D = diag(d);
yzm = Q'*xzm; % Principal components of x (zero-mean)
end 
 

function px = evalGaussianPDF(x,mu,Sigma)
% x should have n-dimensional N vectors in columns
n = size(x,1); % data vectors have n-dimensions
N = size(x,2); % there are N vector-valued samples
C = (2*pi)^(-n/2)*det(Sigma)^(-1/2); % normalization constant
a = x-repmat(mu,1,N); b = inv(Sigma)*a;
% a,b are preparatory random variables, in an attempt to avoid a for loop
px = C*exp(-0.5*sum(a.*b,1)); % px is a row vector that contains p(x_i) values
end


