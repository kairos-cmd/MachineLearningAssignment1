clear all; close all; clc;
n=4;
N_sample=10000;

%set means and covariance
mu(:,1)=[-1/2;-1/2;-1/2;-1/2];
Sigma(:,:,1)=1/4*[2 -0.5 0.3 0; -0.5 1 -0.5 0; 0.3 -0.5 1 0; 0 0 0 2];
mu(:,2)=[1;1;1;1];
Sigma(:,:,2)=[1 0.3 -0.2 0; 0.3 2 0.3 0; -0.2 0.3 1 0; 0 0 0 3] ;

%set class prior probability and true label
p=[0.35,0.65];
label = rand(1,N_sample)>=p(1);
N0=sum(label==0);
N1=sum(label==1);

%plot the samples in each class
r=zeros(n,N_sample);
r(:,label==0)=mvnrnd(mu(:,1),Sigma(:,:,1),N0)';
r(:,label==1)=mvnrnd(mu(:,2),Sigma(:,:,2),N1)';

%plot true class labels
figure(1)
plot(r(1,label==0),r(2,label==0),'o',r(1,label==1),r(2,label==1),'+');
title('Class 0 and Class 1 True Class Labels')
xlabel('x_1'),ylabel('x_2')
legend('Class 0','Class 1')

%Calculate the discriminant Scores
discriminantScore =log(evalGaussian(r,mu(:,2),Sigma(:,:,2))./evalGaussian(r,mu(:,1),Sigma(:,:,1)));
gamma = log(sort(discriminantScore(discriminantScore>=0)));
mid_gamma = [gamma(1)-1 gamma(1:end-1)+diff(gamma)./2 gamma(length(gamma))+1];
 
%make decision for every threshold and calculate error values
for i=1:length(mid_gamma)
    decision=(discriminantScore>=mid_gamma(i));
    pFA(i)=sum(decision==1 & label==0)/N0;
    pCD(i)=sum(decision==1 & label==1)/N1;
    pE(i)=pFA(i)*p(1)+(1-pCD(i))*p(2);
end

%find minimum error and corresponding threshold
[min_error,min_index] = min(pE);
min_decision=(discriminantScore>=mid_gamma(min_index));
min_FA = pFA(min_index);
min_CD=pCD(min_index);

%find ideal minimum error
ideal_decision=(discriminantScore>=log(p(1)/p(2)));
ideal_pFA = sum(ideal_decision == 1&label == 0)/N0;
ideal_pCD = sum(ideal_decision==1&label==1)/N1;
ideal_error = ideal_pFA*p(1)+(1-ideal_pCD)*p(2);

%plot the ROC curve
figure(2);
plot(pFA,pCD,'-',min_FA,min_CD,'o',ideal_pFA,ideal_pCD,'g+');
title('Minimum Expected Risk ROC Curve');
legend('ROC Curve','Calculated Min Error','Theoretical Min Error');
xlabel('P_{False Alarm}');ylabel('P_{Correct Detection}')

%Output every results
fprintf('\n<strong>Theoretical Results</strong>\n');
fprintf('Minimum probability of error: %.2f%%\nThreshold Value : %.2f\n',ideal_error*100,p(1)/p(2));

fprintf('\n<strong>Calculated Results</strong>\n');
fprintf('Minimum probability of error:%.2f%%\nThreshold Value: %.2f \n\n',min_error*100 , exp (mid_gamma(min_index)));

%PartB
sigma_nb=[1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1];
discriminantScore_nb=log(evalGaussian(r,mu(:,2),sigma_nb)./evalGaussian(r,mu(:,1),sigma_nb));
gamma_nb = log(sort(discriminantScore_nb(discriminantScore>=0)));
mid_gamma_nb = [gamma_nb(1)-1 gamma_nb(1:end-1)+diff(gamma_nb)./2 gamma_nb(length(gamma_nb))+1];

%Make decision for every threshold
for i = 1:length(mid_gamma_nb)
    decision_nb=(discriminantScore_nb>=mid_gamma_nb(i));
    pFA_nb(i)=sum(decision_nb==1&label==0)/N0;
    pCD_nb(i)=sum(decision_nb ==1 &label==1)/N1;
    pE_nb(i)=pFA_nb(i)*p(1)+(1-pCD_nb(i))*p(2);
end
%find minimum error and corresponding threshold
[min_error_nb,min_index_nb] = min(pE_nb);
min_decision_nb=(discriminantScore_nb>=mid_gamma_nb(min_index));
min_FA_nb = pFA_nb(min_index_nb);
min_CD_nb=pCD_nb(min_index_nb);

%find ideal minimum error
ideal_decision_nb=(discriminantScore_nb>=log(p(1)/p(2)));
ideal_pFA_nb = sum(ideal_decision_nb == 1&label == 0)/N0;
ideal_pCD_nb = sum(ideal_decision_nb==1&label==1)/N1;
ideal_error_nb = ideal_pFA_nb*p(1)+(1-ideal_pCD_nb)*p(2);

figure(3);
plot(pFA_nb,pCD_nb,'-',min_FA_nb,min_CD_nb,'o');
title('Native Bayesian ROC Curve');
legend('Native Bayesian ROC Curve','Native Bayesian Min Error');
xlabel('P_{False Alarm}');ylabel('P_{Correct Detection}')

fprintf('\n<strong>Theoretical Results</strong>\n');
fprintf('Native Bayesian Minimum probability of error: %.2f%%\nThreshold Value : %.2f\n',ideal_error_nb*100,p(1)/p(2));

fprintf('\n<strong>Calculated Results</strong>\n');
fprintf('Native Bayesian Minimum probability of error:%.2f%%\nThreshold Value: %.2f \n\n',min_error_nb*100 , exp (mid_gamma_nb(min_index_nb)));

%Part C
x0=x(:,label==0);
x1=x(:,label==1);
mu0_hat=mean(x0);
mu1_hat=mean(x1);
Sigma0_hat=cov(x0);
Sigma1_hat=cov(x1);

%Compute scatter matrices
Sb=(mu0_hat-mu1_hat)*(mu0_hat-mu1_hat)';
Sw=Sigma0_hat+Sigma1_hat;
%Eigen decompostion to generate WLDA
[V,D]=eig(inv(Sw)*Sb);
[~,ind]=max(diag(D));
w=V(:,ind);
y=w'*x;
w=sign(mean(y(find(label==1))-mean(y(find(label==0)))))*w;

y=sign(mean(y(find(label==1))-mean(y(find(label==0)))))*y;
%Evaluate for different taus
tau=[min(y)-0.1 sort(y)+0.1];
for ind=1:length(tau)
decision=y>tau(ind);
Num_pos_LDA(ind)=sum(decision);
pFP_LDA(ind)=sum(decision==1 & label==0)/NL(1);
pTP_LDA(ind)=sum(decision==1 & label==1)/NL(2);
pFN_LDA(ind)=sum(decision==0 & label==1)/NL(2);
pTN_LDA(ind)=sum(decision==0 & label==0)/NL(1);
pFE_LDA(ind)=(sum(decision==0 & label==1)...
+ sum(decision==1 & label==0))/(NL(1)+NL(2));
end

%Estimated Minimum Error
[min_pFE_LDA, min_pFE_ind_LDA]=min(pFE_LDA);
minTAU_LDA=tau(min_pFE_ind_LDA);
min_FP_LDA=pFP_LDA(min_pFE_ind_LDA);
min_TP_LDA=pTP_LDA(min_pFE_ind_LDA);
%Plot results
figure;
plot(y(label==0),zeros(1,NL(1)),'X','DisplayName','Label 0');
hold all;
plot(y(label==1),ones(1,NL(2)),'X','DisplayName','Label 1');
ylim([-1 2]);
plot(repmat(tau(min_pFE_ind_LDA),1,2),ylim,'m--',...
'DisplayName','Tau for Min. Error','LineWidth',2);
grid on;
xlabel('y');
title('Fisher LDA Projection of Data');
legend 'show';

figure;
plot(pFP_LDA,pTP_LDA,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP_LDA,min_TP_LDA,'o','DisplayName',...
'Estimated Min. Error','LineWidth',2);
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;
figure;
plot(tau,pFE_LDA,'DisplayName','Errors','LineWidth',2);
hold on;
plot(tau(min_pFE_ind_LDA),pFE_LDA(min_pFE_ind_LDA),'ro',...
'DisplayName','Minimum Error','LineWidth',2);
xlabel('Tau');
ylabel('Proportion of Errors');
title('Probability of Error vs. Tau for Fisher LDA')
grid on;
legend 'show';
fprintf('Estimated for LDA: Tau=%1.2f, Error=%1.2f%%\n',...
minTAU_LDA,100*min_pFE_LDA);

%Function from: Prof.Deniz
function g=evalGaussian(x,mu,Sigma)
%Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N]=size(x);
C=((2*pi)^n*det(Sigma))^(-1/2);%coefficient
E=-0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);%exponent
g=C*exp(E);%final gaussian evaluationend
end
function [x,labels] = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % nality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1));
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    labels(ind)=m-1;
end
end
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z = randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
