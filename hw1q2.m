clear all;close all;clc;
%set the demension and number of samples
n = 3;
N_sample = 10000;

%class0 parameters
mu(:,1)=[1 1 1];
sigma(:,:,1)=[7 0 0;0 7 0;0 0 7];
%class1 parameters
mu(:,2)=[5 5 5];
sigma(:,:,2)=[7 0 0;0 7 0;0 0 7];
%class2 parameters(mixture of 2 Gaussians)
mu(:,3)=[9 9 9];
mu(:,4)=[13 13 13];
sigma(:,:,3)=[7 0 0;0 7 0;0 0 7];
sigma(:,:,4)=[7 0 0;0 7 0;0 0 7];
%set class prior probability
p = [0.3,0.3,0.4];
%generate true labels
gmmParameters.priors = p;
gmmParameters.meanVectors = mu;
gmmParameters.covMatrices = sigma;
[x,componentLabels] = generateDataFromGMM(N_sample, gmmParameters,0);

%calculate the size for each class
N=zeros(n,1);
for i = 1:n
   N(i,1)=length(find(componentLabels==i));
end
for i = 1:n
    if i>=1 & i<=2
        conditionpdf(i,:) = evalGaussianPDF(x,gmmParameters.meanVectors(:,i),gmmParameters.covMatrices(:,:,i)); % Evaluate p(x|L=l)
    else
        conditionpdf(i,:) =...
    0.5 * evalGaussianPDF(x,gmmParameters.meanVectors(:,i),gmmParameters.covMatrices(:,:,i))+0.5 * evalGaussianPDF(x,gmmParameters.meanVectors(:,i+1),gmmParameters.covMatrices(:,:,i+1));
    end
end
%calculate the posterior probability
px=gmmParameters.priors*conditionpdf;
postpdf=conditionpdf.*repmat(gmmParameters.priors',1,N_sample)./repmat(px,n,1);

%plot the true label
figure(1);
symbols1='.x+';
for i = 1:n
    plot3(x(1,componentLabels==i),x(2,componentLabels==i),x(3,componentLabels==i),symbols1(i),'DisplayName',['Class ' num2str(i)]);
    hold on;
end
xlabel('x1');
ylabel('x2');
zlabel('x3');
title('X Vector Distribution and True Labels')
legend('Class 1','Class 2','Class 3')

%use 0-1 loss 
loss_matrix = [0 1 1;1 0 1;1 1 0];
col_loss = size(loss_matrix,2);% return the number of columm of loss_matrix
rol_x = size(x,1);
%calculate the expected risk
expectedRisk =loss_matrix*postpdf;
%calculate the minimum risk to get the classification
[min_eprisk,decisions] = min(expectedRisk,[],1);
figure(2);
symbols2='ox*';
color='rg';
for index1=1:col_loss
    for index2=1:rol_x
        %calculate the prior probability and confusion matrix
        priors(1,index2) = length(find(componentLabels == index2))/length(componentLabels);
        i1_i2 = find(decisions == index1 & componentLabels == index2);
        conf_matrix(index1,index2) = length(i1_i2)/length(find(componentLabels==index1)); 
        if(index1==index2)
            plot3(x(1,i1_i2),x(2,i1_i2),x(3,i1_i2),strcat(symbols2(index2),color(2)),'DisplayName',['Class ' num2str(index2) ' Correct Classification']),
            hold on,
            %axis equal,
        else
            %if index1>index2
                plot3(x(1,i1_i2),x(2,i1_i2),x(3,i1_i2),strcat(symbols2(index2),color(1)),'DisplayName',['Class ' num2str(index2) ' Incorrect Classification']),
                hold on,
                %axis equal,
            %end
        end
    end
end
xlabel('x1');
ylabel('x2');
zlabel('x3');
title('X vector with Classification in 0-1 loss matrix')
legend 'show'
error_t = priors*diag(loss_matrix * conf_matrix);
err_check =  1 - priors*diag(conf_matrix);



%%==============use different loss matrix========================%%

%============use A10 loss matrix=============%

loss_matrix10 = [0 1 10;1 0 10;1 1 0];
col_loss10 = size(loss_matrix10,2);% return the number of columm of loss_matrix
rol_x10 = size(x,1);
%calculate the expected risk
expectedRisk10 =loss_matrix10*postpdf;
%calculate the minimum risk to get the classification
[min_eprisk10,decisions10] = min(expectedRisk10,[],1);
figure(3);
symbols2='ox*';
color='rg';
for index1=1:col_loss10
    for index2=1:rol_x10
        %calculate the prior probability and confusion matrix
        priors10(1,index2) = length(find(componentLabels == index2))/length(componentLabels);
        i1_i2 = find(decisions10 == index1 & componentLabels == index2);
        conf_matrix10(index1,index2) = length(i1_i2)/length(find(componentLabels==index1)); 
        if(index1==index2)
            plot3(x(1,i1_i2),x(2,i1_i2),x(3,i1_i2),strcat(symbols2(index2),color(2)),'DisplayName',['Class ' num2str(index2) ' Correct Classification']),
            hold on,
            %axis equal,
        else
            %if index1>index2
                plot3(x(1,i1_i2),x(2,i1_i2),x(3,i1_i2),strcat(symbols2(index2),color(1)),'DisplayName',['Class ' num2str(index2) ' Incorrect Classification']),
                hold on,
                %axis equal,
            %end
        end
    end
end
xlabel('x1');
ylabel('x2');
zlabel('x3');
title('X vector with Classification in A10 loss matrix')
legend 'show'
error_t10 = priors10*diag(loss_matrix10 * conf_matrix10);
err_check10 =  1 - priors10*diag(conf_matrix10);


%============use A100 loss matrix=============%

loss_matrix100 = [0 1 100;1 0 100;1 1 0];
col_loss100 = size(loss_matrix100,2);% return the number of columm of loss_matrix
rol_x100 = size(x,1);
%calculate the expected risk
expectedRisk100 =loss_matrix100*postpdf;
%calculate the minimum risk to get the classification
[min_eprisk100,decisions100] = min(expectedRisk100,[],1);
figure(4);
symbols2='ox*';
color='rg';
for index1=1:col_loss100
    for index2=1:rol_x100
        %calculate the prior probability and confusion matrix
        priors100(1,index2) = length(find(componentLabels == index2))/length(componentLabels);
        i1_i2 = find(decisions100 == index1 & componentLabels == index2);
        conf_matrix100(index1,index2) = length(i1_i2)/length(find(componentLabels==index1)); 
        if(index1==index2)
            plot3(x(1,i1_i2),x(2,i1_i2),x(3,i1_i2),strcat(symbols2(index2),color(2)),'DisplayName',['Class ' num2str(index2) ' Correct Classification']),
            hold on,
        else
            plot3(x(1,i1_i2),x(2,i1_i2),x(3,i1_i2),strcat(symbols2(index2),color(1)),'DisplayName',['Class ' num2str(index2) ' Incorrect Classification']),
            hold on,
        end
    end
end
xlabel('x1');
ylabel('x2');
zlabel('x3');
title('X vector with Classification in A100 loss matrix')
legend 'show'
error_t100 = priors100*diag(loss_matrix100 * conf_matrix100);
err_check100 =  1 - priors100*diag(conf_matrix100);





















