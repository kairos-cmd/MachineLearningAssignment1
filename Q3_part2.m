clear all;
close all;
x_test = readmatrix('X_test.txt');
x_train = readmatrix('X_train.txt');
label_test = readmatrix("y_test.txt");
label_train = readmatrix("y_train.txt");
x = [x_test;x_train]';
label = [label_test;label_train]';
N = size(x,2);
n = size(x,1);
C = 6;

alpha = 0.1;
sigmaTotal = cov(x'); %for regularization

for i = 1:C
    p(i) = sum(label == i)/N;
    mu(:,i) = mean(x(:,label==i),2);
    sigma(:,:,i) = cov(x(:,label==i)');
    sigma(:,:,i) = sigma(:,:,i) + eye(size(sigma,1))*alpha*trace(sigmaTotal)/rank(sigmaTotal);
end

for i=1:C
    if sum(isnan(sigma(:,:,i))) == 0
        pxgivenl(i,:) = mvnpdf(x',mu(:,i)',sigma(:,:,i))';
    else
        pxgivenl(i,:) = zeros(1,4898);
    end
end

px =p*pxgivenl;
plgivenx = pxgivenl.*repmat(p',1,N)./repmat(px,C,1);

lossMatrix = ones(C,C)-eye(C);
expectedRisks = lossMatrix*plgivenx;
[~,decisions] = min(expectedRisks,[],1);

countError = sum(label~=decisions);
pE = countError/N;

for i = 1:C
    for j = 1:C
        if sum(isnan(sigma(:,:,j))) == 0
            confusionMatrix(i,j) = sum(decisions == i & label == j)/sum(label==j);
        else
            confusionMatrix(i,j) = 0
        end
    end
end

muHat = mean(x,2);

xzm = x - muHat*ones(1,N);

[Q,D] = eig(sigmaTotal);
[d,ind] = sort(diag(D),'descend');
Q = Q(:,ind);
D = diag(d);

y = Q(:,1:3)'*xzm;

percentVar = trace(D(1:3,1:3))/trace(D);

figure
scatter3(y(1,label==1),y(2,label == 1),y(3,label==1),'b*','DisplayName','Class 1')
hold on
scatter3(y(1,label==2),y(2,label == 2),y(3,label==2),'g*','DisplayName','Class 2')
scatter3(y(1,label==3),y(2,label == 3),y(3,label==3),'m*','DisplayName','Class 3')
scatter3(y(1,label==4),y(2,label == 4),y(3,label==4),'c*','DisplayName','Class 4')
scatter3(y(1,label==5),y(2,label == 5),y(3,label==5),'r*','DisplayName','Class 5')
scatter3(y(1,label==6),y(2,label == 6),y(3,label==6),'k*','DisplayName','Class 6')
title('PCA on Human Activity Dataset')
xlabel('y1')
ylabel('y2')
zlabel('y3')
legend
hold off

counterDist = 0;
counterSig = 0;
for i = 1: C
    if sum(isnan(sigma(:,:,i)))==0
        counterSig = counterSig + 1;
        standardDev(i) = trace(sqrt(sigma(:,:,i)))/size(sigma,1);
    end
    for j = 1:C
        if sum(isnan(sigma(:,:,j)))==0
            if i<j
                counterDist = counterDist+1;
                distances(i) = sqrt(sum((x(:,i)-x(:,j)).^2));
            end
        end
    end
end
averageDistance = sum(distances)/counterDist;
averageStdDev = sum(standardDev)/counterSig;


