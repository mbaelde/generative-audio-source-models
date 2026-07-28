addpath(genpath('Statistics'))
%% 1D case
mu = [-3,2,5];
sigma = [0.7,1,0.5];
p = [0.34,0.34,0.32];

x = -10:0.01:10;
density = mixture_normpdf(x,mu,sigma,p);

n_segments = 30;
th = 1e-5;
model.mu = mu;
model.sigma = sigma;
model.p = p;

[segments, approx] = discretize_univariate_normpdf(n_segments, model, th);

figure(1)
clf
plot(x,density)
hold on
for k = 1:n_segments
    plot([segments(k), segments(k)],[0,max(density)],'-r')
end
for k = 1:n_segments-1
    plot([segments(k), segments(k)],[0,approx(k)],'-k')
    plot([segments(k+1), segments(k+1)],[0,approx(k)],'-k')
    plot([segments(k), segments(k+1)],[approx(k),approx(k)],'-k')
end

%% 2D case
mu = [-2,-2;
       2, 2];
sigma(:,:,1) = [1,0.9;
                0.9,1];
sigma(:,:,2) = [ 1  ,-0.2;
                -0.2,1   ];
p = [0.5,0.5];

xmin = -5;
xmax = 5;
xgrid = 0.05;

ymin = -5;
ymax = 5;
ygrid = 0.05;

[X,Y] = meshgrid((xmin:xgrid:xmax),(ymin:ygrid:ymax));
density = mixture_mvnpdf([X(:),Y(:)],mu,sigma,p);

density_reshape = reshape(density,size(X));

n_x = 20;
p_x = floor(size(X,1) / n_x);
n_y = 20;
p_y = floor(size(X,2) / n_y);

approx = zeros(n_x,n_y);
X_a = zeros(n_x+1,n_y+1);
Y_a = zeros(n_x+1,n_y+1);
for i = 1:n_x
    for j = 1:n_y
        approx(i,j) = mean(mean(density_reshape((i-1)*p_x+1:i*p_x , (j-1)*p_y+1:j*p_y)));
    end
end
X_a = mean(reshape(X(1,1:end-1),[length(X(1,1:end-1))/n_x,n_x]));
Y_a = mean(reshape(Y(1:end-1,1)',[length(Y(1:end-1,1))/n_y,n_y]));
delta_x = X_a(2)-X_a(1);
delta_y = Y_a(2)-Y_a(1);
X_a = repmat([X_a - delta_x/2, X_a(end)+delta_x/2],[n_x+1,1]);
Y_a = repmat([Y_a - delta_y/2, Y_a(end)+delta_y/2]',[1,n_y+1]);

figure(1)
clf
surf(X,Y,reshape(density,size(X)))
figure(2)
clf
width = 1;
bar3(approx,width)

mcdf = zeros(n_x,n_y);
for i = 1:n_x
    for j = 1:n_y
        mcdf(i,j) = mixture_mvncdf([X_a(i,j),Y_a(i,j)],[X_a(i+1,j+1),Y_a(i+1,j+1)], mu,sigma,p);
    end
end

figure(3)
clf
surf(X_a(1:end-1,1:end-1),Y_a(1:end-1,1:end-1),mcdf)

%
N = 1000;
samples = mixture_mvnrnd(N, mu, sigma, p);

figure(4) 
clf
plot(samples(:,1),samples(:,2),'*')

[H,C] = hist3(samples,[n_x,n_y]);

figure(5)
clf
bar3(H)

%approx_norm = 20^2 * approx ./ sum(approx(:));
for i = 1:1000
aux = H.*log(mcdf);
likelihood = sum(aux(:));

likelihood_complete = sum(log(mixture_mvnpdf(samples,mu,sigma,p)));
end
%% 3 case
clear
mu = [-1,-1,-1;
      2,2,2];
sigma(:,:,1) = eye(3);
sigma(:,:,2) = [1,0.2,0.3;
                  0.2,1,0.2;
                  0.3,0.2,1];
p = [0.5,0.5];

xmin = -5;
xmax = 5;
xgrid = 0.1;

ymin = -5;
ymax = 5;
ygrid = 0.1;

zmin = -5;
zmax = 5;
zgrid = 0.1;

[X,Y,Z] = meshgrid((ymin:ygrid:ymax),(xmin:xgrid:xmax),(zmin:zgrid:zmax));
density = mixture_mvnpdf([X(:),Y(:),Z(:)],mu,sigma,p);

density_reshape = reshape(density,size(X));

n_x = 10;
p_x = floor(size(X,1) / n_x);
n_y = 10;
p_y = floor(size(X,2) / n_y);
n_z = 10;
p_z = floor(size(X,3) / n_z);

approx = zeros(n_x,n_y,n_z);
for i = 1:n_x
    for j = 1:n_y
        for k = 1:n_z
            aux = density_reshape((i-1)*p_x+1:i*p_x , (j-1)*p_y+1:j*p_y , (k-1)*p_z+1:k*p_z);
            approx(i,j,k) = mean(aux(:));
        end
    end
end
X_a = mean(reshape(X(1,1:end-1,1),[length(X(1,1:end-1,1))/n_x,n_x]));
Y_a = mean(reshape(Y(1:end-1,1,1)',[length(Y(1:end-1,1,1))/n_y,n_y]));
aux_Z = reshape(Z(:,1,:),size(X(:,:,1)));
Z_a = mean(reshape(aux_Z(1,1:end-1,1),[length(Z(1,1:end-1,1))/n_z,n_z]));
delta_x = X_a(2)-X_a(1);
delta_y = Y_a(2)-Y_a(1);
delta_z = Z_a(2)-Z_a(1);
X_a = repmat([X_a - delta_x/2, X_a(end)+delta_x/2],[n_x+1,1,n_x+1]);
Y_a = repmat([Y_a - delta_y/2, Y_a(end)+delta_y/2]',[1,n_y+1,n_y+1]);
Z_a = permute(repmat([Z_a - delta_z/2, Z_a(end)+delta_z/2],[n_y+1,1,n_y+1]),[1,3,2]);


FV = isosurface(X, Y, Z, density_reshape, 0.001);
figure(10)
clf
patch(FV,'FaceColor',[0.1 0.2 0.4],'EdgeColor','none','FaceLighting','phong','AmbientStrength',0.5)
camlight
axis equal;

mcdf = zeros(n_x,n_y,n_z);
for i = 1:n_x
    for j = 1:n_y
        for k = 1:n_z
            mcdf(i,j,k) = mixture_mvncdf([X_a(i,j,k),Y_a(i,j,k),Z_a(i,j,k)],[X_a(i+1,j+1,k+1),Y_a(i+1,j+1,k+1),Z_a(i+1,j+1,k+1)], mu,sigma,p);
        end
    end
    progressbar(i,n_x)
end
l_mcdf = log(mcdf);
%
N = 205;
samples = mixture_mvnrnd(N, mu, sigma, p);

figure(40) 
clf
plot3(samples(:,1),samples(:,2),samples(:,3),'*')

[H] = histcn(samples,n_x-1,n_y-1,n_z-1);

H_u = H(:);
l_mcdf_u = l_mcdf(:);
r_l_mcdf_u = repmat(l_mcdf_u',[70000,1]);

%approx_norm = 20^2 * approx ./ sum(approx(:));
% 
% for i = 1:1000
% aux = H.*log(mcdf);
% likelihood = sum(aux(:));
% 
% likelihood_complete = sum(log(mixture_mvnpdf(samples,mu,sigma,p)));
% end

%%
model = feature{2};
mu = model.mu;
sigma = model.Sigma;
p = model.ComponentProportion;

xmin = 0;
xmax = 8000;
xgrid = 50;

ymin = -1;
ymax = 1;
ygrid = 0.01;

zmin = -1;
zmax = 1;
zgrid = 0.01;

[X,Y,Z] = meshgrid((ymin:ygrid:ymax),(xmin:xgrid:xmax),(zmin:zgrid:zmax));
density = mixture_mvnpdf([X(:),Y(:),Z(:)],mu,sigma,p);

density_reshape = reshape(density,size(X));

FV = isosurface(X, Y, Z, density_reshape, 0.000001);
figure(10)
clf
patch(FV,'FaceColor',[0.1 0.2 0.4],'EdgeColor','none','FaceLighting','phong','AmbientStrength',0.5)
camlight
axis equal;

n_x = 10;
p_x = floor(size(X,1) / n_x);
n_y = 10;
p_y = floor(size(X,2) / n_y);
n_z = 10;
p_z = floor(size(X,3) / n_z);

X_a = mean(reshape(X(1,1:end-1,1),[length(X(1,1:end-1,1))/n_x,n_x]));
Y_a = mean(reshape(Y(1:end-1,1,1)',[length(Y(1:end-1,1,1))/n_y,n_y]));
aux_Z = reshape(Z(:,1,:),size(X(:,:,1)));
Z_a = mean(reshape(aux_Z(1,1:end-1,1),[length(Z(1,1:end-1,1))/n_z,n_z]));
delta_x = X_a(2)-X_a(1);
delta_y = Y_a(2)-Y_a(1);
delta_z = Z_a(2)-Z_a(1);
X_a = repmat([X_a - delta_x/2, X_a(end)+delta_x/2],[n_x+1,1,n_x+1]);
Y_a = repmat([Y_a - delta_y/2, Y_a(end)+delta_y/2]',[1,n_y+1,n_y+1]);
Z_a = permute(repmat([Z_a - delta_z/2, Z_a(end)+delta_z/2],[n_y+1,1,n_y+1]),[1,3,2]);

mcdf = zeros(n_x,n_y,n_z);
for i = 1:n_x
    for j = 1:n_y
        for k = 1:n_z
            mcdf(i,j,k) = mixture_mvncdf([X_a(i,j,k),Y_a(i,j,k),Z_a(i,j,k)],[X_a(i+1,j+1,k+1),Y_a(i+1,j+1,k+1),Z_a(i+1,j+1,k+1)], mu,sigma,p);
            disp(['i:',num2str(i),'; j:',num2str(j),'; k:',num2str(k)])
        end
    end
end
l_mcdf = log(mcdf);
[H] = histcn(data_to_reco,n_x-1,n_y-1,n_z-1);

H_u = H(:);
l_mcdf_u = l_mcdf(:);

aux = H_u.*l_mcdf_u;
likelihood = sum(aux(:),'omitnan');