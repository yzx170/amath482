clear; close all; clc

%% Loading data

load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');
load('cam1_2.mat');
load('cam2_2.mat');
load('cam3_2.mat');
load('cam1_3.mat');
load('cam2_3.mat');
load('cam3_3.mat');
load('cam1_4.mat');
load('cam2_4.mat');
load('cam3_4.mat');


%% Test 1

% Camera 1
data = vidFrames1_1;
centroid_1 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 250);
    % filter stay out zone
    ind(:,[1:300 400:end]) = 0;
    ind(1:200,:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_1(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_1(j,2),480-centroid_1(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 2
data = vidFrames2_1;
centroid_2 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 250);
    % filter stay out zone
    ind(:,[]) = 0;
    ind([],:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_2(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_2(j,2),480-centroid_2(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 3
data = vidFrames3_1;
centroid_3 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 245);
    % filter stay out zone
    ind(:,1:200) = 0;
    ind([],:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_3(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_3(j,2),480-centroid_3(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Alignment
TF = islocalmax(centroid_1(:,1),'MinProminence',20);
Ind = find(TF);
centroid_1_t = centroid_1(Ind(1):end,:);
TF = islocalmax(centroid_2(:,1),'MinProminence',20);
Ind = find(TF);
centroid_2_t = centroid_2(Ind(1):end,:);
TF = islocalmax(centroid_3(:,2),'MinProminence',20); % D1 D2 swapped due to tilt of camera 3
Ind = find(TF);
centroid_3_t = centroid_3(Ind(1):end,:);

% Truncation
p_end = min([size(centroid_1_t,1) size(centroid_2_t,1) size(centroid_3_t,1)]);
centroid_1_t = centroid_1_t(1:p_end,:);
centroid_2_t = centroid_2_t(1:p_end,:);
centroid_3_t = centroid_3_t(1:p_end,:);
% % plot the centroid from 3 cameraa
% plot(1:p_end,centroid_1_t,1:p_end,centroid_2_t,1:p_end,centroid_3_t);
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')

% PCA
combined = [centroid_1_t centroid_2_t centroid_3_t]';
n = size(combined,2);
mn = mean(combined,2);
deviation = combined - repmat(mn,1,n); % calculate variance from mean
[U,~,~] = svd(deviation);
new = U'*deviation; % reconstruct matrix w/ new basis
% Covariance matrix
C_new = cov(new');
PCA_1 = diag(C_new);
DOF_1 = sum(PCA_1 > 400);

% % Reconstruct
% subplot((DOF_1+1),1,1)
% plot(1:n,deviation)
% title('Original Data')
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')
% for j = 1:DOF_1
%     recon = U(:,j)*S(j,j)*V(:,j)';
%     subplot((DOF_1+1),1,j+1)
%     plot(1:n,recon)
%     title(strcat('Mode',32,num2str(j)))
%     xlabel('Frames')
%     ylabel('Coordinates')
% end


%% Test 2

% Camera 1
data = vidFrames1_2;
centroid_1 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 245);
    % filter stay out zone
    ind(:,[1:300 400:end]) = 0;
    ind(1:200,:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_1(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_1(j,2),480-centroid_1(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 2
data = vidFrames2_2;
centroid_2 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 245);
    % filter stay out zone
    ind(:,[]) = 0;
    ind([],:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_2(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_2(j,2),480-centroid_2(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 3
data = vidFrames3_2;
centroid_3 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 240);
    % filter stay out zone
    ind(:,1:200) = 0;
    ind(1:200,:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_3(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_3(j,2),480-centroid_3(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Alignment
TF = islocalmax(centroid_1(:,1),'MinProminence',20);
Ind = find(TF);
centroid_1_t = centroid_1(Ind(1):end,:);
TF = islocalmax(centroid_2(:,1),'MinProminence',20);
Ind = find(TF);
centroid_2_t = centroid_2(Ind(1):end,:);
TF = islocalmax(centroid_3(:,2),'MinProminence',20);
Ind = find(TF);
centroid_3_t = centroid_3(Ind(1):end,:);

% Truncation
p_end = min([size(centroid_1_t,1) size(centroid_2_t,1) size(centroid_3_t,1)]);
centroid_1_t = centroid_1_t(1:p_end,:);
centroid_2_t = centroid_2_t(1:p_end,:);
centroid_3_t = centroid_3_t(1:p_end,:);
% % plot the centroid from 3 cameraa
% plot(1:p_end,centroid_1_t,1:p_end,centroid_2_t,1:p_end,centroid_3_t);
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')

% PCA
combined = [centroid_1_t centroid_2_t centroid_3_t]';
n = size(combined,2);
mn = mean(combined,2);
deviation = combined - repmat(mn,1,n); % calculate variance from mean
[U,~,~] = svd(deviation);
new = U'*deviation; % reconstruct matrix w/ new basis
% Covariance matrix
C_new = cov(new');
PCA_2 = diag(C_new);
DOF_2 = sum(PCA_2 > 400);
P_2 = sum(PCA_2(1:DOF_2))/sum(PCA_2);

% %% filter test
% 
% test = centroid_1-mean(centroid_1,1);
% n = size(test,1);
% t = 1:n;
% test_t = abs(fft(test));
% f = [0:(n)/2-1 -(n)/2:-1];
% test_ts = fftshift(test_t);
% fs = fftshift(f);
% 
% a = .005;
% g = exp(-a*(f).^2)';
% % plot(fftshift(g))
% % plot(fs,test_ts);
% 
% test_tf(:,1) = test_t(:,1).*g;
% test_tf(:,2) = test_t(:,2).*g;
% test_f = ifft(test_tf);
% 
% subplot(2,1,1)
% plot(t,test);
% subplot(2,1,2)
% plot(t,test_f);

% % Reconstruct
% subplot((DOF_2+1),1,1)
% plot(1:n,deviation)
% title('Original Data')
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')
% for j = 1:DOF_2
%     recon = U(:,j)*S(j,j)*V(:,j)';
%     subplot((DOF_2+1),1,j+1)
%     plot(1:n,recon)
%     title(strcat('Mode',32,num2str(j)))
%     xlabel('Frames')
%     ylabel('Coordinates')
% end
 

%% Test 3

% Camera 1
data = vidFrames1_3;
centroid_1 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 250);
    % filter stay out zone
    ind(:,[1:300 400:end]) = 0;
    ind(1:200,:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_1(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_1(j,2),480-centroid_1(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 2
data = vidFrames2_3;
centroid_2 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 250);
    % filter stay out zone
    ind(:,[]) = 0;
    ind([],:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_2(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_2(j,2),480-centroid_2(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 3
data = vidFrames3_3;
centroid_3 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 240);
    % filter stay out zone
    ind(:,1:200) = 0;
    ind([],:) = 0;
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_3(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_3(j,2),480-centroid_3(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Alignment
TF = islocalmax(centroid_1(:,1),'MinProminence',20);
Ind = find(TF);
centroid_1_t = centroid_1(Ind(1):end,:);
TF = islocalmax(centroid_2(:,1),'MinProminence',20);
Ind = find(TF);
centroid_2_t = centroid_2(Ind(1):end,:);
TF = islocalmax(centroid_3(:,2),'MinProminence',20); % swap b/t D1 and D2
Ind = find(TF);
centroid_3_t = centroid_3(Ind(1):end,:);

% Truncation
p_end = min([size(centroid_1_t,1) size(centroid_2_t,1) size(centroid_3_t,1)]);
centroid_1_t = centroid_1_t(1:p_end,:);
centroid_2_t = centroid_2_t(1:p_end,:);
centroid_3_t = centroid_3_t(1:p_end,:);
% % plot the centroid from 3 cameraa
% plot(1:p_end,centroid_1_t,1:p_end,centroid_2_t,1:p_end,centroid_3_t);
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')

% PCA
combined = [centroid_1_t centroid_2_t centroid_3_t]';
n = size(combined,2); % number of observations
mn = mean(combined,2);
deviation = combined - repmat(mn,1,n); % calculate variance from mean
[U,~,~] = svd(deviation);
new = U'*deviation; % reconstruct matrix w/ new basis
% Covariance matrix
C_new = cov(new');
PCA_3 = diag(C_new);
DOF_3 = sum(PCA_3 > 400);
P_3 = sum(PCA_3(1:DOF_3))/sum(PCA_3);

% % Reconstruct
% subplot((DOF_3+1),1,1)
% plot(1:n,deviation)
% title('Original Data')
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')
% for j = 1:DOF_3
%     recon = U(:,j)*S(j,j)*V(:,j)';
%     subplot((DOF_3+1),1,j+1)
%     plot(1:n,recon)
%     title(strcat('Mode',32,num2str(j)))
%     xlabel('Frames')
%     ylabel('Coordinates')
% end

%% Test 4

% Camera 1
data = vidFrames1_4;
centroid_1 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 245);
    % filter stay out zone
    ind(:,[1:300 400:end]) = 0;
    ind(1:200,:) = 0;
    % reduce the area tracked on can sidewall
    [Row,Col] = find(ind);
    if isempty(Row)
    else
        ind((min(Row)+5):end,:) = 0;
    end
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_1(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_1(j,2),480-centroid_1(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 2
data = vidFrames2_4;
centroid_2 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 250);
    % filter stay out zone
    ind(:,[]) = 0;
    ind(1:150,:) = 0;
    % reduce the area tracked on can sidewall
    [Row,Col] = find(ind);
    if isempty(Row)
    else
        ind((min(Row)+5):end,:) = 0;
    end
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_2(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_2(j,2),480-centroid_2(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Camera 3
data = vidFrames3_4;
centroid_3 = zeros(size(data,4),2);
for j = 1:size(data,4)
    % initialization
    temp = data(:,:,:,j);
    temp_gray = rgb2gray(temp);
    
    % locate beacon
    ind = (temp_gray > 235);
    % filter stay out zone
    ind(:,1:200) = 0;
    ind([],:) = 0;
%     % reduce the area tracked on can sidewall
%     [Row,Col] = find(ind);
%     if isempty(Row)
%     else
%         ind(:,(min(Col)+5):end) = 0;
%     end
    [D1,D2] = find(ind); % D1: top to bottom; D2: left to right

    centroid_3(j,:) = mean([D1 D2],1);

    % plot tracked pixels
%     temp_gray(ind) = 0;
%     imshow(temp_gray)
%     title('Tracked Pixels Labeled Black')

    % plot trajectory of centroid
%     scatter(centroid_3(j,2),480-centroid_3(j,1))
%     set(gca,'Ylim',[0 480],'Xlim',[0 640])
%     xlabel('Horizontal Location')
%     ylabel('Vertical Location')
%     drawnow
end

% Alignment
TF = islocalmax(centroid_1(:,1),'MinProminence',20);
Ind = find(TF);
centroid_1_t = centroid_1(Ind(1):end,:);
TF = islocalmax(centroid_2(:,1),'MinProminence',20);
Ind = find(TF);
centroid_2_t = centroid_2(Ind(1):end,:);
TF = islocalmax(centroid_3(:,2),'MinProminence',20); % swap b/t D1 and D2
Ind = find(TF);
centroid_3_t = centroid_3(Ind(1):end,:);

% Truncation
p_end = min([size(centroid_1_t,1) size(centroid_2_t,1) size(centroid_3_t,1)]);
centroid_1_t = centroid_1_t(1:p_end,:);
centroid_2_t = centroid_2_t(1:p_end,:);
centroid_3_t = centroid_3_t(1:p_end,:);
% % plot the centroid from 3 cameraa
% plot(1:p_end,centroid_1_t,1:p_end,centroid_2_t,1:p_end,centroid_3_t);
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')

% PCA
combined = [centroid_1_t centroid_2_t centroid_3_t]';
n = size(combined,2); % number of observations
mn = mean(combined,2);
deviation = combined - repmat(mn,1,n); % calculate variance from mean
[U,~,~] = svd(deviation);
new = U'*deviation; % reconstruct matrix w/ new basis
% Covariance matrix
C_new = cov(new');
PCA_4 = diag(C_new);
DOF_4 = sum(PCA_4 > 400);
P_4 = sum(PCA_4(1:DOF_4))/sum(PCA_4);

% % Reconstruct
% subplot((DOF_4+1),1,1)
% plot(1:n,deviation)
% title('Original Data')
% xlabel('Frames')
% ylabel('Coordinates (X and Y)')
% legend('Cam 1 - Y','Cam 1 - X','Cam 2 - Y','Cam 2 - X','Cam 3 - Y','Cam 3 - X','location','eastoutside')
% for j = 1:DOF_4
%     recon = U(:,j)*S(j,j)*V(:,j)';
%     subplot((DOF_4+1),1,j+1)
%     plot(1:n,recon)
%     title(strcat('Mode',32,num2str(j)))
%     xlabel('Frames')
%     ylabel('Coordinates')
% end


%% Summary

PCA = [PCA_1 PCA_2 PCA_3 PCA_4];
DOF = [DOF_1 DOF_2 DOF_3 DOF_4];
% Power = [P_1 P_2 P_3 P_4];
