clear; close all; clc;
%% Initialization
load Testdata
L=15; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x; % create spatial coordinates
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k); % create frequency coordinates and shift
[X,Y,Z]=meshgrid(x,y,z); % meshgrid into 3D
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

%% Averaging spectrum
sum = zeros(size(n,n,n)); % preallocate summation matrix
for j=1:20
    Un(:,:,:)=reshape(Undata(j,:),n,n,n); % reshape data back to 3D form
    Unt = fftn(Un); % n-dimensional fft
    sum = sum+Unt;  % adding all measurements
end
ave = abs(sum)/j; % average the sum
aves = fftshift(ave); % fftshift to match freq coordinates

thresh_perc = 0.6; % determine the normalized threshold for signature ID

% plot the volume surface at preset threshold
close all, isosurface(Kx,Ky,Kz,abs(aves),thresh_perc*max(max(max(abs(aves)))))
axis([-10 10 -10 10 -10 10]), grid on,
drawnow
pause(2)

% estimate the location of signature frequency
kx0 = 2; ky0 = -1; kz0 = 0; % (2,-1,0)


%% Filtering
tau = 0.3; % determine Gaussian bandwidth
filter = exp(-((Kx-kx0).^2+(Ky-ky0).^2+(Kz-kz0).^2)*tau); % generate Gaussian filter centered at signature frequency
traj = zeros(20,3); % preallocate marble trajectory
for j = 1:20
    Un(:,:,:)=reshape(Undata(j,:),n,n,n);
    Unt = fftn(Un);
    Unts = fftshift(Unt); % shift the transformed data to align w/ filter
    Untfs = filter.*Unts; % apply filter
    Untf = ifftshift(Untfs); % shift back to matlab default
    Unf = ifftn(Untf); % inverse fft back to spatial domain
    close all, isosurface(X,Y,Z,abs(Unf),0.4) % surface plot the filtered data
    [f,v] = isosurface(X,Y,Z,abs(Unf),.4);
    traj(j,:) = mean(v,1);
    axis([-20 20 -20 20 -20 20]), grid on,
    drawnow
    pause(1)
end
plot3(traj(:,1),traj(:,2),traj(:,3));
axis([-20 20 -20 20 -20 20]), grid on



%% ID focal point
focus = traj(end,:); % acoustic wave focus at the 20th measurement location



