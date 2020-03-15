clear; close all; clc;

%% Part I - Initialization

load handel
v = y';
t = (1:length(v))/Fs;
L = t(end);
n = length(t);
k=(2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1]; % n is odd, translate into omega
ks=fftshift(k);


%% I - Basic Spectrogram

a = 1; % window width (inverse)
dt = .1; % time step
tslide=0:dt:L; 
vt_spec = zeros(length(tslide),length(v));
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vt_spec(j,:) = fftshift(abs(vgt)); 
end

figure(1)
pcolor(tslide,(ks./(2*pi)),vt_spec.') % k in terms of Hz
ylabel('Frequency (Hz)')
xlabel('Time (s)')
shading interp 
colormap(hot)


%% I - Different Window Width

figure(2)
for jj = 1:4
    a = 1*10.^(jj-1); % window width (inverse)
    dt = .1; % time step
    tslide=0:dt:L; 
    vt_spec = zeros(length(tslide),length(v));
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        vg=g.*v; 
        vgt=fft(vg); 
        vt_spec(j,:) = fftshift(abs(vgt)); 
    end
    subplot(2,2,jj);
    pcolor(tslide,(ks./(2*pi)),vt_spec.')
    title(['Gaussian Bandwidth = ' num2str(a)]);
    ylabel('Frequency (Hz)')
    xlabel('Time (s)')
    shading interp 
    colormap(hot)
end


%% I - Different Time Step

figure(3)
time = zeros(1,3);
for jj = 1:3
    a = 100; % window width (inverse)
    dt = 1*10^(-jj+1); % time step
    tslide=0:dt:L; 
    vt_spec = zeros(length(tslide),length(v));
    tic
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        vg=g.*v; 
        vgt=fft(vg); 
        vt_spec(j,:) = fftshift(abs(vgt)); 
    end
    time(jj) = toc;
    subplot(2,2,jj);
    pcolor(tslide,(ks./(2*pi)),vt_spec.')
    title(['Time Step = ' num2str(dt) ' Computation Time ' num2str(time(jj)) 's']);
    ylabel('Frequency (Hz)')
    xlabel('Time (s)')
    shading interp 
    colormap(hot)
end


%% I - Different Gabor Window

figure(4)
% widths in three methods are determined to have similar areas
% Gaussian
a = 1; % window width (inverse)
dt = .1; % time step
tslide=0:dt:L; 
vt_spec = zeros(length(tslide),length(v));
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vt_spec(j,:) = fftshift(abs(vgt)); 
end
subplot(3,2,1);
pcolor(tslide,(ks./(2*pi)),vt_spec.')
title('Gaussian');
ylabel('Frequency (Hz)')
xlabel('Time (s)')
shading interp 
colormap(hot)
subplot(3,2,2)
plot(t,g)
title('Gaussian');
ylabel('Filter Coefficient')
xlabel('Time (s)')

% Mexican Hat
dt = .1; % time step
tslide=0:dt:L; 
vt_spec = zeros(length(tslide),length(v));
for j=1:length(tslide)
    g=(1-(t-tslide(j)).^2).*exp(-(t-tslide(j)).^2/2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vt_spec(j,:) = fftshift(abs(vgt)); 
end
subplot(3,2,3);
pcolor(tslide,(ks./(2*pi)),vt_spec.')
title('Mexican Hat Wavelet');
ylabel('Frequency (Hz)')
xlabel('Time (s)')
shading interp 
colormap(hot)
subplot(3,2,4)
plot(t,g)
title('Mexican Hat Wavelet');
ylabel('Filter Coefficient')
xlabel('Time (s)')

% Shannon
a = 2; % Filter step width
dt = .1; % time step
tslide=0:dt:L; 
vt_spec = zeros(length(tslide),length(v));
for j=1:length(tslide)
    g=(abs(t-tslide(j)) <= a/2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vt_spec(j,:) = fftshift(abs(vgt)); 
end
subplot(3,2,5);
pcolor(tslide,(ks./(2*pi)),vt_spec.')
title('Shannon (Step Function)');
ylabel('Frequency (Hz)')
xlabel('Time (s)')
shading interp 
colormap(hot)
subplot(3,2,6)
plot(t,g)
title('Shannon (Step Function)');
ylabel('Filter Coefficient')
xlabel('Time (s)')


%% Part II - Piano

[y,Fs] = audioread('music1.wav');
tr_piano=length(y)/Fs; % record time in seconds
% p8 = audioplayer(y,Fs); playblocking(p8);
v = y';
t = (1:length(v))/Fs;
L = t(end);
n = length(t);
k=(2*pi/L)*[0:(n)/2-1 -(n)/2:-1]; 
ks=fftshift(k);

a = 50; % window width (inverse)
t_index = islocalmax(v,'MinSeparation',.3*Fs,'MinProminence',.4);
tslide=t(t_index); 
vt_spec = zeros(length(tslide),length(v));
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vt_spec(j,:) = fftshift(abs(vgt)); 
end

figure(5)
plot((1:length(y))/Fs,y,t(t_index),y(t_index),'r*');
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');

figure(6)
pcolor(tslide,(ks./(2*pi)),vt_spec.')
ylabel('Frequency (Hz)')
xlabel('Time (s)')
shading interp 
set(gca,'Ylim',[-1000 1000])
colormap(hot)

figure(7)
spectrogram(v,1000,[],[],Fs,'yaxis')
set(gca,'Ylim',[0 1]) % spectrogram default in kHz

[~,I] = max(vt_spec,[],2); 
Hz = ks./(2*pi); 
scores_piano = abs(Hz(I))'; 

figure(11)
spectrogram(v,1000,500,2^12,Fs,'yaxis')
set(gca,'Ylim',[0 6],'Xlim',[.7 1.2]) % spectrogram default in kHz


%% Part II - Recorder

[y,Fs] = audioread('music2.wav');
tr_rec=length(y)/Fs; % record time in seconds
% p8 = audioplayer(y,Fs); playblocking(p8);
v = y';
t = (1:length(v))/Fs;
L = t(end);
n = length(t);
k=(2*pi/L)*[0:(n)/2-1 -(n)/2:-1];
ks=fftshift(k);

a = 50; % window width (inverse)
t_index = islocalmax(v,'MinSeparation',.3*Fs,'MinProminence',.1);
tslide=t(t_index); 
vt_spec = zeros(length(tslide),length(v));
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    vg=g.*v; 
    vgt=fft(vg); 
    vt_spec(j,:) = fftshift(abs(vgt)); 
end

figure(8)
plot((1:length(y))/Fs,y,t(t_index),y(t_index),'r*');
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (recorder)');

figure(9)
pcolor(tslide,(ks./(2*pi)),vt_spec.')
ylabel('Frequency (Hz)')
xlabel('Time (s)')
shading interp 
set(gca,'Ylim',[-2000 2000])
colormap(hot)

figure(10)
spectrogram(v,1000,[],[],Fs,'yaxis')
set(gca,'Ylim',[0 2]) % spectrogram default in kHz

[~,I] = max(vt_spec,[],2); 
Hz = ks./(2*pi); 
scores_recorder = abs(Hz(I))'; 

figure(12)
spectrogram(v,1000,500,2^12,Fs,'yaxis')
set(gca,'Ylim',[0 20],'Xlim',[0 .5]) % spectrogram default in kHz



