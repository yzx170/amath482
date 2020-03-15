clear; close all; clc;

%% Loading

% Names of music files
% Classical
D = dir('Classical_Bach*.mp3');
for j = 1:length(D)
    CL.Bach{j} = D(j).name;
end
D = dir('Classical_Beethoven*.mp3');
for j = 1:length(D)
    CL.Beethoven{j} = D(j).name;
end
D = dir('Classical_Mozart*.mp3');
for j = 1:length(D)
    CL.Mozart{j} = D(j).name;
end
CL.All = [CL.Bach CL.Beethoven CL.Mozart];
% Country
D = dir('Country_Chris_Haugen*.mp3');
for j = 1:length(D)
    CO.CH{j} = D(j).name;
end
D = dir('Country_Dan_Lebowitz*.mp3');
for j = 1:length(D)
    CO.DL{j} = D(j).name;
end
D = dir('Country_Nat_Keefe*.mp3');
for j = 1:length(D)
    CO.NK{j} = D(j).name;
end
CO.All = [CO.CH CO.DL CO.NK];
% Electronic
D = dir('Electronic_Spence*.mp3');
for j = 1:length(D)
    EL.S{j} = D(j).name;
end
D = dir('Electronic_Quincas_Moreira*.mp3');
for j = 1:length(D)
    EL.QM{j} = D(j).name;
end
D = dir('Electronic_Rondo_Brothers*.mp3');
for j = 1:length(D)
    EL.RB{j} = D(j).name;
end
EL.All = [EL.S EL.QM EL.RB];


%% Part I: Different Genre Band Classification

% 1 - Classical.Mozart, 2 - Country.Chris Haugen, 3 - Electronic.QM

% Data - all pieces from each Band
data1 = generateData(CL.Mozart);
data2 = generateData(CO.CH);
data3 = generateData(EL.QM);

% Choose test data
sampleSize = min([size(data1,1) size(data2,1) size(data3,1)]);
number = randperm(sampleSize);
testSize = sampleSize; % cross validate with all data
number = number(1:testSize); 
classification = zeros(testSize,3);
successRate = zeros(testSize,1);

for jj = 1:testSize
    test_i = number(jj);
    % Test data
    test1 = data1(test_i,:);
    test2 = data2(test_i,:);
    test3 = data3(test_i,:);

    % Preparing training data (exclude test file)
    trainer1 = data1; trainer1(test_i,:) = [];
    trainer2 = data2; trainer2(test_i,:) = [];
    trainer3 = data3; trainer3(test_i,:) = [];

    % Classification
    % SVD and projection 
    clear trainerScores
    clear testScores
    n = 3;
    [trainerScores,ind,U] = projSVD(trainer1,trainer2,trainer3,n);
    testScores(:,1) = U(:,1:n)'*test1';
    testScores(:,2) = U(:,1:n)'*test2';
    testScores(:,3) = U(:,1:n)'*test3';

    % Classification by knn
    Class = zeros(3,1);
    for j = 1:3
        Idx = knnsearch(trainerScores',testScores(:,j)','k',3);
        ClusterInd = ind(Idx);
        Class(j) = mode(ClusterInd);
    end
    classification(jj,:) = Class';
    successRate(jj) = sum(Class == [1;2;3])/3;
end
P1_OverallAccuracy = mean(successRate);


%% Scatter Plot

% figure(1)
% scatter3(trainerScores(1,1:8)',trainerScores(2,1:8)',trainerScores(3,1:8)','r');
% hold on;scatter3(trainerScores(1,9:16)',trainerScores(2,9:16)',trainerScores(3,9:16)','b');
% scatter3(trainerScores(1,17:24)',trainerScores(2,17:24)',trainerScores(3,17:24)','m')
% xlabel('Modal Score 1')
% ylabel('Modal Score 2')
% zlabel('Modal Score 3')
% legend('Cluster 1 - Classical','Cluster 2 - Country','Cluster 3 - Electronic')


%% Spectrogram

% figure(2)
% title('Mozart')
% for j = 1:9 
% filename = CL.Mozart{j};[y,Fs] = audioread(filename);
% % check mono or stereo
% if size(y,2) == 2 % double channel
%     y = mean(y,2);
% end
% % truncate data
% y = y(55*Fs+1:60*Fs); % extract only 55-60 sec
% % resample
% y = reshape(y,[5 5*Fs/5]); % one data point every five
% y = mean(y,1);
% Fs = Fs/5; % update sampling frequency
% % spectrogram
% subplot(3,3,j)
% spectrogram(y,1000,[],[],Fs); drawnow
% end
% figure(3)
% title('Country CH')
% for j = 1:9 
% filename = CO.CH{j};[y,Fs] = audioread(filename);
% % check mono or stereo
% if size(y,2) == 2 % double channel
%     y = mean(y,2);
% end
% % truncate data
% y = y(55*Fs+1:60*Fs); % extract only 55-60 sec
% % resample
% y = reshape(y,[5 5*Fs/5]); % one data point every five
% y = mean(y,1);
% Fs = Fs/5; % update sampling frequency
% % spectrogram
% subplot(3,3,j)
% spectrogram(y,1000,[],[],Fs); drawnow
% end
% figure(4)
% title('Electronic QM')
% for j = 1:9 
% filename = EL.QM{j};[y,Fs] = audioread(filename);
% % check mono or stereo
% if size(y,2) == 2 % double channel
%     y = mean(y,2);
% end
% % truncate data
% y = y(55*Fs+1:60*Fs); % extract only 55-60 sec
% % resample
% y = reshape(y,[5 5*Fs/5]); % one data point every five
% y = mean(y,1);
% Fs = Fs/5; % update sampling frequency
% % spectrogram
% subplot(3,3,j)
% spectrogram(y,1000,[],[],Fs); drawnow
% end


%% Part II: Same Genre Band Classification

% 1 - Classical.Mozart, 2 - Country.Chris Haugen, 3 - Electronic.QM

% Data - all pieces from each Band
data1 = generateData(CO.DL);
data2 = generateData(CO.CH);
data3 = generateData(CO.NK);

% Choose test data
sampleSize = min([size(data1,1) size(data2,1) size(data3,1)]);
number = randperm(sampleSize);
testSize = sampleSize; % cross validate with all data
number = number(1:testSize); 
classification = zeros(testSize,3);
successRate = zeros(testSize,1);

for jj = 1:testSize
    test_i = number(jj);
    % Test data
    test1 = data1(test_i,:);
    test2 = data2(test_i,:);
    test3 = data3(test_i,:);

    % Preparing training data (exclude test file)
    trainer1 = data1; trainer1(test_i,:) = [];
    trainer2 = data2; trainer2(test_i,:) = [];
    trainer3 = data3; trainer3(test_i,:) = [];

    % Classification
    % SVD and projection 
    clear trainerScores
    clear testScores
    n = 3;
    [trainerScores,ind,U] = projSVD(trainer1,trainer2,trainer3,n);
    testScores(:,1) = U(:,1:n)'*test1';
    testScores(:,2) = U(:,1:n)'*test2';
    testScores(:,3) = U(:,1:n)'*test3';

    % Classification by knn
    Class = zeros(3,1);
    for j = 1:3
        Idx = knnsearch(trainerScores',testScores(:,j)','k',3);
        ClusterInd = ind(Idx);
        Class(j) = mode(ClusterInd);
    end
    classification(jj,:) = Class';
    successRate(jj) = sum(Class == [1;2;3])/3;
end
P2_OverallAccuracy = mean(successRate);


%% Part III: Same Genre Band Classification

% 1 - Classical.Mozart, 2 - Country.Chris Haugen, 3 - Electronic.QM

% Data - all pieces from each Band
data1 = generateData(CL.All);
data2 = generateData(CO.All);
data3 = generateData(EL.All);

% Choose test data
sampleSize = min([size(data1,1) size(data2,1) size(data3,1)]);
number = randperm(sampleSize);
testSize = sampleSize; % cross validate with all data
number = number(1:testSize); 
classification = zeros(testSize,3);
successRate = zeros(testSize,1);

for jj = 1:testSize
    test_i = number(jj);
    % Test data
    test1 = data1(test_i,:);
    test2 = data2(test_i,:);
    test3 = data3(test_i,:);

    % Preparing training data (exclude test file)
    trainer1 = data1; trainer1(test_i,:) = [];
    trainer2 = data2; trainer2(test_i,:) = [];
    trainer3 = data3; trainer3(test_i,:) = [];

    % Classification
    % SVD and projection 
    clear trainerScores
    clear testScores
    n = 3;
    [trainerScores,ind,U] = projSVD(trainer1,trainer2,trainer3,n);
    testScores(:,1) = U(:,1:n)'*test1';
    testScores(:,2) = U(:,1:n)'*test2';
    testScores(:,3) = U(:,1:n)'*test3';

    % Classification by knn
    Class = zeros(3,1);
    for j = 1:3
        Idx = knnsearch(trainerScores',testScores(:,j)','k',3);
        ClusterInd = ind(Idx);
        Class(j) = mode(ClusterInd);
    end
    classification(jj,:) = Class';
    successRate(jj) = sum(Class == [1;2;3])/3;
end
P3_OverallAccuracy = mean(successRate);





















%% Functions

function output = convSpec(filename)
% Input: filename - string
% Output: spectrogram of resampled audio data - horizontal vector 

% load file
[y,Fs] = audioread(filename);
% check mono or stereo
if size(y,2) == 2 % double channel
    y = mean(y,2);
end
% truncate data
y = y(55*Fs+1:60*Fs); % extract only 55-60 sec
% resample
y = reshape(y,[5 5*Fs/5]); % one data point every five
y = mean(y,1);
Fs = Fs/5; % update sampling frequency
% spectrogram
spec = spectrogram(y,1000,[],[],Fs);
% flatten and absolute value
output = abs(reshape(spec,[size(spec,1)*size(spec,2) 1])');
end

function [scores,ind,U] = projSVD(A,B,C,n)
% Input: training sets A B C - horizontal vector/matrix
%        n - number of features to use
% Output: scores - match (dot product) of each feature n by m
%         ind - index of the feature scores
%         U - for projecting additional data points

trainer = [A;B;C]';
[U,~,~] = svd(trainer,'econ');
scores = U(:,1:n)'*trainer;
ind = [1*ones(1,size(A,1)) 2*ones(1,size(B,1)) 3*ones(1,size(C,1))];
end

function output = generateData(cellname)
% Stack entries in specified cell to form a combined data matrix
trial = convSpec(cellname{1});
output = zeros(length(cellname),size(trial,2));
output(1,:) = trial;
for j = 2:length(cellname)
    output(j,:) = convSpec(cellname{j});
end
end