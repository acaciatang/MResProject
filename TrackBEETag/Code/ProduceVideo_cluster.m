%addpath('/rds/general/user/tst116/home/TrackBEETag/Code');
addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Code');
%addpath('/rds/general/user/tst116/home/TrackBEETag/Results');
addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Results');

%iter = getenv('PBS_ARRAY_INDEX');
iter = 1;
%file1 = ['video' iter '_reshaped.mat'];
file1 = ['video' num2str(iter) '_reshaped.mat'];
%file2 = ['Video' iter '.mp4'];
file2 = ['Video' num2str(iter) '.mp4'];
%out = ['Video' iter '_tagged.avi'];
out = ['Video' num2str(iter) '_tagged'];

ProduceVideo(file1, file2, out)