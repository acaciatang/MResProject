%addpath('/rds/general/user/tst116/home/TrackBEETag/Code');
addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Code');
%addpath('/rds/general/user/tst116/home/TrackBEETag/Data');
addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Data');

%iter = getenv('PBS_ARRAY_INDEX');
iter = 1;
%file = ['Video' iter '.avi'];
file = ['Video' num2str(iter) '.mp4']
%out = ['video' iter];
out = ['video' num2str(iter)]

Tracking(file, out)