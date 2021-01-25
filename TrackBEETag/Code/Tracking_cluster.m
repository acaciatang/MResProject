addpath('/rds/general/user/tst116/home/TrackBEETag/Code');
addpath('/rds/general/user/tst116/home/TrackBEETag/Data');

iter = getenv('PBS_ARRAY_INDEX');
%iter = 1;
file = ['Video' iter '.avi'];
%file = ['Video' num2str(iter) '.avi']
out = ['video' iter];
%out = ['video' num2str(iter)]

Tracking(file, out)