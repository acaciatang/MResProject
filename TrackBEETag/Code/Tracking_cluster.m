<<<<<<< HEAD
addpath('/rds/general/user/tst116/home/TrackBEETag/Code');
%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Code');
addpath('/rds/general/user/tst116/home/TrackBEETag/Data');
%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Data');

iter = getenv('PBS_ARRAY_INDEX');
%for iter = 1:3
    file = ['video' iter '_mov.mat'];
    %file = ['video' num2str(iter) '_mov.mat']
    out = ['video' iter];
    %out = ['video' num2str(iter)]
    Tracking(file, out)
%end
=======
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
>>>>>>> 126ede04d1a1b475f59cfa8bd9fa31d136a1ca57
