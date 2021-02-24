addpath('/rds/general/user/tst116/home/TrackBEETag/Code');
addpath('/rds/general/user/tst116/home/TrackBEETag/Data');

iter = getenv('PBS_ARRAY_INDEX');
file = ['video' iter '.mp4'];
out = ['video' iter];
Tracking(file, out)
%end


%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Code');
%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Data');
%for iter = 1:3
%    file = ['video' num2str(iter) '_mov.mat']
%    out = ['video' num2str(iter)]
%    Tracking(file, out)
%end