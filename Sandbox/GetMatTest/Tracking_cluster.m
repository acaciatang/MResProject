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