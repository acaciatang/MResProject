addpath('/rds/general/user/tst116/home/TrackBEETag/Code');
%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Code');
addpath('/rds/general/user/tst116/home/TrackBEETag/Results');
%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Results');

iter = getenv('PBS_ARRAY_INDEX');
%for iter = 1:3
    file1 = ['video' iter '_reshaped.mat'];
    %file1 = ['video' num2str(iter) '_reshaped.mat'];
    file2 = ['video' iter '_mov.mat'];
    %file2 = ['video' num2str(iter) '_mov.mat'];
    out = ['video' iter '_tagged.avi'];
    %out = ['Video' num2str(iter) '_tagged'];
    ProduceVideo(file1, file2, out)
%end
