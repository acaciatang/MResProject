<<<<<<< HEAD
addpath('/rds/general/user/tst116/home/TrackBEETag/Code');
%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Code');
addpath('/rds/general/user/tst116/home/TrackBEETag/Results');
%addpath('/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Results');

iter = getenv('PBS_ARRAY_INDEX');
%for iter = 1:3
    file1 = ['video' iter '_reshaped.mat'];
    %file1 = ['video' num2str(iter) '_reshaped.mat'];
    file2 = ['video' iter '.mp4'];
    %file2 = ['video' num2str(iter) '_mov.mat'];
    out = ['video' iter '_tagged.avi'];
    %out = ['Video' num2str(iter) '_tagged'];
    ProduceVideo(file1, file2, out)
%end
=======
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
>>>>>>> 126ede04d1a1b475f59cfa8bd9fa31d136a1ca57
