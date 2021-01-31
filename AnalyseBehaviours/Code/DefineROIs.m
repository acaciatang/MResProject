load(file)
BW = roipoly(im)

% manually outline then copy position of nest area, save it as nest
% manually outline then copy position of foraging area, save it as feed

out = '/Users/acacia/Desktop/gitrepo/MResProject/TrackBEETag/Results/BEE'

save([out '_nest.mat'], 'nest')
save([out '_feed.mat'], 'feed')