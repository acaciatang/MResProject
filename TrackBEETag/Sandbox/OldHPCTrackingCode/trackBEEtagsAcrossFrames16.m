%% Beta/example code to track beetags across all frames of a video

%codelist = [109 36]; %List of codes in the frame - supplying this is more robust, but optional
codelist = [74 104 162 237 346 365 413 437 467 525 596 720 881 1077 1127 1203 1368 1486 1555 1730 1797 2107 2340 2418 2512 2607 2897 2954 3380 3443];
addpath('/Users/acacia/Desktop/antibiotics/A');
files = dir('*.MP4');

for iter = 1:length(files)
    file = files(iter);
    file = file.name;
    [pathstr, name, ext] = fileparts(file);
    out = name;
    Tracking16(file, out)
end

function f = Tracking16(file, out)
    disp(file)
    disp(out)
    mov = VideoReader([file]); %Make a VideoReader object for the movie

    nframes = mov.NumberOfFrames; %how many frames are in the video?


    %Create empty frame for tracking output
    trackingData = struct();

    %% Loop across frames
    for i = 1:nframes
        
        %% Read in each frames and track codes in it
        disp(strcat('tracking frame_', num2str(i), '_of_', num2str(nframes)));
        im = read(mov, i);
        
        %Two example options for tracking in each frame (only comment in one at a time):
        
        %example 1, maybe a little more robust but slower
        
        F = locate16BitCodes(im, 'threshMode', 1, 'bradleyFilterSize', [15 15], 'bradleyThreshold', 3);
        
        
        %example option 2, faster, simpler - just uses a simple threshold value instead
        %of doing adaptive filtering, less good for inhomogenously lit images
        
        %F = locateCodes(im, 'thresh', 0.25);
        
        %Append this single frame data to the master tracking output
        trackingData(i).F = F;
        
    end


    %% if there's no 'codelist' object defined, extract it from all the unique codes tracked in the movie
    if ~exist('codelist')
        for i = 1:nframes
            %for i = 1:numel(trackingData)
            curNumbers = [trackingData(i).F.number];
            %%
            if i == 1
                allNumbers = [] ;
            else
                allNumbers = [allNumbers curNumbers];
            end
            codelist = unique(allNumbers);
        end
    end
    %%
    %% Save data
    save([out '.mat'], 'trackingData')
    
    disp('rearranging data into e asier format');
    trackingDataReshaped = struct();
    for i = 1:nframes
        %%
        F = trackingData(i).F;
        for j = 1:numel(codelist)
            %%
            if ~isempty(F)
                FS = F([F.number] == codelist(j));
                if ~isempty(FS)
                    trackingDataReshaped(j).CentroidX(i) = FS.Centroid(1);
                    trackingDataReshaped(j).CentroidY(i) = FS.Centroid(2);
                    trackingDataReshaped(j).FrontX(i) = FS.frontX;
                    trackingDataReshaped(j).FrontY(i) = FS.frontY;
                    trackingDataReshaped(j).number(i) = FS.number;
                end
            end
        end
        
    end


    %% Save data
    save([out '_reshaped.mat'], 'trackingDataReshaped')

    %% Replay video
    disp('replaying video with tracking data shown');
    TD = trackingDataReshaped;

    outputMovieName = [out '.avi'];
    outputMovie = 0; %Set to 1 if you want to save a movie, set to 0 if not

    %If we're in movie writing mode, output the video
    if outputMovie == 0
    vidObj = VideoWriter(outputMovieName);
    open(vidObj)
    end

    for i = 1:nframes
        %%
        im =  read(mov, i);
        imshow(im);
        hold on;
        for j = 1:numel(TD)
            if numel(TD(j).CentroidX) >= i & ~isempty(TD(j).CentroidX(i))
                try
                plot([TD(j).CentroidX(i) TD(j).FrontX(i)], [TD(j).CentroidY(i) TD(j).FrontY(i)], 'b-','LineWidth', 3);
                text(TD(j).CentroidX(i), TD(j).CentroidY(i), num2str(TD(j).number(i)),'FontSize', 25, 'Color', 'r');
                catch
                    continue
                end
            end
        end
        drawnow
        currFrame = getframe;
        writeVideo(vidObj, currFrame);
        hold off;
    end

    close(vidObj);

f=0;
