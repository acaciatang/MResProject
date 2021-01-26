%% Beta/example code to track beetags across all frames of a video

%codelist = [109 36]; %List of codes in the frame - supplying this is more robust, but optional

%[filename pathname] = uigetfile('*'); %User-specified file input - this can be modified to be automated if you need to track over lots of files
function f = Tracking(file, out)
    disp(file)
    disp(out)
    mov = VideoReader(file); %Make a VideoReader object for the movie

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
        
        F = locateCodes(im, 'threshMode', 1, 'bradleyFilterSize', [15 15], 'bradleyThreshold', 3);
        
        
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
    save([out '.mat'], 'trackingData')

    %makes it work somehow
    TD = trackingDataReshaped;

f = 0;
