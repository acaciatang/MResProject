%% Beta/example code to track beetags across all frames of a video

%codelist = [109 36]; %List of codes in the frame - supplying this is more robust, but optional
function f = Tracking16(file, out)
    disp(file)
    disp(out)
    mov = VideoReader([file]); %Make a VideoReader object for the movie
    nframes = mov.NumFrames; %how many frames are in the video?

    im = read(mov, 1);
    save([out 'firstFrame.mat'], 'im')
    
    nchunks = floor(nframes/1000) + 1;
    chunks = 1:1000:nchunks*1000;
    
    for iter = 1:nchunks
    
        alpha = chunks(iter);
        if iter ~= nchunks
            omega = chunks(iter+1)-1;
        else
            omega = nframes;
        end
        
        %Create empty frame for tracking output
        trackingData = struct();

        %% Loop across frames
        for i = alpha:omega
            %% Read in each frames and track codes in it
            disp(strcat('tracking frame_', num2str(i), '_of_', num2str(nframes)));
            im = read(mov, i);

            %Two example options for tracking in each frame (only comment in one at a time):

            %example 1, maybe a little more robust but slower

            F = locate16BitCodes(im, 'threshMode', 1, 'bradleyFilterSize', [15 15], 'bradleyThreshold', 3);


            %example option 2, faster, simpler - just uses a simple threshold value instead
            %of doing adaptive filtering, less good for inhomogenously lit images

            %F = locateCodes(im, 'thresh', 0.25);
            
            itemID = i - (iter-1) * 1000;
            
            %Append this single frame data to the master tracking output
            trackingData(itemID).F = F;

        end
        
        save([out '_' num2str(iter) '.mat'], 'trackingData')
    end
f=0;
