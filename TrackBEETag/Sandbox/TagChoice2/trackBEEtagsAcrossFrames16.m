%% Beta/example code to track beetags across all frames of a video

%codelist = [109 36]; %List of codes in the frame - supplying this is more robust, but optional
%codelist = [74 104 162 237 346 365 413 437 467 525 596 720 881 1077 1127 1203 1368 1486 1555 1730 1797 2107 2340 2418 2512 2607 2897 2954 3380 3443];
%codelist = [75 173 196 312 377 420 442 598 696 781 911 1044 1085 1175 1419 1799 1896 1986 2118 2196 2295 2381 2452 2682 2857 3261 3388 3486 3626 3994];
%codelist = [151 180 222 325 354 393 455 502 609 645 678 697 752 844 985 1062 1104 1180 1268 1437 1498 2006 2211 2488 2535 2701 2877 2915 3341 3488];
%codelist = [93 121 152 181 226 341 361 427 456 514 579 613 679 809 862 1028 1112 1465 1704 1947 2230 2413 2585 2827 2880 2919 3085 3375 3524 3757];

files = dir('*.MP4');

for iter = 1:length(files)
    file = files(iter);
    file = file.name;
    [pathstr, name, ext] = fileparts(file);
    out = name;
    
    if contains(file,'A')
        codelist = [74 104 162 237 346 365 413 437 467 525 596 720 881 1077 1127 1203 1368 1486 1555 1730 1797 2107 2340 2418 2512 2607 2897 2954 3380 3443]
    elseif contains(file,'B')
        codelist = [75 173 196 312 377 420 442 598 696 781 911 1044 1085 1175 1419 1799 1896 1986 2118 2196 2295 2381 2452 2682 2857 3261 3388 3486 3626 3994]
    elseif contains(file,'C')
        codelist = [151 180 222 325 354 393 455 502 609 645 678 697 752 844 985 1062 1104 1180 1268 1437 1498 2006 2211 2488 2535 2701 2877 2915 3341 3488]
	elseif contains(file,'D')
        codelist = [93 121 152 181 226 341 361 427 456 514 579 613 679 809 862 1028 1112 1465 1704 1947 2230 2413 2585 2827 2880 2919 3085 3375 3524 3757]
    end
          
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
end
