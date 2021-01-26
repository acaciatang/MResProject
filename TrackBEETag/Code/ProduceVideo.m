function f = ProduceVideo(file1, file2, out)
<<<<<<< HEAD
    load(file1) % video[]_reshaped.mat
    load(file2) % video[]_mov.mat
    TD = trackingDataReshaped;
=======
    load(file1)
    TD = trackingDataReshaped;
    mov = VideoReader(file2);
>>>>>>> 126ede04d1a1b475f59cfa8bd9fa31d136a1ca57

    outputMovieName = [out '.avi'];
    outputMovie = 0; %Set to 1 if you want to save a movie, set to 0 if not

    %If we're in movie writing mode, output the video
    if outputMovie == 0
    vidObj = VideoWriter(outputMovieName);
    open(vidObj)
    end

    for i = 1:mov.NumberOfFrames
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

f = 0;