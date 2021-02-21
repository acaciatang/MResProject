function f = GetMov(file, out)
    disp(file)
    mov = VideoReader(file); %Make a VideoReader object for the movie
    save([out '_mov.mat'], 'mov')
    f=0;