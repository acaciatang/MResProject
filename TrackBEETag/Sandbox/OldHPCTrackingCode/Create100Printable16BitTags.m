%% Example for generating 10 pdfs, each with 100 labeled BEETags in it
load master16BitCodeList.mat

ntags = 10; %How many rows and columns of tags to print? Will print ntags^2 tags (i.e. ntags = 10 produces an image with 100 tags)
    
f = figure('Visible', 'Off');
set(f, 'Position', [0 0 4000 4000])
for j = [0 100];
    %%
    for i = 1:(ntags^2)
        
        subplot(ntags,ntags,i);
        
        num = printtags(j + i);
        
        im = createPrintable16BitCode(num, 20);  
        
        imshow(im);
        text(-25, 180, num2str(num), 'FontSize', 8, 'Rotation', 90);
        text(185, 90, '->');
    end
    
    % Prints directly to a pdf (and therefore scalable) image of 100 tags
    % instead of printing to figur
    print(strcat(num2str(j), '-', num2str(j+99), 'keyed.pdf'), '-dpdf', '-bestfit');
end