%% Example for generating 10 pdfs, each with 100 labeled BEETags in it
taglist = [74 121 137 151 162 180 181 186 220 222 237 311 312 341 393 402 421 427 456 467 502 534 574 596 626 645 664 681 696 697 720 765 781 794 824 862 985 1074 1077 1419 1486 1797 1846 1875 1947 1966 2192 2211 2908 2915 74 121 137 151 162 180 181 186 220 222 237 311 312 341 393 402 421 427 456 467 502 534 574 596 626 645 664 681 696 697 720 765 781 794 824 862 985 1074 1077 1419 1486 1797 1846 1875 1947 1966 2192 2211 2908 2915];

ntags = 10; %How many rows and columns of tags to print? Will print ntags^2 tags (i.e. ntags = 10 produces an image with 100 tags)
    
f = figure('Visible', 'Off');
set(f, 'Position', [0 0 4000 4000])
for i = 1:(ntags^2)
    subplot(ntags,ntags,i);

    num = taglist(i);

    im = createPrintable16BitCode(num, 20);  

    imshow(im);
    text(-25, 180, num2str(num), 'FontSize', 8, 'Rotation', 90);
    text(185, 90, '->');  
    disp('Drew ' + i)

end
% Prints directly to a pdf (and therefore scalable) image of 100 tags
% instead of printing to figur
print(strcat('tags.pdf'), '-dpdf', '-bestfit');