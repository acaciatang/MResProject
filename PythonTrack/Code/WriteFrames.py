#imports
import sys
import os
import subprocess
import av

def writeframes(mp4, out):
    print(out)
    container = av.open(mp4)
    f = 0
    for frame in container.decode(video=0):
        if f == 0:
            frame.to_image().save('../Data/Training/' + os.path.splitext(os.path.basename(out))[0] + '_%05d.png' % frame.index)
            print(f)
        elif f %900 == 0:
            frame.to_image().save('../Data/Test/' + os.path.splitext(os.path.basename(out))[0] + '_%05d.png' % frame.index)
            print(f)
        f = f+1



def main(argv):
    for (dir, subdir, files) in subprocess.os.walk("/Volumes/Seagate Backup Plus Drive/Replicate 1"):
        for file in files:
            DIR = [i for i in file]
            if DIR[-4:-1] == ['.', 'M', 'P'] and DIR[-1] == "4":
                writeframes(dir + '/' + file, file)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)