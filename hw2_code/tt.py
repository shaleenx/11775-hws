import os

fread = open("list/test.video", "r")                                                                                                                             
for line in fread.readlines():
    video_name = line.replace('\n', '')
    filename = "downsampled_videos/" + video_name + ".ds.mp4"
    if not os.path.isfile(filename):
        print(filename)
