import os
import subprocess
files=os.listdir("videos");
for file in files:
    print(file);
    tutorial_num=file.split("lec")[1].split(".")[0];
    # print(tutorial_num);
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{tutorial_num}.mp3"]);
