"""
Neeche instructions ko padh:

First install library pydub using "pip install pydub"
Place the audio file in same folder as the script
Run the script "python audio_script.py"
Audios will be stored in a directory named "splitted_audio".


"""


from pydub import AudioSegment
import math
import os


parent_dir = "/media/iiserb/SANDISK_128/Dataset/"

directory = "splitted_audio"

path = os.path.join(parent_dir + directory)

os.mkdir(path)

class SplitWavAudioMubin():
    def __init__(self, filename, folder="/media/iiserb/SANDISK_128/Dataset/"):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + directory + '/' + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splited successfully')


audio_name = input("Name of audio: ")
audio = SplitWavAudioMubin(audio_name)
# Seconds per split --> 10

audio.multiple_split(10)
