from pydub import AudioSegment


AudioSegment.from_mp3("./10 Hours of Construction Sound.mp3").export("./10_hours.wav",  format="wav")

#import subprocess

#subprocess.call(['ffmpeg', '-i', '10 Hours of Construction Sound.mp3',
 #                  '10_hours.wav'])
