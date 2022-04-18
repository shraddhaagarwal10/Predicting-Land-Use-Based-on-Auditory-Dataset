import glob
import os
from gammatone.plot import take_input
import numpy as np

#ind_files = np.array(glob.glob("/media/iiserb/SANDISK_128/Dataset/Landuse-Prediction-via-Urban-Sound-tagging/audio_datasets/#Industrial/*"))
comm_files = np.array(glob.glob("/media/iiserb/SANDISK_128/Dataset/Landuse-Prediction-via-Urban-Sound-tagging/audio_datasets/Commercial2/*"))
res_files = np.array(glob.glob("/media/iiserb/SANDISK_128/Dataset/Landuse-Prediction-via-Urban-Sound-tagging/audio_datasets/Residential/*"))

#for i, audio in enumerate(ind_files):
#	try:
#		print("Current audio in progress: Industrial ", i) 
#		take_input(audio, "Industrial")

#	except:
#		continue


for i, audio in enumerate(comm_files):
	try:
		print("Current audio in progress: Commercial ", i) 
		take_input(audio, "Commercial")


	except:
		continue

#for i, audio in enumerate(res_files[6000:]):
#	try:
#		print("Current audio in progress: Residential ", i) 
#		take_input(audio, "Residential")

#	except:
#		continue
