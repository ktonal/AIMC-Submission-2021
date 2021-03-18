from mimikit.audios.file_walker import AudioFileWalker
from mimikit.h5data.write import _sizeof_fmt
import pandas as pd
import os


DBS = {
    "throat": ["Gutural.m4a", "Throat Singing.mp3"],
    "yodel": ["Yodel_female.wav", "Yodel_male.wav", "Yodel_male_Duet.wav"],
    "pansori": ["pansori_Neptune_data_male.mp3", "Pansori_Bum_female.mp3"],

}

data_root = os.path.join(os.path.split(__file__)[0], "data")
all_files = list(AudioFileWalker(roots=[data_root]))
file_sizes = [_sizeof_fmt(os.path.getsize(file_path)) for file_path in all_files]

files_df = pd.DataFrame({'name': [os.path.split(f)[1] for f in all_files], 'size': file_sizes})


def random_draw(n_files):
    return list(files_df.sample(n_files)["name"])