import os
from pydub import AudioSegment


def convert_mp3_to_wav(dataset_directory):
    origin_directory = os.path.join(dataset_directory, "voices")
    converted_directory = os.path.join(dataset_directory, "converted voices")
    if not os.path.exists(converted_directory):
        os.makedirs(converted_directory)
        for file in os.listdir(origin_directory):
            file_path = os.path.join(origin_directory, file)
            sound = AudioSegment.from_mp3(file_path)
            sound.export(os.path.join(converted_directory, f"{file.split(".")[0]}.wav"), format="wav")
            