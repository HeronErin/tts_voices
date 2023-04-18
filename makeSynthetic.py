import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()
print("Imported right")


voice_samples, conditioning_latents = load_voice("kento")


import os, random
# from concurrent.futures import ThreadPoolExecutor
#
# def strip_non_ascii(string):
#   ''' Returns the string without non ASCII characters'''
#   stripped = (c for c in string if 0 < ord(c) < 127)
#   return ''.join(stripped)
#
os.mkdir("wavs")
#
i = 0
lines = open("input.txt", "r").read().split("\n")[i:]
for l in lines:
    gen = tts.tts_with_preset(l, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                            preset="ultra_fast")
    torchaudio.save(f'wavs/{i}.wav', gen.squeeze(0).cpu(), 24000)

    f = open("metadata.csv", "a")
    f.write(f"{i}||{l}\n")
    f.close()
    print(f"Finished with {i}")
    i+=1

