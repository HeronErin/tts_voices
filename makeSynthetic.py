import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()
print("Imported right")


voice_samples, conditioning_latents = load_voice("darknet_monotone")
gen = tts.tts_with_preset("this is some test text!", voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                          preset=preset)
torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)

# import os, random
# from concurrent.futures import ThreadPoolExecutor
#
# def strip_non_ascii(string):
#   ''' Returns the string without non ASCII characters'''
#   stripped = (c for c in string if 0 < ord(c) < 127)
#   return ''.join(stripped)
#
# os.mkdir("wavs")
#
# i = 0
# lines = open("input.txt", "r").read().split("\n")
# for l in lines:
#     f = open("/tmp/t", "w")
#     f.write(strip_non_ascii(l))
#     f.close()
#
#     !cat /tmp/t | python scripts/tortoise_tts.py --voice darknet_monotone --preset ultra_fast -O /tmp
#     os.system(f"mv /tmp/darknet_monotone_combined.wav wavs/{i}.wav")
#
#     f = open("metadata.csv", "a")
#     f.write(f"{i}||{l}\n")
#     f.close()
#     print(v"Finished with {i}")
#     i+=1
