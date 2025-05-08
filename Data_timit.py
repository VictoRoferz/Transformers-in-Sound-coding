import deeplake
import os
from scipy.io.wavfile import write

#ds = deeplake.load("hub://activeloop/timit-dataset", read_only=True)
ds = deeplake.load("hub://activeloop/timit-train", read_only=True)
print(ds)

max_files = 100

out_dir = r"C:\Csyte\2024-DCCTN-Deep-Complex-Convolution-Transformer-with-Frequency-Transformation\Database\Original_Samples\Clean"
try:
    os.mkdir(out_dir)
except FileExistsError:
    pass

"""for i , sample in enumerate(ds, start=1):
    audio = sample["audio"].numpy()
    sr = sample["audio"].array.shape[0] and 16000
    out_path = os.path.join(out_dir, f"Clean_{i}.wav")
    write(out_path, 16000, audio)
    if i>200:
        break"""
        
tensor = ds.tensors["audios"]
#sample_rate = tensor.metadata.get("sample_rate", 16000)
sample_rate = 16000

for i, sample in enumerate(ds, start=1):
    if max_files  and i > max_files:
        break
    waveform  = sample["audios"].numpy()
    print(waveform)
    out_path = os.path.join(out_dir,f"Clean_{i}.wav")
    write(out_path,sample_rate, waveform)
    if i % 100 == 0:
        print(f" -> Wrote {i} files..")

print(f" Finished exporting {i} TIMIT iles to {out_dir}")