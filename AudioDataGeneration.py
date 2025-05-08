import librosa as lr
import numpy as np
import soundfile as sf
import os
from pathlib import Path


def rms_energy(x):
    return 10 * np.log10((1e-12 + x.dot(x)) / len(x))


def SPL_cal(x, SPL):
    SPL_before = 20 * np.log10(np.sqrt(np.mean(x ** 2)) / (20 * 1e-6))
    y = x * 10 ** ((SPL - SPL_before) / 20)
    return y


def add_noise(signal, noise, fs, snr, signal_energy='rms'):
    # Generate random section of masker
    if len(noise) != len(signal):
        idx = np.random.randint(0, len(noise) - len(signal))
        noise = noise[idx:idx + len(signal)]
    # Scale noise wrt speech at target SNR
    N_dB = rms_energy(noise)
    if signal_energy == 'rms':
        S_dB = rms_energy(signal)
    else:
        raise ValueError('signal_energy has to be either "rms" or "P.56"')

    # Rescale N
    N_new = S_dB - snr
    noise_scaled = 10 ** (N_new / 20) * noise / 10 ** (N_dB / 20)
    noisy_signal = signal + noise_scaled
    return noisy_signal


#def datagenerator(In_path, Out_path, Noise_file, SNR, Num_audio_sample, sample_margin, Fs):
def datagenerator(In_path, Out_path, noise_path: Path, SNR, Num_audio_sample, sample_margin, Fs):
    """
    noise_path: full Path to one .wav file in Different_Noise/
    """
    Noise_file = noise_path.stem   # e.g. "noise-free-sound-0001"
    for i in range(Num_audio_sample):
        # build the clean input path
        clean_path = Path(In_path) / "Clean" / f"Clean_{i+sample_margin}.wav"
        clean, _ = lr.load(str(clean_path), sr=Fs)

        # load this specific noise file
        noise, _ = lr.load(str(noise_path), sr=Fs)

        # scale & mix
        clean_scaled = SPL_cal(clean, 65)
        try:
            noisy = add_noise(clean_scaled, noise, Fs, SNR)
        except ValueError:
            print(f"Skipping noise {Noise_file}: too short")
            continue
            
        noisy = SPL_cal(noisy, 65)

        # output paths
        out_clean = Path(Out_path) / "Clean" / f"Clean_{i+sample_margin}.wav"
        out_noisy = Path(Out_path) / "Noisy" / f"Noisy_{Noise_file}_{SNR}_dB_{i+sample_margin}.wav"

        sf.write(str(out_clean), clean_scaled, Fs)
        sf.write(str(out_noisy), noisy, Fs)
        print(f"[{Noise_file} @ {SNR}dB] sample {i+sample_margin}")


if __name__ == '__main__':
    base = Path(os.getcwd()) / "Database" / "Original_Samples"

    # make output dirs (as you already do) …
    for split in ("Train","Dev","Test"):
        for sub in ("Clean","Noisy","Enhanced" if split=="Test" else "Noisy"):
            (base / split / sub).mkdir(parents=True, exist_ok=True)

    Fs = 16000

    # find every .wav under both subfolders
    noise_dir = base / "Different_Noise"
    noise_paths = sorted(noise_dir.rglob("*.wav"))

    # define your splits
    splits = {
      "Train": dict(snrs=[0,5],  n=10, margin=1),
      "Dev":   dict(snrs=[0,5],  n=5,  margin=11),
      "Test":  dict(snrs=[5,10], n=5,  margin=16),
    }

    for split, cfg in splits.items():
        out_path = base / split
        print(f"--- Generating {split} ({cfg['n']} samples) ---")
        for noise_path in noise_paths:
            for snr in cfg["snrs"]:
                datagenerator(
                    In_path=str(base),
                    Out_path=str(out_path),
                    noise_path=noise_path,
                    SNR=snr,
                    Num_audio_sample=cfg["n"],
                    sample_margin=cfg["margin"],
                    Fs=Fs
                )
        print(f"Completed {split}.\n")
