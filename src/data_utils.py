import os
import json
import subprocess
from pathlib import Path
import librosa
import soundfile as sf

def load_manifests (ids_path: str, ts_path: str):
    """
    Loads filtered JSON manifests for the IDs and timestamps.
    
    return:
        ids: Dict[str, List[str]]
        stamps: Dict[str, Dict[str, List[List[int]]]]
    """
    
    with open(ids_path, 'r') as f:
        ids = json.load(f)
    with open(ts_path, 'r') as f:
        stamps = json.load(f)
    return ids, stamps

def download_full_audio(ids: dict, out_dir: str, sr: int = 22050):
    """
    Download full audio for each YouTube ID as WAV into out_dir/category/vid.wav
    """
    os.makedirs(out_dir, exist_ok=True)
    for category, vids in ids.items():
        cat_dir = Path(out_dir) / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for vid in vids:
            out_path = cat_dir / f"{vid}.wav"
            if out_path.exists():
                continue
            cmd = [
                'yt-dlp',
                f'https://www.youtube.com/watch?v={vid}',
                '--extract-audio',
                '--audio-format', 'wav',
                '--output', str(out_path.with_suffix(''))
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to download audio for {vid}")


def segment_audio(ids: dict, stamps: dict, raw_dir: str, out_dir: str, fps: int = 30, sr: int = 22050):
    """
    Slice full WAVs into timestamped segments and save into out_dir/category/vid_index.wav
    """
    os.makedirs(out_dir, exist_ok=True)
    for category in ids:
        full_dir = Path(raw_dir) / category
        seg_dir = Path(out_dir) / category
        seg_dir.mkdir(parents=True, exist_ok=True)
        for vid in ids[category]:
            full_path = full_dir / f"{vid}.wav"
            if not full_path.exists():
                continue
            audio, _ = librosa.load(full_path, sr=sr)
            for i, (start_f, end_f) in enumerate(stamps[category].get(vid, [])):
                start_s = start_f / fps
                end_s = end_f / fps
                start_i = int(start_s * sr)
                end_i = int(end_s * sr)
                clip = audio[start_i:end_i]
                out_path = seg_dir / f"{vid}_{i}.wav"
                if out_path.exists():
                    continue
                sf.write(out_path, clip, sr)


def load_processed_dataset(proc_dir: str):
    """
    Load WAV clips from proc_dir/category into X (audio arrays) and y (labels)
    Categories mapped alphabetically.
    """
    X, y = [], []
    label_map = {cat: idx for idx, cat in enumerate(sorted(os.listdir(proc_dir)))}
    for cat, idx in label_map.items():
        for wav in Path(proc_dir).joinpath(cat).glob('*.wav'):
            audio, _ = librosa.load(wav, sr=None)
            X.append(audio)
            y.append(idx)
    return X, y
