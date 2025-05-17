import os
import json
import subprocess
from pathlib import Path
import librosa
import soundfile as sf
import yt_dlp

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

def download_full_audio_api(ids: dict, out_dir: str, sr: int = 22050):
    """
    Download & convert each YouTube ID’s audio into WAV files under out_dir/category/vid.wav
    New strat using yt_dlp to convert easier to wav
    """
    os.makedirs(out_dir, exist_ok=True)
    for category, vids in ids.items():
        cat_dir = Path(out_dir) / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for vid in vids:
            out_path = cat_dir / f"{vid}.wav"
            if out_path.exists():
                continue
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(cat_dir / f'{vid}.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                # suppress progress bar for cleaner logs
                'quiet': True,
                'no_warnings': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f'https://www.youtube.com/watch?v={vid}'])
                print(f"Downloaded & converted: {vid}")
            except Exception as e:
                print(f"Failed to download {vid}: {e}")

def segment_audio(ids: dict, stamps: dict, raw_dir: str, out_dir: str, fps: int = 30, sr: int = 22050):
    """
    Load whichever <video_id>.<ext> exists, slice by timestamp, write .wav clips.
    """
    os.makedirs(out_dir, exist_ok=True)
    for category in ids:
        seg_dir = Path(out_dir) / category
        seg_dir.mkdir(parents=True, exist_ok=True)
        full_dir = Path(raw_dir) / category
        for vid in ids[category]:
            # find the downloaded file (any extension)
            files = list(full_dir.glob(f"{vid}.*"))
            if not files:
                print(f"No raw audio found for {vid}, skipping segments.")
                continue
            full_path = files[0]
            audio, _ = librosa.load(full_path, sr=sr)
            for i, (sf_, ef_) in enumerate(stamps[category].get(vid, [])):
                start_i = int((sf_/fps) * sr)
                end_i   = int((ef_/fps) * sr)
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
