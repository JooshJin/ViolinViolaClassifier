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

def download_and_trim_all(ids: dict, stamps: dict, out_dir: str, fps: int = 30):
    """
    Download and trim all Youtube segments using timestamps and directories. Saved under out_dir/Violin/ and out_dir/Viola/ as MP4s.
    """
       
    os.makedirs(out_dir, exist_ok = True)
    for category, vids in ids.items():
        cat_dir = Path(out_dir) / category
        cat_dir.mkdir(parents = True, exist_ok = True)
        
        for vid in vids:
            segments = stamps[category].get(vid, [])
            
            for i, (start_f, end_f) in enumerate(segments):
                start_s = start_f / fps
                end_s = end_f / fps
                out_path = cat_dir / f"{vid}_{i}.mp4"
                
                if out_path.exists():
                    continue
                cmd = [
                    'yt-dlp',
                    f'https://www.youtube.com/watch?v={vid}',
                    '--output', str(out_path),
                    '--download-sections', f'*{start_s:.1f}-{end_s:.1f}',
                    '--format', 'bestvideo+bestaudio'
                ]
                subprocess.run(cmd, check=True)
                
def preprocess_audio(raw_dir: str, proc_dir: str, sr: int = 22050, duration: float = 5.0):
    """
    Convert each mp4 in raw_dir to a fixed length WAV in proc_dir
    """
    for category in os.listdir(raw_dir):
        in_cat = Path(raw_dir) / category
        out_cat = Path(proc_dir) / category
        out_cat.mkdir(parents = True, exist_ok = True)
        for mp4 in in_cat.glob('*.mp4'):
            wav_path = out_cat / (mp4.stem + '.wav')
            if wav_path.exists():
                continue
            audio, _ = librosa.load(mp4, sr=sr, duration=duration)
            sf.write(wav_path, audio, sr)
            
def load_processed_dataset(proc_dir: str):
    """
    Load all WAVs from proc_dir/category/ into arrays and labels

    returns:
        X: List[np.ndarray], y: List[int]
        Categories are mapped by alphabetical order of subfolders.
    """
    X, y = [], []
    label_map = {cat: idx for idx, cat in enumerate(sorted(os.listdir(proc_dir)))}
    for cat, idx in label_map.items():
        folder = Path(proc_dir) / cat
        for wav in folder.glob('*.wav'):
            audio, _ = librosa.load(wav, sr=None)
            X.append(audio)
            y.append(idx)
    return X, y
