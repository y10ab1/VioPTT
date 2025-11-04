import os
import glob
import subprocess
from pydub import AudioSegment
from tqdm import tqdm

DATASET_DIR = "/mnt/gestalt/home/tkwang/ViolinMamba/violin-transcription/dataset"

def parse_filename(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) < 3:
        return None
    # concat items after 3
    print(parts)
    
    sample_info = "_".join(parts[3:])

    sample_info = sample_info.replace(".mid", "").split("-")

    print(sample_info)



    if len(parts[:-2]) > 1:
        youtube_id = "-".join(sample_info[:-2])
    else:
        youtube_id = sample_info[-3]
    start_sec = int(sample_info[-2])
    end_sec = int(sample_info[-1])
    return youtube_id, start_sec, end_sec

def download_youtube_audio(youtube_id, start, end, output_path):
    temp_mp3 = "temp_audio.mp3"
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Download audio
    print(f"Downloading: {url}")
    subprocess.run([
        "yt-dlp", "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", temp_mp3,
        url
    ], check=True)

    # Trim using pydub
    audio = AudioSegment.from_mp3(temp_mp3)
    clipped = audio[start*1000:end*1000]
    clipped.export(output_path, format="wav")
    os.remove(temp_mp3)
    print(f"Saved audio to {output_path}")

def main():
    mid_files = glob.glob(os.path.join(DATASET_DIR, '*', "*.mid"))
    print(f"Found {len(mid_files)} MIDI files")
    for mid_file in tqdm(mid_files):
        print(mid_file)
        info = parse_filename(mid_file)
        if not info:
            print(f"Skipping malformed filename: {mid_file}")
            continue
        youtube_id, start, end = info

        # the path is "mnt/gestalt/home/tkwang/ViolinMamba/violin-transcription/dataset/Kayser/Kayser_Op20-36_SunKim_HfIh7cr04_c-0000-0079.mid"
        output_wav = mid_file.replace(".mid", "_audio.wav")
        if os.path.exists(output_wav):
            print(f"Already exists: {output_wav}")
            continue
        try:
            download_youtube_audio(youtube_id, start, end, output_wav)
        except Exception as e:
            print(f"Failed to download {youtube_id}: {e}")

if __name__ == "__main__":
    print("Starting conversion...")
    main()
