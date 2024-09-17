import os
import subprocess
import sys
import multiprocessing

from tqdm import tqdm
from pytube import Playlist, YouTube
import yt_dlp

from tool import read_json, normalize_text


def process_videos(teleplay_name, output_path):
    teleplay_folder = os.path.join(output_path, teleplay_name)
    teleplay_videos_folder = os.path.join(teleplay_folder, "videos")
    videos = os.listdir(teleplay_videos_folder)
    for video_file in tqdm(videos, desc=f"Processing {teleplay_name}", unit='video'):
        if not video_file.endswith(".mp4"):
            continue
        input_path = os.path.join(teleplay_videos_folder, video_file)
        file_name = os.path.basename(input_path).split(".")[0]

        os.makedirs(os.path.join(teleplay_folder, "audios"), exist_ok=True)
        output_audio = os.path.join(teleplay_folder, "audios", f"{file_name}.wav")
        if os.path.exists(output_audio):
            print(f"Skipped (already exists): {file_name}.wav")
        else:
            subprocess.run(
                [
                    'ffmpeg',
                    '-i', input_path,
                    '-vn', '-ar',
                    '48000', '-ac',
                    '1',
                    output_audio
                ],
                capture_output=True
            )
    done_folder_path = os.path.join(output_path, "Done")
    os.makedirs(done_folder_path, exist_ok=True)
    done_file_path = os.path.join(done_folder_path, f"{teleplay_name}.done")
    with open(done_file_path, "w") as f:
        f.write(f'{teleplay_name} is done.')


class DownloadMethod2:
    def __init__(self, teleplay_download_url_dict, output_path):
        self.teleplay_download_url_dict = teleplay_download_url_dict
        self.output_path = output_path

    def get_video_urls(self, teleplay_url):
        ydl_opts = {
            'extract_flat': True,
            'skip_download': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(teleplay_url, download=False)
            if 'entries' in result:
                return [video['url'] for video in result['entries']]
            else:
                return []

    def download_video(self, video_url, video_output_folder):
        ydl_opts = {
            'outtmpl': os.path.join(video_output_folder, '%(title)s.%(ext)s'),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'noplaylist': True,
            'nooverwrites': True,
            'continuedl': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except yt_dlp.utils.DownloadError as e:
                print(f"Skipping video {video_url} due to error: {e}")

    def download_videos(self, video_urls, video_output_folder):
        ps = []
        for video_url in video_urls:
            p = multiprocessing.Process(target=self.download_video, args=(video_url, video_output_folder))
            ps.append(p)
            p.start()

        for p in ps:
            p.join()

    def process_one(self, teleplay_name, teleplay_url):
        video_output_folder = os.path.join(self.output_path, f"{teleplay_name}", "videos")
        os.makedirs(video_output_folder, exist_ok=True)
        video_urls = self.get_video_urls(teleplay_url)
        self.download_videos(video_urls, video_output_folder)
        process_videos(teleplay_name, self.output_path)

    def download_and_process(self):
        ps = []
        for name, url in self.teleplay_download_url_dict.items():
            ps.append(
                multiprocessing.Process(
                    target=self.process_one,
                    args=(name, url,)
                )
            )
        for i in ps:
            i.start()
        for i in ps:
            i.join()


class DownloadMethod1:
    def __init__(self, teleplay_download_url_dict, output_path):
        self.teleplay_download_url_dict = teleplay_download_url_dict
        self.output_path = output_path

    def process_one(self, teleplay_name, teleplay_url):
        done_folder_path = os.path.join(self.output_path, "Done")
        done_file_path = os.path.join(done_folder_path, f"{teleplay_name}.done")
        if os.path.exists(done_file_path):
            return

        os.makedirs(os.path.join(self.output_path, "log"), exist_ok=True)
        log_file = os.path.join(self.output_path, "log", f"{teleplay_name}.log")
        if os.path.exists(log_file):
            idx = 1
            temp_file = log_file.split(".")[0] + f"_{idx}.log"
            while os.path.exists(temp_file):
                idx = idx + 1
                temp_file = log_file.split(".")[0] + f"_{idx}.log"
            log_file = temp_file

        with open(log_file, "w", encoding='utf-8') as f:
            sys.stdout = f
            sys.stderr = f
            self.download_videos(teleplay_name, teleplay_url)
            process_videos(teleplay_name, self.output_path)

    def download_videos(self, teleplay_name, teleplay_url):
        playlist = Playlist(teleplay_url)
        print(f"Downloading teleplay: {teleplay_name}")

        video_output_folder = os.path.join(self.output_path, f"{teleplay_name}", "videos")
        os.makedirs(video_output_folder, exist_ok=True)

        videos = playlist.videos
        with tqdm(total=len(videos), desc=f"Downloading {teleplay_name}", unit='video') as pbar:
            for idx, video in enumerate(videos, start=1):
                video_filename = f"{normalize_text(video.title)}.mp4"
                file_path = os.path.join(video_output_folder, video_filename)

                if os.path.exists(file_path):
                    print(f"Skipped (already exists): {video.title}")
                    pbar.update(1)
                    continue
                try:
                    video.streams.get_highest_resolution().download(
                        output_path=video_output_folder,
                        filename=video_filename
                    )
                    print(f"Downloaded: {video.title}")
                    pbar.update(1)
                except Exception as e:
                    print(f"Failed to download video {video.title}: {e}")
                    if "is age restricted, and can't be accessed without logging in." in str(e):
                        print("Age restricted : " + teleplay_name + "\t" + video.title)
                        continue
                    print(f"Retry to download video {video.title}")
                    while True:
                        try:
                            os.remove(file_path)
                            video.streams.get_highest_resolution().download(
                                output_path=video_output_folder,
                                filename=video_filename
                            )
                            break
                        except Exception as e:
                            continue
                    print(f"Downloaded: {video.title}")
                    pbar.update(1)

    def download_and_process(self):
        ps = []
        for name, url in self.teleplay_download_url_dict.items():
            ps.append(
                multiprocessing.Process(
                    target=self.process_one,
                    args=(name, url,)
                )
            )
        for i in ps:
            i.start()
        for i in ps:
            i.join()

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def main(teleplay_download_url, output_path):
    os.makedirs(output_path, exist_ok=True)
    teleplay_download_url_dict = read_json(teleplay_download_url)

    # DownloadMethod1(teleplay_download_url_dict, output_path).download_and_process()
    DownloadMethod2(teleplay_download_url_dict, output_path).download_and_process()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("teleplay_download_url", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    main(**vars(args))
