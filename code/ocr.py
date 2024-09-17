import json
import multiprocessing
import os

from moviepy.editor import *
from paddleocr import PaddleOCR
from tqdm import tqdm

from tool import time_to_seconds, read_json, get_episode


def video_ocr(
        video_path,
        video_time_range,
        video_cut,
        video_fps,
        ocr_result_path
):
    clip = VideoFileClip(video_path)
    print("Ori FPS:{} Duration:{} Height:{} Width:{}".format(clip.fps, clip.duration, clip.w, clip.h))

    start_time, end_time = video_time_range
    if start_time is None:
        cut_clip = clip
    else:
        cut_clip = clip.subclip(start_time, end_time)
    cut_clip = cut_clip.crop(
        x1=video_cut[1],
        x2=clip.w - video_cut[2],
        y2=clip.h - video_cut[3],
        height=clip.w // video_cut[0] - video_cut[4],
    )
    cut_clip = cut_clip.set_fps(video_fps)
    print("Cut FPS:{} Duration:{} Height:{} Width:{}".format(cut_clip.fps, cut_clip.duration, cut_clip.w, cut_clip.h))

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
    frame_rate = 1 / cut_clip.fps

    results = []
    for cnt, cur_frame in enumerate(cut_clip.iter_frames()):
        try:
            result = ocr.ocr(cur_frame, det=True)
            if result is not None:
                see = []
                for sub_result in result[0]:
                    see.append(sub_result[1])
                cur_time = float(frame_rate * (cnt + 1) + start_time)
                results.append({'time': cur_time, "text": see})
        except Exception:
            pass

    with open(ocr_result_path, 'w', encoding='utf-8') as of:
        json.dump(results, of, ensure_ascii=False, indent=4)


def process(
        video_root_dir,
        video_time_range,
        video_credits_fixed_limit,
        video_cut,
        video_fps,
        output_path
):
    pool_args = []
    for teleplay in os.listdir(video_root_dir):
        if teleplay in ['Done', 'log']:
            continue
        teleplay_videos_dir = os.path.join(video_root_dir, teleplay, "videos")
        for teleplay_video in os.listdir(teleplay_videos_dir):
            teleplay_video_episode = get_episode(teleplay_video.split(".")[0])
            teleplay_video_episode = f"{int(teleplay_video_episode):03}"
            if teleplay == "新边城浪子" and teleplay_video_episode == "4":
                continue
            video_path = os.path.join(teleplay_videos_dir, teleplay_video)
            if teleplay_video_episode in video_time_range[teleplay].keys():
                video_time_range = video_time_range[teleplay][teleplay_video_episode]
                if teleplay in video_credits_fixed_limit.keys():
                    video_credits_fixed_limit = video_credits_fixed_limit[teleplay].copy()
                else:
                    video_credits_fixed_limit = [0, 0]
                if teleplay == "书剑恩仇录" and teleplay_video_episode == "1":
                    video_credits_fixed_limit[0] = 188
                if teleplay == "新边城浪子" and teleplay_video_episode == "1":
                    video_credits_fixed_limit[0] = 120
                    video_credits_fixed_limit[1] = 150

                video_time_range = [
                    time_to_seconds(video_time_range[0]) + video_credits_fixed_limit[0],
                    time_to_seconds(video_time_range[1]) - video_credits_fixed_limit[1]
                ]
            else:
                video_time_range = [None, None]

            teleplay_ocr_result_dir = os.path.join(output_path, teleplay)
            os.makedirs(teleplay_ocr_result_dir, exist_ok=True)
            ocr_result_path = os.path.join(
                teleplay_ocr_result_dir,
                f"{teleplay}_{teleplay_video_episode}.json"
            )
            if not os.path.exists(ocr_result_path):
                pool_args.append(
                    (
                        video_path,
                        video_time_range,
                        video_cut[teleplay],
                        video_fps,
                        ocr_result_path
                    )
                )

    print(len(pool_args))
    with multiprocessing.Pool(processes=32) as pool:
        list(tqdm(pool.starmap(video_ocr, pool_args), total=len(pool_args)))


def main(video_root_dir, ocr_configure, output_path):
    ocr_config = read_json(ocr_configure)
    video_time_range = ocr_config["video_time_range"]
    video_credits_fixed_limit = ocr_config["video_credits_fixed_limit"]
    # cut_rate  x1  x2  y1  height
    video_cut = ocr_config["video_cut"]
    video_fps = ocr_config["video_fps"]

    process(
        video_root_dir,
        video_time_range,
        video_credits_fixed_limit,
        video_cut,
        video_fps,
        output_path
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_root_dir", type=str)
    parser.add_argument("ocr_configure", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    main(**vars(args))
