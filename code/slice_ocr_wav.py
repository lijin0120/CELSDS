import os

from pydub import AudioSegment
from tqdm import tqdm

from extract_speaker_embedding import extract_speaker_embedding
from get_wav_speaker import get_wav_speaker_label
from tool import read_json, write_json, get_episode, divide_list


def get_teleplay_episode_wav_dict(data_root_dir):
    teleplay_episode_wav_dict = {}
    for teleplay in os.listdir(data_root_dir):
        if teleplay in ['Done', 'log']:
            continue
        teleplay_episode_wav_dict[teleplay] = {}
        teleplay_wav_dir = os.path.join(data_root_dir, teleplay, "wavs")
        for teleplay_wav in os.listdir(teleplay_wav_dir):
            teleplay_wav_episode = get_episode(teleplay_wav)
            teleplay_wav_episode = f"{int(teleplay_wav_episode):03}"
            teleplay_wav_path = os.path.join(teleplay_wav_dir, teleplay_wav)
            teleplay_episode_wav_dict[teleplay][teleplay_wav_episode] = teleplay_wav_path
    return teleplay_episode_wav_dict


def slice_wav(
        original_wav,
        start_time,
        end_time,
        output_path
):
    audio = AudioSegment.from_wav(original_wav)
    segment_audio = audio[start_time:end_time]
    segment_audio.export(output_path, format="wav")


def slice_ocr_wav_and_generate_segment_wav_json(
        teleplay_episode_wav_dict,
        ocr_results_dir,
        time_bias,
        segment_wav_save_dir,
        segment_wav_json_path
):
    segment_wav_results = {}
    for teleplay in tqdm(os.listdir(ocr_results_dir)):
        if teleplay in ['Done', 'log']:
            continue
        segment_wav_results[teleplay] = {}
        teleplay_ocr_results_dir = os.path.join(ocr_results_dir, teleplay)
        for teleplay_episode_ocr_result_file in tqdm(os.listdir(teleplay_ocr_results_dir)):
            episode = teleplay_episode_ocr_result_file.split("_")[1]
            episode = f"{int(episode):03}"
            segment_wav_results[teleplay][episode] = []
            teleplay_episode_wav = teleplay_episode_wav_dict[teleplay][episode]
            teleplay_episode_segment_wav_output_dir = os.path.join(segment_wav_save_dir, teleplay, episode)
            os.makedirs(teleplay_episode_segment_wav_output_dir, exist_ok=True)

            teleplay_episode_ocr_result_data = read_json(
                os.path.join(teleplay_ocr_results_dir, teleplay_episode_ocr_result_file)
            )
            for idx, item in enumerate(teleplay_episode_ocr_result_data):
                start_time, end_time = item["time"]
                interval = end_time - start_time
                start_time = start_time - interval if start_time > interval else 0.0
                if idx >= 1:
                    start_time = start_time if start_time > teleplay_episode_ocr_result_data[idx - 1]["time"][1] else \
                        teleplay_episode_ocr_result_data[idx - 1]["time"][1]
                end_time = end_time - time_bias if end_time - start_time > time_bias else end_time
                item["time"] = [start_time, end_time]
                start_time, end_time = int(start_time * 1000), int(end_time * 1000)
                segment_wav_name = f"{teleplay}_{episode}_{start_time}_{end_time}"
                segment_wav_path = os.path.join(
                    teleplay_episode_segment_wav_output_dir,
                    f"{segment_wav_name}.wav",
                )
                if not os.path.exists(segment_wav_path):
                    slice_wav(
                        teleplay_episode_wav,
                        start_time,
                        end_time,
                        segment_wav_path
                    )

                segment_wav = AudioSegment.from_wav(segment_wav_path)
                segment_wav_results[teleplay][episode].append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "wav_name": segment_wav_name,
                    "path": segment_wav_path,
                    "duration": segment_wav.duration_seconds,
                    "num_sample": int(segment_wav.duration_seconds * segment_wav.frame_rate),
                    "sample_rate": segment_wav.frame_rate,
                    "ocr_text": "，".join(item["text"]) + "。"
                })
            segment_wav_results[teleplay][episode] = sorted(segment_wav_results[teleplay][episode],
                                                            key=lambda x: int(x['start_time']))

    write_json(segment_wav_results, segment_wav_json_path)
    return segment_wav_results


def extract_segment_wav_speaker_embedding(
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        segment_wav_json_path,
        segment_wav_save_dir,
        segment_wav_speaker_embedding_save_dir,
        segment_wav_16k_save_dir
):
    segment_wav_objects = read_json(segment_wav_json_path)
    input_data = []
    for teleplay in segment_wav_objects.keys():
        for episode in segment_wav_objects[teleplay].keys():
            for segment_wav_object in segment_wav_objects[teleplay][episode]:
                wav_path = segment_wav_object["path"]
                sample_rate = segment_wav_object["sample_rate"]
                input_data.append(f"{wav_path}\t{sample_rate}")

    extract_speaker_embedding(
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        input_data,
        segment_wav_save_dir,
        segment_wav_speaker_embedding_save_dir,
        segment_wav_16k_save_dir
    )


def get_segment_wav_speaker_label(
        segment_wav_json_path,
        segment_wav_speaker_embedding_save_dir,
        reference_speaker_embedding_save_dir
):
    input_data = read_json(segment_wav_json_path)
    get_wav_speaker_label(
        input_data,
        segment_wav_speaker_embedding_save_dir,
        reference_speaker_embedding_save_dir,
        segment_wav_json_path
    )


def merge_segment_wav_to_dialogue_segment(
        segment_wav_json_path,
        divide_num,
        dialogue_segment_save_path
):
    segment_wav_result = read_json(segment_wav_json_path)

    dialogue_segments = {}
    for teleplay in segment_wav_result.keys():
        dialogue_segments[teleplay] = {}
        for episode in segment_wav_result[teleplay].keys():
            dialogue_segments[teleplay][episode] = []
            teleplay_episode_segment_wav_objects = segment_wav_result[teleplay][episode]
            divided_teleplay_episode_segment_wav_objects = divide_list(teleplay_episode_segment_wav_objects, divide_num)
            for dialogue_segment_idx, dialogue_segment in enumerate(divided_teleplay_episode_segment_wav_objects):
                text = ""
                wavs_path = []
                for item in dialogue_segment:
                    text = text + item["speaker"] + "：“" + item["ocr_text"] + "”"
                    wavs_path.append(item["path"])
                dialogue_segments[teleplay][episode].append({
                    "dialogue_segment_index": dialogue_segment_idx,
                    "dialogue_segment": text,
                    "wavs_path": wavs_path
                })
    write_json(dialogue_segments, dialogue_segment_save_path)


def main(
        audio_root_dir,
        filtered_and_merged_ocr_results_dir,
        segment_wav_save_dir,
        segment_wav_json_path,
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        segment_wav_speaker_embedding_save_dir,
        segment_wav_16k_save_dir,
        reference_speaker_embedding_save_dir,
        dialogue_segment_save_path,
        time_bias,
        divide_num
):
    teleplay_episode_wav_dict = get_teleplay_episode_wav_dict(audio_root_dir)
    slice_ocr_wav_and_generate_segment_wav_json(
        teleplay_episode_wav_dict,
        filtered_and_merged_ocr_results_dir,
        time_bias,
        segment_wav_save_dir,
        segment_wav_json_path
    )
    extract_segment_wav_speaker_embedding(
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        segment_wav_json_path,
        segment_wav_save_dir,
        segment_wav_speaker_embedding_save_dir,
        segment_wav_16k_save_dir
    )
    get_segment_wav_speaker_label(
        segment_wav_json_path,
        segment_wav_speaker_embedding_save_dir,
        reference_speaker_embedding_save_dir,
    )
    merge_segment_wav_to_dialogue_segment(
        segment_wav_json_path,
        divide_num,
        dialogue_segment_save_path
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_root_dir", type=str)
    parser.add_argument("filtered_and_merged_ocr_results_dir", type=str)
    parser.add_argument("segment_wav_save_dir", type=str)
    parser.add_argument("segment_wav_json_path", type=str)
    parser.add_argument("available_gpus", type=list)
    parser.add_argument("max_processes_per_gpu", type=int)
    parser.add_argument("speaker_model_path", type=str)
    parser.add_argument("segment_wav_speaker_embedding_save_dir", type=str)
    parser.add_argument("segment_wav_16k_save_dir", type=str)
    parser.add_argument("reference_speaker_embedding_save_dir", type=str)
    parser.add_argument("dialogue_segment_save_path", type=str)
    parser.add_argument("--time_bias", type=int, default=400)
    parser.add_argument("--divide_num", type=int, default=15)
    args = parser.parse_args()

    main(**vars(args))
