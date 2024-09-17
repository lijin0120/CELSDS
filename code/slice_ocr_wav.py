import os

from pydub import AudioSegment
from tqdm import tqdm

from extract_speaker_embedding import extract_speaker_embedding
from get_wav_speaker import get_wav_speaker_label
from tool import read_json, write_json, get_teleplay_wav_episode, divide_list


def get_teleplay_episode_wav_dict(
        teleplay_episode_dict,
        data_root_dir
):
    teleplay_episode_wav_dict = {}
    for teleplay in os.listdir(data_root_dir):
        if teleplay in teleplay_episode_dict.keys():
            teleplay_episode_wav_dict[teleplay] = {}
            teleplay_wav_dir = os.path.join(data_root_dir, teleplay, "wavs")
            for teleplay_wav in os.listdir(teleplay_wav_dir):
                teleplay_wav_episode = get_teleplay_wav_episode(teleplay_wav)
                if teleplay_wav_episode == teleplay_episode_dict[teleplay]:
                    continue
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


def slice_ocr_wav_and_generate_single_wav_json(
        teleplay_episode_dict,
        teleplay_episode_wav_dict,
        ocr_results_dir,
        time_bias,
        segment_wav_output_dir,
        single_wav_json_result_output_path
):
    single_wav_results = {}
    for teleplay in tqdm(os.listdir(ocr_results_dir)):
        if teleplay not in teleplay_episode_dict.keys():
            continue
        # print(teleplay)
        single_wav_results[teleplay] = {}
        teleplay_ocr_results_dir = os.path.join(ocr_results_dir, teleplay)
        for teleplay_episode_ocr_result_file in tqdm(os.listdir(teleplay_ocr_results_dir)):
            episode = teleplay_episode_ocr_result_file.split("_")[1]
            episode = f"{int(episode):03}"
            if episode == teleplay_episode_dict[teleplay]:
                continue
            # print(episode)
            single_wav_results[teleplay][episode] = []
            teleplay_episode_wav = teleplay_episode_wav_dict[teleplay][episode]
            teleplay_episode_segment_wav_output_dir = os.path.join(segment_wav_output_dir, teleplay, episode)
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
                single_wav_results[teleplay][episode].append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "wav_name": segment_wav_name,
                    "path": segment_wav_path,
                    "duration": segment_wav.duration_seconds,
                    "num_sample": int(segment_wav.duration_seconds * segment_wav.frame_rate),
                    "sample_rate": segment_wav.frame_rate,
                    "ocr_text": "，".join(item["text"]) + "。"
                })
            single_wav_results[teleplay][episode] = sorted(single_wav_results[teleplay][episode],
                                                           key=lambda x: int(x['start_time']))

    write_json(single_wav_results, single_wav_json_result_output_path)
    return single_wav_results


def extract_single_wav_speaker_embedding(
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        single_wav_path,
        wav_root_dir,
        wav_speaker_embedding_root_dir,
        wav_16k_root_dir
):
    single_wav_objects = read_json(single_wav_path)
    input_data = []
    for teleplay in single_wav_objects.keys():
        for episode in single_wav_objects[teleplay].keys():
            for single_wav_object in single_wav_objects[teleplay][episode]:
                wav_path = single_wav_object["path"]
                sample_rate = single_wav_object["sample_rate"]
                input_data.append(f"{wav_path}\t{sample_rate}")

    extract_speaker_embedding(
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        input_data,
        wav_root_dir,
        wav_speaker_embedding_root_dir,
        wav_16k_root_dir
    )


def get_single_wav_speaker_label(
        single_wav_result_path,
        segment_wav_speaker_embedding_root_dir,
        teleplay_speaker_embedding_root_dir,
        output_file_path
):
    input_data = read_json(single_wav_result_path)
    get_wav_speaker_label(
        input_data,
        segment_wav_speaker_embedding_root_dir,
        teleplay_speaker_embedding_root_dir,
        output_file_path
    )


def merge_single_wav_to_plot(
        single_wav_result_path,
        divide_num,
        plot_result_path
):
    single_wav_result = read_json(single_wav_result_path)

    plots = {}
    for teleplay in single_wav_result.keys():
        plots[teleplay] = {}
        for episode in single_wav_result[teleplay].keys():
            plots[teleplay][episode] = []
            teleplay_episode_single_wav_objects = single_wav_result[teleplay][episode]
            divided_teleplay_episode_single_wav_objects = divide_list(teleplay_episode_single_wav_objects, divide_num)
            for plot_idx, plot in enumerate(divided_teleplay_episode_single_wav_objects):
                text = ""
                wavs_path = []
                for item in plot:
                    text = text + item["speaker"] + "：“" + item["ocr_text"] + "”"
                    wavs_path.append(item["path"])
                plots[teleplay][episode].append({
                    "plot_index": plot_idx,
                    "text": text,
                    "wavs_path": wavs_path
                })
    write_json(plots, plot_result_path)


if __name__ == "__main__":
    teleplay_episode_dict = {
        "书剑恩仇录": "020",
        "何以笙箫默": "026",
        "侠客行": "023",
        "大明王朝1566": "042",
        "天涯·明月·刀2012": "039",
        "宫锁心玉": "025",
        "庆余年": "021",
        "新边城浪子": "047",
        "楚留香新传": "038",
        "沉香如屑": "058",
        "流星·蝴蝶·剑": "009",
        "还珠格格": "003",
    }
    data_root_dir = "/CDShare2/2023/wangtianrui/dataset/SUSC/v2"
    filtered_and_merged_ocr_results_dir = "/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/train_set/ocr/filtered_and_merged_5fps"
    time_bias = 400
    segment_wav_root_dir = "/CDShare2/2023/wangtianrui/dataset/SUSC/train/segment_wavs/ocr/wavs"
    single_wav_result_path = "/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/train_set/train_ocr_single_wav.json"
    divide_num = 15
    plot_result_path = "/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/train_set/train_plots.json"

    available_gpus = [5]
    max_processes_per_gpu = 1
    speaker_model_path = '/Work20/2023/wangtianrui/model_temp/wespeaker/voxceleb_resnet293_LM'
    segment_wav_speaker_embedding_root_dir = "/CDShare2/2023/wangtianrui/dataset/SUSC/train/segment_wavs/ocr/speaker_embedding"
    segment_wav_16k_root_dir = "/CDShare2/2023/wangtianrui/dataset/SUSC/train/segment_wavs/ocr/wavs_16k"
    teleplay_speaker_embedding_root_dir = "/CDShare2/2023/wangtianrui/dataset/SUSC/v2_16k"

    # teleplay_episode_wav_dict = get_teleplay_episode_wav_dict(teleplay_episode_dict, data_root_dir)
    # slice_ocr_wav_and_generate_single_wav_json(
    #     teleplay_episode_dict,
    #     teleplay_episode_wav_dict,
    #     filtered_and_merged_ocr_results_dir,
    #     time_bias,
    #     segment_wav_root_dir,
    #     single_wav_result_path
    # )
    # extract_single_wav_speaker_embedding(
    #     available_gpus,
    #     max_processes_per_gpu,
    #     speaker_model_path,
    #     single_wav_result_path,
    #     segment_wav_root_dir,
    #     segment_wav_speaker_embedding_root_dir,
    #     segment_wav_16k_root_dir
    # )
    # get_single_wav_speaker_label(
    #     single_wav_result_path,
    #     segment_wav_speaker_embedding_root_dir,
    #     teleplay_speaker_embedding_root_dir,
    #     single_wav_result_path
    # )
    # merge_single_wav_to_plot(
    #     single_wav_result_path,
    #     divide_num,
    #     plot_result_path
    # )
