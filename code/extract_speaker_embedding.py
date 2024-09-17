from multiprocessing import Pool
import os

import librosa as lib
import torch
import wespeaker
from tqdm import tqdm

from tool import npywrite, audiowrite, read_json


def process_input_data(
        input_data,
        gpu_id,
        speaker_model_path,
        wav_save_dir,
        wav_speaker_embedding_save_dir,
        wav_16k_save_dir
):
    model = wespeaker.load_model_local(speaker_model_path)
    model.set_gpu(gpu_id)
    with torch.no_grad():
        for data in tqdm(input_data):
            path, sr = data.strip().split("\t")
            save_path = path.replace(
                wav_save_dir,
                wav_speaker_embedding_save_dir
            ).split(".")[0] + ".npy"
            if os.path.exists(save_path):
                continue
            save_path_16k = path.replace(
                wav_save_dir,
                wav_16k_save_dir
            ).split(".")[0] + ".wav"

            if int(sr) != 16000:
                if os.path.exists(save_path_16k):
                    pass
                else:
                    wav_16k, sr = lib.load(path, sr=16000)
                    audiowrite(save_path_16k, wav_16k)
                path = save_path_16k
            try:
                embedding = model.extract_embedding(path).detach().cpu().numpy()
                npywrite(save_path, embedding)
            except:
                print(path)


def process_wrapper(args):
    return process_input_data(*args)


def extract_speaker_embedding(
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        input_data,
        wav_save_dir,
        wav_speaker_embedding_save_dir,
        wav_16k_save_dir
):
    num_gpus = len(available_gpus)
    total_processes = num_gpus * max_processes_per_gpu

    print(len(input_data))

    chunk_size = len(input_data) // total_processes
    chunks = [input_data[i * chunk_size: (i + 1) * chunk_size] for i in range(total_processes)]
    remaining_lines = input_data[total_processes * chunk_size:]

    if remaining_lines:
        chunks[-1].extend(remaining_lines)

    pool_args = []
    for i, chunk in enumerate(chunks):
        gpu_id = available_gpus[i % num_gpus]
        pool_args.append(
            (
                chunk,
                gpu_id,
                speaker_model_path,
                wav_save_dir,
                wav_speaker_embedding_save_dir,
                wav_16k_save_dir
            )
        )

    with Pool(processes=total_processes) as pool:
        pool.map(process_wrapper, pool_args)


def main(
        speaker_wav_path,
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        wav_save_dir,
        wav_speaker_embedding_save_dir,
        wav_16k_save_dir
):
    speaker_wav_objects = read_json(speaker_wav_path)
    input_data = []
    for teleplay in speaker_wav_objects.keys():
        for speaker in speaker_wav_objects[teleplay].keys():
            wav_path = speaker_wav_objects[teleplay][speaker]
            _, sample_rate = lib.load(wav_path, sr=None)
            input_data.append(f"{wav_path}\t{sample_rate}")

    extract_speaker_embedding(
        available_gpus,
        max_processes_per_gpu,
        speaker_model_path,
        input_data,
        wav_save_dir,
        wav_speaker_embedding_save_dir,
        wav_16k_save_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("speaker_wav_path", type=str)
    parser.add_argument("available_gpus", type=list)
    parser.add_argument("max_processes_per_gpu", type=int)
    parser.add_argument("speaker_model_path", type=str)
    parser.add_argument("wav_save_dir", type=str)
    parser.add_argument("wav_speaker_embedding_save_dir", type=str)
    parser.add_argument("wav_16k_save_dir", type=str)
    args = parser.parse_args()

    main(**vars(args))
