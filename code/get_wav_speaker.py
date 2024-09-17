import os
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from tool import load_npy, calculate_similarity, read_json, write_json


def find_best_speaker(
        teleplay,
        obj_representation,
        reference_speaker_embedding_save_dir
):
    speaker_dir = os.path.join(reference_speaker_embedding_save_dir, teleplay)
    max_similarity = -1
    best_speaker = None

    for speaker in os.listdir(speaker_dir):
        speaker_folder = os.path.join(speaker_dir, speaker)
        speaker_representation = load_npy(os.path.join(speaker_folder, "whole.npy"))
        similarity = calculate_similarity(obj_representation, speaker_representation)
        if similarity > max_similarity:
            max_similarity = similarity
            best_speaker = speaker
    return best_speaker


def process_objects(
        objects,
        segment_wav_speaker_embedding_save_dir,
        reference_speaker_embedding_save_dir
):
    teleplay, episode, wav_objects = objects
    results = []
    for obj in tqdm(wav_objects):
        teleplay = obj['teleplay'] if teleplay is None else teleplay
        episode = obj["episode"] if episode is None else episode
        wav_name = obj['wav_name']
        npy_path = os.path.join(segment_wav_speaker_embedding_save_dir, teleplay, episode, f"{wav_name}.npy")
        obj_representation = load_npy(npy_path)
        best_speaker = find_best_speaker(teleplay, obj_representation, reference_speaker_embedding_save_dir)
        obj['speaker'] = best_speaker
        results.append((teleplay, episode, obj))
    return results


def split_data(data, num_chunks):
    chunks = []
    if isinstance(data, list):
        avg_chunk_size = len(data) // num_chunks
        remainder = len(data) % num_chunks
        start = 0

        for i in range(num_chunks):
            end = start + avg_chunk_size + (1 if i < remainder else 0)
            chunks.append((None, None, data[start:end]))
            start = end
    elif isinstance(data, dict):
        for teleplay in data.keys():
            for episode in data[teleplay].keys():
                teleplay_episode_wav_objects = data[teleplay][episode]
                chunks.append((teleplay, episode, teleplay_episode_wav_objects))

    return chunks


def process_wrapper(args):
    return process_objects(*args)


def get_wav_speaker_label(
        input_data,
        segment_wav_speaker_embedding_save_dir,
        reference_speaker_embedding_save_dir,
        segment_wav_json_path
):
    assert isinstance(input_data, list) or isinstance(input_data, dict)
    num_workers = cpu_count()
    data_chunks = split_data(input_data, num_workers)
    pool_args = []
    for data_chunk in data_chunks:
        pool_args.append(
            (
                data_chunk,
                segment_wav_speaker_embedding_save_dir,
                reference_speaker_embedding_save_dir
            )
        )
    with Pool(cpu_count()) as pool:
        results_chunks = pool.map(process_wrapper, pool_args)

    results = None
    if isinstance(input_data, dict):
        results = {}
        for result in results_chunks:
            for teleplay, episode, wav_object in result:
                if teleplay not in results.keys():
                    results[teleplay] = {}
                if episode not in results[teleplay].keys():
                    results[teleplay][episode] = []
                results[teleplay][episode].append(wav_object)
    elif isinstance(input_data, list):
        results = [item for sublist in results_chunks for item in sublist]

    write_json(results, segment_wav_json_path)
