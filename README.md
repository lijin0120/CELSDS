<div align="center">
    <h1>
    CELSDS
    </h1>
    <p>
    This is the official implement of A Chinese Expressive Long-dialogue Speech Dataset with Scripts (Submitted to ICASSP 2025). The data is partially stored at <a href="https://drive.google.com/drive/folders/1dBVtVzx-HJuRwrBKXdFWzSAbk1_sasxQ?usp=sharing">CELSDS</a> .
    </p>
    <p>
    </p>
</div>

## Step 1: Preparation Before Downloading the Video

Save the download links of the TV series in a JSON file, such as `data/teleplay_download_url.json`.
An example of the JSON content is shown below:

```
{
  "大明王朝1566": "https://www.youtube.com/playlist?list=PLUDHmjG40FQBGQ54JEtrDZ3SiL018hCB6",
  ...
}
```

## Step 2: Download and Process Videos

To download the videos for each TV series and extract the corresponding audio, run

```sh
python code/download_and_process.py ${teleplay_download_url} ${output_path}
```

When extracting audio from videos, the `ffmpeg` tool is required,
so make sure `ffmpeg` is installed and working correctly.

Parameters:

* `teleplay_download_url`: The JSON file containing the download links prepared in Step 1.
* `output_path`: The directory where the downloaded videos and extracted audio will be saved.

Example usage:

```sh
python code/download_and_process.py data/teleplay_download_url.json data/video_and_audio
```

## Step 3: Preparation Before OCR

To remove the influence of video intro and outro sequences, and to improve the accuracy of OCR, different OCR settings
need to be configured for each TV series.
These settings should be saved in a JSON file, such as `data/teleplay_ocr_configure.json`.
An example of the JSON content is shown below:

```
{
  "video_fps": 5,
  "video_cut": {
    "庆余年": [15, 400, 400, 45, 45],
    ...
  },
  "video_credits_fixed_limit": {
    "庆余年": [95, 165],
    ...
  },
  "video_time_range": {
    "庆余年": {
      "1": ["00:01:30", "00:44:46"],
      ...
    },
    ...
  }
}
```

Parameters:

* `video_fps`: The frames-per-second rate to be applied to the video.
* `video_cut`: Parameters to crop each frame to better match subtitle boundaries.
  The first and fifth values correspond to modifications for the parameter `height`, the second value to `x1`, the third
  to `x2`, and the fourth to `y2`.
  For details on `x1`, `x2`, `y2`, and `height`, refer to the `moviepy` library documentation.
* `video_credits_fixed_limit`: The fixed duration of intro and outro sequences for each TV series.
  If the duration varies across episodes, this parameter can be ignored.
* `video_time_range`: The start and end time of each episode.
  If the intro and outro sequences are not of fixed duration, you need to annotate the start and end time excluding
  them.
  Otherwise, just mark the actual start and end time of the video, ignoring whether the intro and outro exist,
  as `video_credits_fixed_limit` already handles their fixed durations.

## Step 4: OCR

To obtain the transcripts corresponding to the speech, use `PaddleOCR` to extract subtitles from the videos, run

```sh
python code/ocr.py ${video_root_dir} ${ocr_configure} ${output_path}
```

Parameters:

* `video_root_dir`: The directory where the videos downloaded in Step 2 are saved.
* `ocr_configure`: The JSON file containing the OCR configurations prepared in Step 3.
* `output_path`: The directory where the OCR results will be saved.

Example usage:

```sh
python code/ocr.py data/video_and_audio data/teleplay_ocr_configure.json data/ocr/original
```

## Step 5: Filter and Merge OCR Results

To merge duplicate OCR results and filter out those with poor quality, run

```sh
python code/filter_and_merge.py ${original_ocr_result_dir} ${output_path}
```

Parameters:

* `original_ocr_result_dir`:  The directory containing the OCR results obtained in Step 4.
* `output_path`: The directory where the filtered and merged OCR results will be saved.

Example usage:

```sh
python code/filter_and_merge_ocr_results.py data/ocr/original/ data/ocr/filter_and_merged
```

## Step 6: Preparation Before Extracting Speaker Embedding for Reference Speaker Speech

Before extracting the speaker embeddings for the reference speaker speech, we need to save the path of reference speaker
speech in a JSON file, such as `data/teleplay_speaker_wav.json`.
An example of the JSON file is as follows:

```
{
  "大明王朝1566": {
    "谭纶": "/path/to/大明王朝1566/谭纶/whole.wav",
    "朱翊钧": "/path/to/大明王朝1566/朱翊钧/whole.wav",
    ...
  },
  ...
}
```

The file `reference_speaker_wav.zip` in the
link https://drive.google.com/drive/folders/1dBVtVzx-HJuRwrBKXdFWzSAbk1_sasxQ?usp=sharing
provides all the reference speaker speech and speaker embeddings used in this dataset.

## Step 7: Extracting Speaker Embeddings for Reference Speaker Speech

To annotate speaker labels for the speech segments, we need to extract the speaker embeddings for the reference speaker
speech in advance, run

```sh
python code/extract_speaker_embedding.py ${speaker_wav_path} ${available_gpus} ${max_processes_per_gpu} ${speaker_model_path} ${wav_save_dir} ${wav_speaker_embedding_save_dir} ${wav_16k_save_dir}
```

Parameters:

* `speaker_wav_path`: The JSON file that stores the paths of the reference speaker speech, annotated in Step 6.
* `available_gpus`: A list of GPU ids available for extracting speaker embeddings.
* `max_processes_per_gpu`: The maximum number of tasks per GPU for parallel processing during embedding extraction.
* `speaker_model_path`: The path to the model used for extracting speaker embeddings.
* `wav_save_dir`: The directory where the reference speaker speech is saved.
* `wav_speaker_embedding_save_dir`: The directory where the extracted speaker embeddings are saved.
* `wav_16k_save_dir`: The directory where the 16k sampled reference speaker speech is saved.
  The ResNet293 model requires the audio to be converted to a 16k sample rate for speaker embedding extraction.

Example usage:

```sh
python code/extract_speaker_embedding.py data/teleplay_speaker_wav.json [0,1] 2 /path/to/wespeaker/voxceleb_resnet293_LM data/reference_speaker_wav data/reference_speaker_wav_speaker_embedding data/reference_speaker_wav_16k
```

## Step 8: Speech Segmentation, Annotating Weak Speaker Labels, and Dividing Dialogue Segments

After filtering and merging the OCR results, split the corresponding transcribed speech based on the timestamps.
After extracting speaker embeddings, annotate the weak speaker labels for the speech segments.
Divide the complete dialogues from each episode into multiple dialogue segments. Run

```sh
python code/slice_ocr_wav.py ${audio_root_dir} ${filtered_and_merged_ocr_results_dir} ${segment_wav_save_dir} ${segment_wav_json_path} ${available_gpus} ${max_processes_per_gpu} ${speaker_model_path} ${segment_wav_speaker_embedding_save_dir} ${segment_wav_16k_save_dir} ${reference_speaker_embedding_save_dir} ${dialogue_segment_save_path}
```

Parameters:

* `audio_root_dir`: The directory where the audio extracted in Step 2 is saved.
* `filtered_and_merged_ocr_results_dir`: The directory where the filtered and merged OCR results from Step 5 are saved.
* `segment_wav_save_dir`: The directory where the speech segments are saved.
* `segment_wav_json_path`: The path to the JSON file that stores information about the speech segments.
* `available_gpus`:  A list of GPU ids available for extracting speaker embeddings.
* `max_processes_per_gpu`: The maximum number of tasks per GPU for parallel processing during embedding extraction.
* `speaker_model_path`: The path to the model used for extracting speaker embeddings.
* `segment_wav_speaker_embedding_save_dir`: The directory where the extracted speaker embeddings for the speech segments
  are saved.
* `segment_wav_16k_save_dir`: The directory where the 16k sampled speech segments are saved. The ResNet293 model
  requires the audio to be converted to a 16k sample rate for speaker embedding extraction.
* `reference_speaker_embedding_save_dir`: The directory where the reference speaker embeddings extracted in Step 7 are
  saved.
* `dialogue_segment_save_path`: The path to the JSON file where the dialogue segments are saved.

Optional parameters:

* `--time_bias`: The time offset to adjust the end timestamp of the speech. The end timestamp is calculated based on the
  subtitle timestamps,
  which are typically delayed compared to the corresponding speech. This parameter compensates for that delay. Unit:
  milliseconds.
* `--divide_num`: The number of dialogue segments to split from the complete dialogue of each episode.

Example usage:

```sh
python code/slice_ocr_wav.py data/video_and_audio data/ocr/filter_and_merged data/segment_wav data/teleplay_segment_wav.json [0,1] 2 /path/to/wespeaker/voxceleb_resnet293_LM data/segment_wav_speaker_embedding data/segment_wav_16k data/reference_speaker data/teleplay_dialogue_segment.json
```

```sh
python code/slice_ocr_wav.py data/video_and_audio data/ocr/filter_and_merged data/segment_wav data/teleplay_segment_wav.json [0,1] 2 /path/to/wespeaker/voxceleb_resnet293_LM data/segment_wav_speaker_embedding data/segment_wav_16k data/reference_speaker data/teleplay_dialogue_segment.json --time_bias 400 --divide_num 15
```

The files `test_segment_wav.json` and `train_segment_wav.json` in the
link https://drive.google.com/drive/folders/1dBVtVzx-HJuRwrBKXdFWzSAbk1_sasxQ?usp=sharing
provide the speech segments and their related information for this dataset.

## Step 9: Preparation Before Script Generation

Before generating the script, we need to collect summaries of each episode for every TV series in a JSON file,
such as `data/teleplay_episode_summary.json`. An example of the JSON file is as follows:

```
{
  "庆余年": {
    "001": {
      "source1": "大学生张庆想请求叶教授担任自己的导师，却遭到了毫不留情的拒绝...",
      "source2": "在范闲的记忆中，自己是现代社会一个患了重症肌无力的将死之人...",
      ...
    },
    ...
  },
  ...
}
```

## Step 10: Generate Scripts

Before using ChatGPT to generate the script, we can modify the relevant parameters for calling the ChatGPT API in
the `code/llm.py` file, as shown below:

```
llm_name = "gpt-3.5-turbo"
llm_configure = {
    "openai_api_key": "openai_api_key",
    "openai_base_url": 'openai_base_url',
}
```

To generate scripts using ChatGPT, run

```sh
python code/llm.py ${dialogue_segment} ${episode_summary} ${train_or_test} ${output_path}
```

Parameters:

* `dialogue_segment`: The path to the JSON file where the dialogue segments are saved from Step 8.
* `episode_summary`: The path to the JSON file where the episode summaries for each TV series are saved from Step 9.
* `train_or_test`: Specifies whether to use the prompt for training set or test set, as the prompts differ for the two.
* `output_path`: The path where the final output JSON file will be saved.

Optional parameters:

* `--max_retry_num`: Maximum number of retry attempts for the ChatGPT API call.
* `--retry_delay`: Delay between retry attempts for the ChatGPT API call.
* `--parallel_num`: Number of concurrent API calls to ChatGPT.
* `--multi_turn_dialogue`: Whether to use multi-turn dialogue for inference.

Example usage:

```sh
python code/llm.py data/teleplay_dialogue_segment.json data/teleplay_episode_summary.json train data/teleplay_script.json
```

```sh
python code/llm.py data/teleplay_dialogue_segment.json data/teleplay_episode_summary.json train data/teleplay_script.json --max_retry_num 5 --retry_delay 0.5 --parallel_num 32 --multi_turn_dialogue True
```

The files `test_set.json` and `train_set.json` in the
link https://drive.google.com/drive/folders/1dBVtVzx-HJuRwrBKXdFWzSAbk1_sasxQ?usp=sharing provide the generated scripts
and related content for this dataset.