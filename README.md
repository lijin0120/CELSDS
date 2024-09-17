## Step 1: Preparation Before Downloading

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

To remove the influence of video intro and outro sequences, and to improve the accuracy of OCR, different OCR settings need to be configured for each TV series. 
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
The first and fifth values correspond to modifications for the parameter `height`, the second value to `x1`, the third to `x2`, and the fourth to `y2`. 
For details on `x1`, `x2`, `y2`, and `height`, refer to the `moviepy` library documentation.
* `video_credits_fixed_limit`: The fixed duration of intro and outro sequences for each TV series. 
If the duration varies across episodes, this parameter can be ignored.
* `video_time_range`: The start and end time of each episode. 
If the intro and outro sequences are not of fixed duration, you need to annotate the start and end time excluding them. 
Otherwise, just mark the actual start and end time of the video, ignoring whether the intro and outro exist, as `video_credits_fixed_limit` already handles their fixed durations.

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

## Step 5: Filter and Merge

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

## Step ?: ChatGPT

To generate scripts using ChatGPT, run

```sh
python code/llm.py ${dialogue_segment} ${episode_summary} ${train_or_test} ${output_path}
```

Parameters:

* `dialogue_segment`: Path to the JSON file containing the dialogue segments.
* `episode_summary`: Path to the JSON file that contains summaries of each episode of the TV series.
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
