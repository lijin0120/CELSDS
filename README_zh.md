## 步骤 1: 下载视频前的准备

我们需要将电视剧的下载链接保存到一个json文件中，例如`data/teleplay_download_url.json`，json文件内容样例如下：

```
{
  "大明王朝1566": "https://www.youtube.com/playlist?list=PLUDHmjG40FQBGQ54JEtrDZ3SiL018hCB6",
  ...
}
```

## 步骤 2: 下载并处理视频

为了下载电视剧视频并从中提取出相应音频，运行

```sh
python code/download_and_process.py ${teleplay_download_url} ${output_path}
```

在从视频中提取音频时，我们需要使用ffmpeg工具，请确保ffmpeg工具能够正常使用。

参数：

* `teleplay_download_url`: 在上一个阶段准备的电视剧下载链接文件。
* `output_path`: 下载的视频和提取出的音频的保存目录。

使用样例：

```sh
python code/download_and_process.py data/teleplay_download_url.json data/video_and_audio
```

## 步骤 3: OCR前的准备

为了去除视频片头片尾的影响，并且为了提高ocr的准确率，需要根据不同电视剧设置不同的ocr参数，这些参数需要保存在一个json文件中，例如
`data/teleplay_ocr_configure.json`，json文件内容样例如下：

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

参数：

* `video_fps`: 重新设置的视频每秒的帧率。
* `video_cut`: 在裁剪视频每帧图片时，为了更加贴合字幕边界设置的参数。第一个和第五个参数对应对参数`height`的修改，
  第二个参数对应对参数`x1`的修改，第三个参数对应对参数`x2`的修改，第四个参数对应对参数`y2`的修改。
  参数`x1`，`x2`，`y2`和`height`的解释请参考`moviepy`库的使用文档。
* `video_credits_fixed_limit`: 每部电视剧固定的片头片尾时长。如果每集片头片尾时长不固定，则忽略此项。
* `video_time_range`: 每部电视剧每集的开始和结束时间。如果电视剧没有固定的片头片尾时长，则需要标注无片头片尾的视频开始和结束时间。
  否则只需要标注视频的实际开始和结束时间，无需理会是否存在片头片尾，因为参数`video_credits_fixed_limit`已经标注固定的片头片尾时长。

## 步骤 4: OCR

为了获取语音对应的转录，使用PaddleOCR提取视频字幕，运行

```sh
python code/ocr.py ${video_root_dir} ${ocr_configure} ${output_path}
```

参数：

* `video_root_dir`: 在 步骤 2 下载的视频的保存目录。
* `ocr_configure`: 在 步骤 3 保存的ocr要使用的相关配置项的json文件。
* `output_path`: ocr结果的保存目录。

使用样例：

```sh
python code/ocr.py data/video_and_audio data/teleplay_ocr_configure.json data/ocr/original
```

## 步骤 5: 筛选及合并OCR结果

为了将重复的ocr结果合并起来，同时将ocr效果较差的结果筛掉，运行

```sh
python code/filter_and_merge.py ${original_ocr_result_dir} ${output_path}
```

参数：

* `original_ocr_result_dir`: 在 步骤 4 获得的ocr结果的保存目录。
* `output_path`: 将ocr结果进行筛选及合并后的保存目录。

使用样例：

```sh
python code/filter_and_merge.py data/ocr/original/ data/ocr/filter_and_merged
```

## 步骤 6: 提取参考人物语音的说话人嵌入前的准备

在提取参考人物语音的说话人嵌入前，需要将人物的参考语音的路径写入一个json文件中，例如`data/teleplay_speaker_wav.json`
，json文件内容样例如下：

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

链接`https://drive.google.com/drive/folders/1dBVtVzx-HJuRwrBKXdFWzSAbk1_sasxQ?usp=sharing`
中的文件`reference_speaker_wav.zip`中已给出本数据集使用到的所有参考人物语音及说话人嵌入。

## 步骤 7: 提取参考人物语音的说话人嵌入

为了标注切分出的语音的说话人标签，需要提前提取出参考人物语音的说话人嵌入。

```sh
python code/extract_speaker_embedding.py ${speaker_wav_path} ${available_gpus} ${max_processes_per_gpu} ${speaker_model_path} ${wav_save_dir} ${wav_speaker_embedding_save_dir} ${wav_16k_save_dir}
```

参数：

* `speaker_wav_path`: 在 步骤 6 标注的保存了参考人物语音路径的json文件。
* `available_gpus`: 提取说话人嵌入时可使用的GPU id。列表类型。
* `max_processes_per_gpu`: 提取说话人嵌入时每个GPU进行并行处理时的最大任务数量。
* `speaker_model_path`: 提取说话人嵌入使用的模型的路径。
* `wav_save_dir`: 参考人物语音的保存目录。
* `wav_speaker_embedding_save_dir`: 参考人物的语音提取出的说话人嵌入的保存目录。
* `wav_16k_save_dir`: 参考人物的16k采样率语音的保存目录。使用ResNet293模型提取说话人嵌入时需将语音转换为16k采样率。

使用样例：

```sh
python code/extract_speaker_embedding.py data/teleplay_speaker_wav.json [0,1] 2 /path/to/wespeaker/voxceleb_resnet293_LM data/reference_speaker_wav data/reference_speaker_wav_speaker_embedding data/reference_speaker_wav_16k
```

## 步骤 8: 分割语音、标注弱说话人标签以及分割出对话片段

将ocr结果筛选及合并后，根据时间戳分割出对应转录的语音。提取出语音说话人嵌入后标注语音片段对应的弱说话人标签。将每集的完整对话分割为多个对话片段。运行

```sh
python code/slice_ocr_wav.py ${audio_root_dir} ${filtered_and_merged_ocr_results_dir} ${segment_wav_save_dir} ${segment_wav_json_path} ${available_gpus} ${max_processes_per_gpu} ${speaker_model_path} ${segment_wav_speaker_embedding_save_dir} ${segment_wav_16k_save_dir} ${reference_speaker_embedding_save_dir} ${dialogue_segment_save_path}
```

参数：

* `audio_root_dir`: 在 步骤 2 提取出的音频的保存目录。
* `filtered_and_merged_ocr_results_dir`: 在 步骤 5 将ocr结果进行筛选及合并后的保存目录。
* `segment_wav_save_dir`: 分割出的语音的保存目录。
* `segment_wav_json_path`: 保存分割出的语音相关信息的json文件路径。
* `available_gpus`: 提取说话人嵌入时可使用的GPU id。列表类型。
* `max_processes_per_gpu`: 提取说话人嵌入时每个GPU进行并行处理时的最大任务数量。
* `speaker_model_path`: 提取说话人嵌入使用的模型的路径。
* `segment_wav_speaker_embedding_save_dir`: 分割出的语音提取出的说话人嵌入的保存目录。
* `segment_wav_16k_save_dir`: 16k采样率语音的保存目录。使用ResNet293模型提取说话人嵌入时需将语音转换为16k采样率。
* `reference_speaker_embedding_save_dir`: 在 步骤 7 提取出的参考人物的说话人嵌入保存目录。
* `dialogue_segment_save_path`: 保存分割出的对话片段的json文件路径。

可选参数：

* `--time_bias`: 语音结束时间戳的前移偏移量。语音结束时间戳通过字幕时间戳计算得出，而字幕相比对应的语音通常会有一定延迟，可通过该参数补偿延迟。单位毫秒。
* `--divide_num`: 每集的完整对话分割出的对话片段的数量。

使用样例：

```sh
python code/slice_ocr_wav.py data/video_and_audio data/ocr/filter_and_merged data/segment_wav data/teleplay_segment_wav.json [0,1] 2 /path/to/wespeaker/voxceleb_resnet293_LM data/segment_wav_speaker_embedding data/segment_wav_16k data/reference_speaker data/teleplay_dialogue_segment.json
```

```sh
python code/slice_ocr_wav.py data/video_and_audio data/ocr/filter_and_merged data/segment_wav data/teleplay_segment_wav.json [0,1] 2 /path/to/wespeaker/voxceleb_resnet293_LM data/segment_wav_speaker_embedding data/segment_wav_16k data/reference_speaker data/teleplay_dialogue_segment.json --time_bias 400 --divide_num 15
```

链接`https://drive.google.com/drive/folders/1dBVtVzx-HJuRwrBKXdFWzSAbk1_sasxQ?usp=sharing`
中的文件`test_segment_wav.json`和`train_segment_wav.json`中已给出本数据集分割出的语音片段及其相关信息。

## 步骤 9: 生成剧本前的准备

在生成剧本之前，需要收集每部电视剧每一集的剧情概述至一个json文件中，例如：`data/teleplay_episode_summary.json`
，json文件内容样例如下：

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

## 步骤 10: 生成剧本

在使用ChatGPT生成剧本前，可以修改`code/llm.py`文件中有关ChatGPT接口调用的相关参数，如下：

```
llm_name = "gpt-3.5-turbo"
llm_configure = {
    "openai_api_key": "openai_api_key",
    "openai_base_url": 'openai_base_url',
}
```

使用ChatGPT生成剧本，运行

```sh
python code/llm.py ${dialogue_segment} ${episode_summary} ${train_or_test} ${output_path}
```

参数：

* `dialogue_segment`: 在 步骤 8 保存的切分出的对话片段的json文件路径。
* `episode_summary`: 在 步骤 9 保存的每部电视剧每一集剧情概述的json文件路径。
* `train_or_test`: 由于构建训练集和测试集时的prompt不一致，可以选择任意一种使用。
* `output_path`: 保存最终结果的json文件路径。

可选参数：

* `--max_retry_num`: 调用ChatGPT接口最大尝试次数。
* `--retry_delay`: 重新尝试调用ChatGPT接口的间隔时间。
* `--parallel_num`: 并行调用ChatGPT接口的并行数量。
* `--multi_turn_dialogue`: 是否使用多轮对话进行推理。

使用样例：

```sh
python code/llm.py data/teleplay_dialogue_segment.json data/teleplay_episode_summary.json train data/teleplay_script.json
```

```sh
python code/llm.py data/teleplay_dialogue_segment.json data/teleplay_episode_summary.json train data/teleplay_script.json --max_retry_num 5 --retry_delay 0.5 --parallel_num 32 --multi_turn_dialogue True
```

链接`https://drive.google.com/drive/folders/1dBVtVzx-HJuRwrBKXdFWzSAbk1_sasxQ?usp=sharing`
中的文件`test_set.json`和`train_set.json`中已给出本数据集生成的剧本及相关内容。