## 步骤 1: 下载前的准备

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

## Step 3: OCR前的准备

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

## Step 4: OCR

为了获取语音对应的转录，使用PaddleOCR提取视频字幕，运行

```sh
python code/ocr.py ${video_root_dir} ${ocr_configure} ${output_path}
```

参数：

* `video_root_dir`: 在 step 2 下载的视频的保存目录。
* `ocr_configure`: 在 step 3 保存的ocr要使用的相关配置项的json文件。
* `output_path`: ocr结果的保存目录。

使用样例：

```sh
python code/ocr.py data/video_and_audio data/teleplay_ocr_configure.json data/ocr/original
```

## Step 5: 筛选及合并

为了将重复的ocr结果合并起来，同时将ocr效果较差的结果筛掉，运行

```sh
python code/filter_and_merge.py ${original_ocr_result_dir} ${output_path}
```

参数：

* `original_ocr_result_dir`: 在 step 4 获得的ocr结果的保存目录。
* `output_path`: 将ocr结果进行筛选及合并后的保存目录。

使用样例：

```sh
python code/filter_and_merge_ocr_results.py data/ocr/original/ data/ocr/filter_and_merged
```

## Step ?: ChatGPT

使用ChatGPT生成剧本，运行

```sh
python code/llm.py ${dialogue_segment} ${episode_summary} ${train_or_test} ${output_path}
```

参数：

* `dialogue_segment`: 保存切分出的对话片段的json文件路径。
* `episode_summary`: 保存每部电视剧每一集剧情概述的json文件路径。
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
