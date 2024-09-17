## Preparation before downloading
We need to save the download link of the TV series in a json file, such as `data/teleplay_download_url.json`,
the json file content sample is shown below:
```
{
  "大明王朝1566": "https://www.youtube.com/playlist?list=PLUDHmjG40FQBGQ54JEtrDZ3SiL018hCB6",
  ...
}
```

## Download and process videos
In order to download the videos of each TV series and extract the corresponding audio, run
```sh
python code/download_and_process.py ${teleplay_download_url} ${output_path}
```
在从视频中提取音频时，我们需要使用ffmpeg工具，请确保ffmpeg工具能够正常使用。

参数解释：
* `teleplay_download_url`: 在上一个阶段准备的电视剧下载链接文件。
* `output_path`: 下载的视频和提取出的音频的保存目录。

使用样例：
```sh
python code/download_and_process.py ../data/teleplay_download_url.json ../data/video_and_audio
```

## OCR
为了获取语音对应的转录，使用PaddleOCR提取视频字幕，run
```sh
python code/ocr.py ${video_root_dir} ${ocr_configure} ${output_path}
```

参数解释：
* `video_root_dir`: 在上一个阶段下载的视频的保存目录。
* `ocr_configure`: 保存ocr要使用的相关配置项的json文件。
* `output_path`: ocr结果的保存目录。

使用样例：
```sh
python code/ocr.py ../data/video_and_audio ../data/teleplay_ocr_configure.json ../data/ocr/original
```