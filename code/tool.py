import json
import re

import cn2an
from zhconv import convert
from difflib import SequenceMatcher


def read_json(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_data = json.load(jf)
    return json_data


def write_json(json_data, json_file):
    with open(json_file, "w", encoding='utf-8') as jf:
        json.dump(json_data, jf, ensure_ascii=False, indent=4)


def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return int(h * 3600 + m * 60 + s)


def unify_symbols(text):
    text = text.replace("\\", "_").replace("/", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace(">", "_").replace("<", "_").replace("|", "_").strip()
    return text


def normalize_text(text):
    text = unify_symbols(text)
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    return normalized_text


def retain_chinese_and_english(text):
    return re.sub(r'[^\u4e00-\u9fffa-zA-Z，、。]', '', text)


def retain_chinese(text):
    return re.sub(r'[^\u4e00-\u9fff，、。]', '', text)


def convert_traditional_to_simplified(text):
    return convert(text, 'zh-hans')


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


def remove_punctuation(text):
    return re.sub(r'[，。！？、…·【】{}「」《》“”‘’：;,.!?\'\"\-\(\)\[\]\s]', '', text)


def get_episode(file_name):
    pattern_1 = re.compile(r'(?<!\d|S|格)(\d{1,2})(?!\d|FPS|版|K)')
    pattern_2 = re.compile(r'[零一二三四五六七八九十百千万]+')
    pattern_3 = re.compile(r'(?<!\d)\d{3}(?!\d)')
    if pattern_1.search(file_name):
        episode = pattern_1.search(file_name).group(1)
    elif pattern_2.search(file_name):
        episode = pattern_2.search(file_name).group()
        if not episode.isdigit():
            episode = cn2an.cn2an(episode, "smart")
    else:
        episode = pattern_3.search(file_name).group(0)

    return int(episode)
