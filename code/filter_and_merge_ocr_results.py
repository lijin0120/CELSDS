import os

from tqdm import tqdm

from tool import read_json, write_json, retain_chinese, convert_traditional_to_simplified, text_similarity, \
    retain_chinese_and_english, contains_chinese, normalize_text


def merge_item(item_list):
    result = None
    max_similarity = 0.0
    start_time, end_time = None, None
    for front_item in item_list:
        if start_time is None:
            start_time, end_time = front_item["time"][0], front_item["time"][1]
        else:
            start_time = front_item["time"][0] if front_item["time"][0] < start_time else start_time
            end_time = front_item["time"][1] if front_item["time"][1] > end_time else end_time

        sum_similarity = []
        front_text = "".join(front_item["text"])
        for back_item in item_list:
            back_text = "".join(back_item["text"])
            cur_similarity = text_similarity(front_text, back_text)
            sum_similarity.append(cur_similarity)
        avg_similarity = sum(sum_similarity) / len(sum_similarity)
        if avg_similarity > max_similarity:
            result = front_item
            max_similarity = avg_similarity
    result["time"][0] = start_time
    result["time"][1] = end_time

    assert result
    return result


def merge_duplicates(result_list):
    assert result_list != []
    merged_list = []

    result_list.sort(key=lambda x: x['time'][0])

    last_item = None
    temp_list = []
    for idx, item in enumerate(result_list):
        if last_item is None:
            last_item = item
            if idx == len(result_list) - 1:
                merged_list.append(last_item)
            continue

        last_text = "".join(last_item["text"])
        cur_text = "".join(item["text"])
        if text_similarity(last_text, cur_text) >= 0.666666:
            temp_list.append(last_item)
            last_item = item
            if idx == len(result_list) - 1:
                temp_list.append(last_item)
                if len(temp_list) >= 3:
                    merged_item = max(temp_list, key=lambda x: x['num'])
                else:
                    merged_item = max(temp_list, key=lambda x: x['num'])
                merged_list.append(merged_item)
        else:
            if not temp_list:
                merged_list.append(last_item)
                last_item = item
                if idx == len(result_list) - 1:
                    merged_list.append(last_item)
            else:
                temp_list.append(last_item)
                # print(temp_list)
                if len(temp_list) >= 3:
                    merged_item = max(temp_list, key=lambda x: x['num'])
                else:
                    merged_item = max(temp_list, key=lambda x: x['num'])
                merged_list.append(merged_item)
                last_item = item
                if idx == len(result_list) - 1:
                    merged_list.append(last_item)
                temp_list = []

    final_list = []
    for item in merged_list:
        item.pop('num', None)
        if item["time"][0] == item["time"][1]:
            continue
        final_list.append(item)

    return final_list


def check(object):
    if len(object['text']) == 1:
        return False
    for text in object['text']:
        if len(text[0]) > 1:
            return False
    return True


def filter_and_merge_ocr_results(
        not_pass_one_strings,
        other_strings,
        teleplay_list,
        original_results_dir,
        filtered_and_merged_results_dir
):
    for teleplay in tqdm(os.listdir(original_results_dir)):
        if teleplay not in teleplay_list:
            continue
        original_teleplay_ocr_results_dir = os.path.join(original_results_dir, teleplay)
        for original_ocr_result_file in tqdm(os.listdir(original_teleplay_ocr_results_dir)):
            original_ocr_result_data = read_json(
                os.path.join(original_teleplay_ocr_results_dir, original_ocr_result_file))
            merged_data = []
            prev_text_key = None

            for item in original_ocr_result_data:
                time = item['time']
                text_list = []
                temp_list = []
                for text in item['text']:
                    text[0] = convert_traditional_to_simplified(retain_chinese_and_english(normalize_text(text[0])))
                    if contains_chinese(text[0]):
                        temp_list.append(text)
                item['text'] = temp_list

                if check(item) or len(item['text']) >= 4:
                    continue
                # if len(item['text']) == 1 and len(item['text'][0][0]) == 1 and item['text'][0][0] in not_pass_one_strings:
                #     text_list.append(item['text'][0][0])
                #     continue
                for idx, text in enumerate(item['text']):
                    if text[1] < 0.8:
                        continue
                    if teleplay in ["何以笙箫默"]:
                        if "·" not in text[0]:
                            text_list.append(text[0])
                    else:
                        text_list.append(retain_chinese(text[0]))

                text_key = [t for t in text_list if t and t not in other_strings]
                if not text_key:
                    continue
                text_key = tuple(text_list)

                if prev_text_key is not None and text_key == prev_text_key:
                    merged_data[-1]['end_time'] = time
                    merged_data[-1]['num'] += 1
                else:
                    merged_data.append({
                        'text': text_key,
                        'start_time': time,
                        'end_time': time,
                        'num': 1
                    })
                prev_text_key = text_key

            merged_result = []
            for item in merged_data:
                merged_result.append({
                    "time": [item['start_time'], item['end_time']],
                    "text": list(item['text']),
                    'num': item['num']
                })
            merged_result = merge_duplicates(merged_result)

            os.makedirs(
                os.path.join(
                    filtered_and_merged_results_dir,
                    teleplay
                ), exist_ok=True
            )
            write_json(
                merged_result,
                os.path.join(
                    filtered_and_merged_results_dir,
                    teleplay,
                    original_ocr_result_file.split(".")[0] + "_merged.json"
                )
            )
        #     break
        # break


def check_final_results(teleplay_list, filtered_and_merged_results_dir):
    for teleplay in os.listdir(filtered_and_merged_results_dir):
        if teleplay not in teleplay_list:
            continue
        teleplay_results_dir = os.path.join(filtered_and_merged_results_dir, teleplay)
        for file in os.listdir(teleplay_results_dir):
            teleplay_episode_file = os.path.join(teleplay_results_dir, file)
            teleplay_episode_data = read_json(teleplay_episode_file)

            previous_end_time = None
            for i, item in enumerate(teleplay_episode_data):
                current_start_time = item['time'][0]
                text = item["text"]
                # if len(text) > 2:
                #     print(f"{teleplay}\t{teleplay_episode_file}\t{text}")

                if previous_end_time is not None and current_start_time <= previous_end_time:
                    print(f"不符合的情况:\n对象1: {teleplay_episode_data[i - 1]}\n对象2: {item}\n")

                previous_end_time = item['time'][1]


if __name__ == "__main__":
    not_pass_one_strings = [
        # '爸', '嗯', '我', '对', '烦', '雪', '要', '走', '热', '谁', '您', '来', '没', '哎', '呀', '这', '耶', '嘘',
        # '你', '喂', '嗨', '爹', '给', '奶', '它', '跳', '妞', '啊', '滚', '坐', '但', '她', '说', '好', '哦', '呸',
        # '乖', '他', '血', '朕', '快', '错', '是', '不', '跑', '慢', '鬼', '吃', '痛', '呐', '狗', '听', '兄', '姐',
        # '妈', '咱', '请', '喝'
    ]
    other_strings = [
        # '镶我們盘盘烈烈', '邦', '把握青春年华', '剧剧审字第号', '国和业', '络', '安民泰网',
        # '团和业', '安氏泰网', '真页', '量景', '镶我们鼻毒烈烈', '鲸', '回和', '皇文化传播有限公司',
        # '安民泰國', '承镶承', '青岛海皇文化传播有限公司', '安氏泰國', '国和座', '歌终', '秋终', '承承'
    ]
    teleplay_list = [
        "书剑恩仇录",
        "何以笙箫默",
        "侠客行",
        "大明王朝1566",
        "天涯·明月·刀2012",
        "宫锁心玉",
        "庆余年",
        "新边城浪子",
        "楚留香新传",
        "沉香如屑",
        "流星·蝴蝶·剑",
        "还珠格格"
    ]
    # s = set(other_strings)
    # print(list(s))
    ocr_results_root_dir = "/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/train_set/ocr"
    original_ocr_results_dir = os.path.join(ocr_results_root_dir, "original_5fps")
    filtered_and_merged_ocr_results_dir = os.path.join(ocr_results_root_dir, "filtered_and_merged_5fps")

    filter_and_merge_ocr_results(
        not_pass_one_strings,
        other_strings,
        teleplay_list,
        original_ocr_results_dir,
        filtered_and_merged_ocr_results_dir
    )

    check_final_results(teleplay_list, filtered_and_merged_ocr_results_dir)
