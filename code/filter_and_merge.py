import os

from tqdm import tqdm

from tool import read_json, write_json, retain_chinese, convert_traditional_to_simplified, text_similarity, \
    retain_chinese_and_english, contains_chinese, normalize_text


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


def main(
        original_ocr_result_dir,
        output_path
):
    for teleplay in tqdm(os.listdir(original_ocr_result_dir)):
        if teleplay in ['Done', 'log']:
            continue
        original_teleplay_ocr_results_dir = os.path.join(original_ocr_result_dir, teleplay)
        for original_ocr_result_file in tqdm(os.listdir(original_teleplay_ocr_results_dir)):
            original_ocr_result_data = read_json(
                os.path.join(original_teleplay_ocr_results_dir, original_ocr_result_file)
            )
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

                for idx, text in enumerate(item['text']):
                    if text[1] < 0.8:
                        continue
                    if teleplay in ["何以笙箫默"]:
                        if "·" not in text[0]:
                            text_list.append(text[0])
                    else:
                        text_list.append(retain_chinese(text[0]))

                if not text_list:
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
                    output_path,
                    teleplay
                ), exist_ok=True
            )
            write_json(
                merged_result,
                os.path.join(
                    output_path,
                    teleplay,
                    original_ocr_result_file.split(".")[0] + "_merged.json"
                )
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("original_ocr_result_dir", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    main(**vars(args))
