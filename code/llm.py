# -*- coding: utf-8 -*-
import json
import os
import re
import time
from multiprocessing import Pool

import openai
from tqdm import tqdm

from tool import read_json, normalize_text, text_similarity, remove_punctuation

llm_name = "gpt-3.5-turbo"
llm_configure = {
    "openai_api_key": "openai_api_key",
    "openai_base_url": 'openai_base_url',
}

return_status_dict = {
    "对话内容不一致": "你刚才的结果不符合我的要求：不要对内容进行删减、合并、扩写或者总结，你仅可以纠正对白中存在的错别字。请重新理解我的要求并给出符合要求的结果。",
    "对话数量不一致": "你刚才的结果不符合我的要求：不要合并对白！不要合并对白！不要合并对白！不能缺失对白！不能缺失对白！不能缺失对白！请重新理解我的要求并给出符合要求的结果。",
    "描述内容不足": "你刚才的结果不符合我的要求：在每个人物话语的上一行需要穿插对环境变化的描述，对人物的神态、动作、内心活动、说话语气的描述，描述尽可能地详细。你在刚才的回答中，描述的内容太少，请重新理解我的要求并给出符合要求的结果。"
}


class ChatGPT:
    def __init__(self, llm_name, openai_api_key, openai_base_url):
        self.llm_name = llm_name
        openai.api_key = openai_api_key
        openai.base_url = openai_base_url

    def inference(self, txt):
        response = openai.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": txt}
            ]
        )
        return response.choices[0].message.content

    def multi_turn(self, question1, answer1, question2):
        response = openai.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": question1},
                {"role": "assistant", "content": answer1},
                {"role": "user", "content": question2}
            ],
        )
        return response.choices[0].message.content


def check_llm_result(result, plot, train_or_test):
    pattern = r"[\u4e00-\u9fa5]+(?:\d)?(?:·[\u4e00-\u9fa5]+)?(?:（[\u4e00-\u9fa5]+(?:·[\u4e00-\u9fa5]+)?）)?：?[^”]+”"
    dialogues = re.findall(pattern, plot)
    dialogues_content = ""
    for dialogue in dialogues:
        dialogues_content = remove_punctuation(dialogues_content + dialogue.split("“")[1].split("”")[0])

    result = result.replace(" ", "").replace(".", "。").replace(",", "，").replace("(", "（").replace(")", "）").replace(
        ":", "：").replace("?", "？").replace("!", "！")

    new_result = ""
    desc_num = 0
    speech_content_num = 0
    speech_contents = []
    speech_content = ""
    lines = result.strip().split("\n")
    for idx, line in enumerate(lines):
        if line != "":
            if line.startswith("（"):
                desc_num = desc_num + 1
                new_result = new_result + line + "\n"
            else:
                if idx != 0:
                    speech_contents.append(
                        remove_punctuation(
                            line.replace("\"", "“", 1).replace("\"", "”", 1)
                        )
                    )
                    speech_content_num = speech_content_num + 1
                    if "：“" not in line:
                        continue
                    speech_content = speech_content + \
                                     line.replace("\"", "“", 1).replace("\"", "”", 1).split("“")[1].split("”")[0]
                new_result = new_result + line.replace("\"", "“", 1).replace("\"", "”", 1) + "\n"

    if train_or_test == "test":
        if len(dialogues) != len(speech_contents):
            print("对话数量不一致")
            print(len(dialogues))
            print(len(speech_contents))
            return new_result, False, None, "对话数量不一致"

        for dialogue, speech_content in zip(dialogues, speech_contents):
            dialogue = remove_punctuation(dialogue)
            if dialogue != speech_content:
                print("对话内容不一致")
                print(dialogue)
                print(speech_content)
                return new_result, False, None, "对话内容不一致"

        if desc_num < len(speech_contents):
            print("描述内容不足")
            return new_result, False, None, "描述内容不足"
    elif train_or_test == "train":
        sim = text_similarity(dialogues_content, remove_punctuation(speech_content))
        if sim < 0.95:
            print("对话内容不一致")
            return new_result, False, sim, "对话内容不一致"

        if len(dialogues) != len(speech_contents):
            print("对话数量不一致")
            print(len(dialogues))
            print(len(speech_contents))
            return new_result, False, sim, "对话数量不一致"

        if desc_num < speech_content_num / 3:
            print("描述内容不足")
            return new_result, False, sim, "描述内容不足"

    print("Pass")
    return new_result, True, None, "Pass"


def get_llm_prompts(train_or_test):
    llm_prompts = {
        "train": [
            """你现在是一名编剧，你可以根据故事的背景，角色说话的对白编写出剧本。所有角色的对白都必须出现在剧本中，不能缺失。如果一个角色连续说话，禁止你把对白内容进行合并。如果对白中存在错别字，你可以进行纠正，但仅限于纠错，不能对对白进行扩写、精简或总结。你会首先定位出输入的对话属于背景中的哪一段描述，再结合背景中不同角色之间的关系写出符合说话内容的剧本。在人物话语的上一行可以穿插对环境变化的描述，对人物的神态、动作、内心活动、说话语气的描述，描述尽可能地详细。
#以下是样例的对话内容：
海瑞：“卑职在。”赵贞吉：“为什么要写那道奏疏？”海瑞：“上疏是为臣的天职。”赵贞吉：“你在奏疏里都写了些什么？”海瑞：“赵大人来审问卑职，皇上却没将卑职的奏疏给赵大人看过。”
#以下是输出格式的样例：
（海瑞的面容憔悴，他站得笔直，尽管身体虚弱，但他的眼神依然坚毅，声音中带着一丝平静的自信。）
海瑞：“卑职在。”
（赵贞吉的语气更加严厉，显得十分严肃。）
赵贞吉：“为什么要写那道奏疏？”
（海瑞微微低头，声音平稳而不失坚定。）
海瑞：“上疏是为臣的天职。”
（赵贞吉的神色一变，显得有些激动，他的语气中夹杂着不满和质疑。）
赵贞吉：“你在奏疏里都写了些什么？”
（海瑞的面容依旧平静，他抬起头，与赵贞吉对视，语气中带着一丝冷漠。）
海瑞：“赵大人来审问卑职。”
海瑞：“皇上却没将卑职的奏疏给赵大人看过。”
####### 好了，请你结合以上的要求和样例输出格式完成你的任务：
#以下是背景：
{}
#以下是角色台词：
{}""",
        ],
        "test": [
            """你现在是一名编剧，你可以根据故事的背景，角色说话的内容编写出剧本。所有角色的说话内容都必须出现在剧本中，不能缺失，同时角色的说话内容必须保持不变，不能修改。你会首先定位出输入的对话属于背景中的哪一段描述，再结合背景中不同角色之间的关系写出符合说话内容的剧本。在每个人物话语的上一行需要穿插对环境变化的描述，对人物的神态、动作、内心活动、说话语气的描述，描述尽可能地详细。
#以下是样例的对话内容：
海瑞：“卑职在。”赵贞吉：“为什么要写那道奏疏？”海瑞：“上疏是为臣的天职。”赵贞吉：“你在奏疏里都写了些什么？”海瑞：“赵大人来审问卑职，皇上却没将卑职的奏疏给赵大人看过。”
#以下是输出格式的样例：
（海瑞的面容憔悴，他站得笔直，尽管身体虚弱，但他的眼神依然坚毅，声音中带着一丝平静的自信。）
海瑞：“卑职在。”
（赵贞吉的语气更加严厉，显得十分严肃。）
赵贞吉：“为什么要写那道奏疏？”
（海瑞微微低头，声音平稳而不失坚定。）
海瑞：“上疏是为臣的天职。”
（赵贞吉的神色一变，显得有些激动，他的语气中夹杂着不满和质疑。）
赵贞吉：“你在奏疏里都写了些什么？”
（海瑞的面容依旧平静，他抬起头，与赵贞吉对视，语气中带着一丝冷漠。）
海瑞：“赵大人来审问卑职，皇上却没将卑职的奏疏给赵大人看过。”
####### 好了，请你结合以上的要求和样例输出格式完成你的任务：
#以下是背景：
{}
#以下是角色台词：
{}""",
        ]
    }
    return llm_prompts[train_or_test]


def process_llm_data(args):
    llm_object, llm_input_datas, llm_max_retry_num, llm_retry_delay, train_or_test, multi_turn_dialogue = args
    llm_results = {}

    for item in tqdm(llm_input_datas):
        teleplay = item["teleplay"]
        episode = item["episode"]
        prompts = item["prompts"]
        summary = item["summary"]
        dialogue_segment = item["dialogue_segment"]
        dialogue_segment_idx = item["dialogue_segment_index"]
        wavs_path = item["wavs_path"]

        if teleplay not in llm_results:
            llm_results[teleplay] = {}
        if episode not in llm_results[teleplay]:
            llm_results[teleplay][episode] = {
                "summary": item["summary"],
                "dialogue_segments": []
            }

        for prompt_idx, prompt in enumerate(prompts):
            llm_input_data = normalize_text(prompt.format(summary, dialogue_segment))
            all_result = []
            llm_result = None
            status = None
            for ai, attempt in enumerate(range(llm_max_retry_num)):
                try:
                    if multi_turn_dialogue:
                        if llm_result in [None, ""]:
                            llm_result = normalize_text(llm_object.inference(llm_input_data))
                        else:
                            llm_result = normalize_text(llm_object.multi_turn(
                                question1=llm_input_data,
                                answer1=llm_result,
                                question2=return_status_dict[status]
                            ))
                    else:
                        llm_result = normalize_text(llm_object.inference(llm_input_data))
                    llm_result, check_pass, sim, status = check_llm_result(llm_result, dialogue_segment, train_or_test)
                    all_result.append((llm_result, sim))
                    if not check_pass:
                        continue
                    llm_results[teleplay][episode]["dialogue_segments"].append({
                        "dialogue_segment_index": dialogue_segment_idx,
                        "prompt_index": prompt_idx,
                        "dialogue_segment": dialogue_segment,
                        "script": llm_result,
                        "acceptable": True,
                        "wavs_path": wavs_path
                    })
                    break
                except Exception as e:
                    print(f"{teleplay} {episode} dialogue_segment_{dialogue_segment_idx} prompt_{prompt_idx} try {attempt + 1} fail, error msg: {e}")
                    time.sleep(llm_retry_delay)
            else:
                print(f"{teleplay} {episode} dialogue_segment_{dialogue_segment_idx} prompt_{prompt_idx} reached the maximum number of retries, still unsuccessful.")
                llm_results[teleplay][episode]["dialogue_segments"].append({
                    "dialogue_segment_index": dialogue_segment_idx,
                    "prompt_index": prompt_idx,
                    "dialogue_segment": dialogue_segment,
                    "script": all_result,
                    "acceptable": False,
                    "wavs_path": wavs_path
                })

    return llm_results


def get_llm_result(
        llm_objects,
        llm_max_retry_num,
        llm_retry_delay,
        dialogue_segment,
        episode_summary,
        llm_prompts,
        train_or_test,
        multi_turn_dialogue,
        parallel_num,
        output_path
):
    assert llm_objects is not None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    llm_input_datas = []
    llm_results = {}

    for teleplay in dialogue_segment:
        for episode in dialogue_segment[teleplay]:
            summary = max(
                [summary for summary in episode_summary[teleplay][episode].values()],
                key=len
            )
            for item in dialogue_segment[teleplay][episode]:
                item["teleplay"] = teleplay
                item["episode"] = episode
                item["summary"] = summary
                item["prompts"] = llm_prompts
                llm_input_datas.append(item)

    print(len(llm_input_datas))
    llm_data_chunks = [[] for _ in llm_objects]
    for idx, data in enumerate(llm_input_datas):
        llm_data_chunks[idx % len(llm_objects)].append(data)

    args_list = [
        (llm_objects[idx], data_chunk, llm_max_retry_num, llm_retry_delay, train_or_test, multi_turn_dialogue)
        for idx, data_chunk in enumerate(llm_data_chunks)
    ]

    with Pool(processes=parallel_num) as pool:
        results = pool.map(process_llm_data, args_list)

    for result in results:
        for teleplay, episodes in result.items():
            if teleplay not in llm_results:
                llm_results[teleplay] = {}
            for episode, data in episodes.items():
                if episode not in llm_results[teleplay]:
                    llm_results[teleplay][episode] = data
                else:
                    llm_results[teleplay][episode]["dialogue_segments"].extend(data["dialogue_segments"])

    with open(output_path, "w", encoding='utf-8') as of:
        json.dump(llm_results, of, ensure_ascii=False, indent=4)


def create_llm_object(name, configure, parallel_num):
    openai_api_key = configure["openai_api_key"]
    openai_base_url = configure["openai_base_url"]
    assert openai_api_key and openai_base_url and type(openai_api_key) == type(openai_base_url)
    if isinstance(openai_api_key, list) and isinstance(openai_base_url, list):
        llm_object = []
        for oak, obu in zip(openai_api_key, openai_base_url):
            llm_object.extend([ChatGPT(name, oak, obu) for _ in range(parallel_num)])
    else:
        llm_object = [ChatGPT(name, openai_api_key, openai_base_url) for _ in range(parallel_num)]

    return llm_object


def main(
        dialogue_segment,
        episode_summary,
        train_or_test,
        output_path,
        max_retry_num,
        retry_delay,
        parallel_num,
        multi_turn_dialogue
):
    llm = create_llm_object(llm_name, llm_configure, parallel_num)
    prompts = get_llm_prompts(train_or_test)
    dialogue_segment_data = read_json(dialogue_segment)
    episode_summary_data = read_json(episode_summary)

    get_llm_result(
        llm,
        max_retry_num,
        retry_delay,
        dialogue_segment_data,
        episode_summary_data,
        prompts,
        train_or_test,
        multi_turn_dialogue,
        parallel_num,
        output_path
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dialogue_segment", type=str)
    parser.add_argument("episode_summary", type=str)
    parser.add_argument("train_or_test", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--max_retry_num", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=0.5)
    parser.add_argument("--parallel_num", type=int, default=32)
    parser.add_argument("--multi_turn_dialogue", type=bool, default=True)
    args = parser.parse_args()

    main(**vars(args))
