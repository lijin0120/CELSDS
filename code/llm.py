# -*- coding: utf-8 -*-
import json
import os
import re
import time
from multiprocessing import Pool

import httpx
import openai
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
import qianfan

from tool import read_json, normalize_text, text_similarity, remove_punctuation

openai_llm = [
    "gpt-3.5-turbo",
    "claude-3-haiku-20240307",
]
doubao_llm = [
    "ep-20240629095146-29dnq",
    "ep-20240825121924-tn7rs"
]
qianfan_llm = [
    "completions_pro",
    "ERNIE-3.5-8K"
]

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
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": txt}
            ]
        )
        print(response)
        return response.choices[0].message.content


class Doubao:
    def __init__(self, llm_name, ark_api_key, ark_base_url):
        self.llm_name = llm_name
        self.ark_base_url = ark_base_url
        self.proxies = {
            'http://': None,  # 代理1
            'https://': None,  # 代理2
        }
        os.environ["ARK_API_KEY"] = ark_api_key

    def inference(self, txt):
        client = Ark(
            base_url=self.ark_base_url,
            http_client=httpx.Client(proxies=self.proxies),
        )
        completion = client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": txt},
            ],
        )
        return completion.choices[0].message.content

    def multi_turn(self, question1, answer1, question2):
        client = Ark(
            base_url=self.ark_base_url,
            http_client=httpx.Client(proxies=self.proxies),
        )
        completion = client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": question1},
                {"role": "assistant", "content": answer1},
                {"role": "user", "content": question2}
            ],
        )
        return completion.choices[0].message.content


class Ernie:
    def __init__(self, llm_name, qianfan_ak, qianfan_sk):
        self.llm_name = llm_name
        self.qianfan_ak = qianfan_ak
        self.qianfan_sk = qianfan_sk
        os.environ["QIANFAN_AK"] = qianfan_ak
        os.environ["QIANFAN_SK"] = qianfan_sk

    def inference(self, txt):
        result = qianfan.ChatCompletion().do(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": txt}
            ],
            temperature=0.95,
            top_p=0.8,
            penalty_score=1,
            disable_search=False,
            enable_citation=False
        )["result"]
        return result

    def multi_turn(self, question1, answer1, question2):
        result = qianfan.ChatCompletion().do(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": question1},
                {"role": "assistant", "content": answer1},
                {"role": "user", "content": question2}
            ],
            temperature=0.95,
            top_p=0.8,
            penalty_score=1,
            disable_search=False,
            enable_citation=False
        )["result"]
        return result


def remove_chinese_punctuation(text):
    chinese_punctuation = "，。！？；：“”‘’（）【】《》、"
    # 创建一个翻译表，将中文标点符号映射为空字符
    trans_table = str.maketrans('', '', chinese_punctuation)
    # 使用 translate 方法进行替换
    return text.translate(trans_table)


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
            # print(dialogues_content)
            # print(remove_punctuation(speech_content))
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
    print(train_or_test)
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


def process_llm_data(
        llm_object,
        llm_input_datas,
        llm_max_retry_num,
        llm_retry_delay,
        train_or_test,
        multi_turn_dialogue,
        output_file
):
    llm_results = {}

    for item in tqdm(llm_input_datas):
        teleplay = item["teleplay"]
        episode = item["episode"]
        prompts = item["prompts"]
        abstract = item["abstract"]
        # train
        plot = item["text"]
        plot_idx = item["plot_index"]
        wavs_path = item["wavs_path"]
        # test
        # plots = item["plots"]

        if teleplay not in llm_results:
            llm_results[teleplay] = {}
        if episode not in llm_results[teleplay]:
            llm_results[teleplay][episode] = {
                "abstract": item["abstract"],
                "plots": []
            }

        # train
        for prompt_idx, prompt in enumerate(prompts):
            llm_input_data = normalize_text(prompt.format(abstract, plot))
            all_result = []
            llm_result = None
            status = None
            for ai, attempt in enumerate(range(llm_max_retry_num)):
                try:
                    if multi_turn_dialogue:
                        if llm_result in [None, ""]:
                            llm_result = normalize_text(llm_object.inference(llm_input_data))
                        else:
                            # print(return_status_dict[status])
                            llm_result = normalize_text(llm_object.multi_turn(
                                question1=llm_input_data,
                                answer1=llm_result,
                                question2=return_status_dict[status]
                            ))
                    else:
                        llm_result = normalize_text(llm_object.inference(llm_input_data))
                    llm_result, check_pass, sim, status = check_llm_result(llm_result, plot, train_or_test)
                    # print(llm_result)
                    all_result.append((llm_result, sim))
                    if not check_pass:
                        continue
                    llm_results[teleplay][episode]["plots"].append({
                        "plot_index": plot_idx,
                        "prompt_index": prompt_idx,
                        "text": plot,
                        "scenario": llm_result,
                        "acceptable": True,
                        "wavs_path": wavs_path
                    })
                    break
                except Exception as e:
                    print(
                        f"{teleplay} {episode} plot_{plot_idx} prompt_{prompt_idx} 尝试 {attempt + 1} 失败，错误信息：{e}")
                    time.sleep(llm_retry_delay)
            else:
                print(f"{teleplay} {episode} plot_{plot_idx} prompt_{prompt_idx} 达到最大重试次数，仍未成功。")
                llm_results[teleplay][episode]["plots"].append({
                    "plot_index": plot_idx,
                    "prompt_index": prompt_idx,
                    "text": plot,
                    "scenario": all_result,
                    "acceptable": False,
                    "wavs_path": wavs_path
                })
            # with open(output_file, "w", encoding='utf-8') as of:
            #     json.dump(llm_results, of, ensure_ascii=False, indent=4)
        # break

        # test
        # for plot_idx, plot in enumerate(plots):
        #     for prompt_idx, prompt in enumerate(prompts):
        #         llm_input_data = normalize_text(prompt.format(abstract, plot))
        #         all_result = []
        #         for attempt in range(llm_max_retry_num):
        #             try:
        #                 llm_result = normalize_text(llm_object.inference(llm_input_data))
        #                 llm_result, check_pass, sim = check_llm_result(llm_result, plot, train_or_test)
        #                 # print(llm_result)
        #                 all_result.append((llm_result, sim))
        #                 if not check_pass:
        #                     continue
        #                 llm_results[teleplay][episode]["plots"].append({
        #                     "plot_index": plot_idx,
        #                     "prompt_index": prompt_idx,
        #                     "text": plot,
        #                     "scenario": llm_result,
        #                     "acceptable": True
        #                 })
        #                 break
        #             except Exception as e:
        #                 print(
        #                     f"{teleplay} {episode} plot_{plot_idx} prompt_{prompt_idx} 尝试 {attempt + 1} 失败，错误信息：{e}")
        #                 time.sleep(llm_retry_delay)
        #         else:
        #             print(f"{teleplay} {episode} plot_{plot_idx} prompt_{prompt_idx} 达到最大重试次数，仍未成功。")
        #             llm_results[teleplay][episode]["plots"].append({
        #                 "plot_index": plot_idx,
        #                 "prompt_index": prompt_idx,
        #                 "text": plot,
        #                 "scenario": all_result,
        #                 "acceptable": False
        #             })
        #         with open(output_file, "w", encoding='utf-8') as of:
        #             json.dump(llm_results, of, ensure_ascii=False, indent=4)
        #     break
        # break

    return llm_results


def worker_function(args):
    idx, data_chunk, llm_objects, llm_max_retry_num, llm_retry_delay, train_or_test, multi_turn_dialogue, llm_result_json_file = args
    output_file = f"{os.path.splitext(llm_result_json_file)[0]}_{idx}.json"
    llm_object = llm_objects[idx]
    return process_llm_data(
        llm_object,
        data_chunk,
        llm_max_retry_num,
        llm_retry_delay,
        train_or_test,
        multi_turn_dialogue,
        output_file
    )


def check(results, item):
    teleplay = item["teleplay"]
    episode = item["episode"]
    if teleplay in results.keys() and episode in results[teleplay].keys():
        for plot in results[teleplay][episode]["plots"]:
            if plot["plot_index"] == item["plot_index"] and plot["scenario"] != []:
                return True
    return False


def get_llm_result(
        llm_objects,
        llm_max_retry_num,
        llm_retry_delay,
        teleplay_episode_plots,
        teleplay_episode_abstract,
        llm_prompts,
        train_or_test,
        multi_turn_dialogue,
        llm_result_json_file
):
    assert llm_objects is not None
    if isinstance(llm_objects, list):
        print(len(llm_objects))
    os.makedirs(os.path.dirname(llm_result_json_file), exist_ok=True)

    llm_input_datas = []
    llm_results = {}
    # part_result = "/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/test_set/llm_results/not_check.json"
    # part_results = read_json(part_result)

    for teleplay in teleplay_episode_plots:
        for episode in teleplay_episode_plots[teleplay]:
            abstract = max(
                [abstract for abstract in teleplay_episode_abstract[teleplay][episode].values()],
                key=len
            )
            # test
            # llm_input_datas.append(
            #     {
            #         "teleplay": teleplay,
            #         "episode": episode,
            #         "plots": teleplay_episode_plots[teleplay][episode],
            #         "abstract": abstract,
            #         "prompts": llm_prompts
            #     }
            # )
            # train
            for item in teleplay_episode_plots[teleplay][episode]:
                item["teleplay"] = teleplay
                item["episode"] = episode
                item["abstract"] = abstract
                item["prompts"] = llm_prompts
                # if check(part_results, item):
                #     continue
                llm_input_datas.append(item)

    print(len(llm_input_datas))
    llm_data_chunks = [[] for _ in llm_objects]
    for idx, data in enumerate(llm_input_datas):
        llm_data_chunks[idx % len(llm_objects)].append(data)

    args_list = [
        (idx, data_chunk, llm_objects, llm_max_retry_num, llm_retry_delay, train_or_test, multi_turn_dialogue,
         llm_result_json_file)
        for idx, data_chunk in enumerate(llm_data_chunks)
    ]

    with Pool(processes=min(len(llm_data_chunks), 64)) as pool:
        results = pool.map(worker_function, args_list)

    for result in results:
        for teleplay, episodes in result.items():
            if teleplay not in llm_results:
                llm_results[teleplay] = {}
            for episode, data in episodes.items():
                if episode not in llm_results[teleplay]:
                    llm_results[teleplay][episode] = data
                else:
                    llm_results[teleplay][episode]["plots"].extend(data["plots"])

    with open(llm_result_json_file, "w", encoding='utf-8') as of:
        json.dump(llm_results, of, ensure_ascii=False, indent=4)


def continue_to_generate(
        llm_object,
        llm_max_retry_num,
        llm_retry_delay,
        llm_prompts,
        llm_result_json_file
):
    result_data = read_json(llm_result_json_file)
    flag = False
    for teleplay in tqdm(result_data.keys()):
        for episode in result_data[teleplay].keys():
            abstract = result_data[teleplay][episode]["abstract"]
            for plot in tqdm(result_data[teleplay][episode]["plots"]):
                if plot["scenario"] == "结果不达标":
                    plot_idx = plot["plot_index"]
                    prompt_idx = plot["prompt_index"]
                    prompt = llm_prompts[prompt_idx]
                    plot_text = plot["text"]
                    llm_input_data = normalize_text(prompt.format(abstract, plot_text))
                    for attempt in range(llm_max_retry_num):
                        try:
                            llm_result = normalize_text(llm_object.inference(llm_input_data))
                            llm_result, check_pass = check_llm_result(llm_result, plot_text)
                            print(llm_result)
                            if not check_pass:
                                continue
                            plot["scenario"] = llm_result
                            break
                        except Exception as e:
                            print(
                                f"{teleplay} {episode} plot_{plot_idx} prompt_{prompt_idx} 尝试 {attempt + 1} 失败，错误信息：{e}")
                            time.sleep(llm_retry_delay)
                    else:
                        print(f"{teleplay} {episode} plot_{plot_idx} prompt_{prompt_idx} 达到最大重试次数，仍未成功。")
                        plot["scenario"] = "结果不达标"
                    with open(llm_result_json_file, "w", encoding='utf-8') as of:
                        json.dump(result_data, of, ensure_ascii=False, indent=4)
                    flag = True
                    break
        #     break
        if flag:
            break


def create_llm_object(name, configure, parallel_num):
    llm_object = None
    if name in openai_llm:
        openai_api_key = configure["openai_api_key"]
        openai_base_url = configure["openai_base_url"]
        assert openai_api_key and openai_base_url and type(openai_api_key) == type(openai_base_url)
        if isinstance(openai_api_key, list) and isinstance(openai_base_url, list):
            llm_object = []
            for oak, obu in zip(openai_api_key, openai_base_url):
                llm_object.extend([ChatGPT(name, oak, obu) for _ in range(parallel_num)])
        else:
            llm_object = [ChatGPT(name, openai_api_key, openai_base_url)]

    return llm_object


if __name__ == "__main__":
    llm_name = "gpt-3.5-turbo"
    llm_configure = {
        "openai_api_key": "openai_api_key",
        "openai_base_url": 'openai_base_url',
    }

    max_retry_num = 1
    retry_delay = 0.5
    parallel_num = 64
    multi_turn_dialogue = True

    llm = create_llm_object(llm_name, llm_configure, parallel_num)
    train_or_test = "train"
    prompts = get_llm_prompts(train_or_test)

    teleplay_episode_plots_file = "/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/test_set/test_asr_plots.json"
    teleplay_episode_abstract_file = "/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/teleplay_episode_summary.json"
    teleplay_episode_plots_data = read_json(teleplay_episode_plots_file)
    teleplay_episode_abstract_data = read_json(teleplay_episode_abstract_file)

    result_json_file = f"/Work21/2023/lijin/workspace/Code/SUSC/data/json/susc/test_set/llm_results/{llm_name}_results.json"

    get_llm_result(
        llm,
        max_retry_num,
        retry_delay,
        teleplay_episode_plots_data,
        teleplay_episode_abstract_data,
        prompts,
        train_or_test,
        multi_turn_dialogue,
        result_json_file
    )

    # continue_to_generate(
    #     llm,
    #     max_retry_num,
    #     retry_delay,
    #     prompts,
    #     result_json_file
    # )
