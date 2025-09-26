import threading
import re
import requests
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
import base64


parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.0000001)
parser.add_argument('--vqa-dir', type=str, default='data/COVERAGE_TEST_VQA')
parser.add_argument('--captions-file', type=str, default='sample_captions/llava1.6-vicuna_llama3_th1.0/captions_final.json')
args = parser.parse_args()

args.output_dir =  '/'.join(args.captions_file.split('/')[:-1])


_round_robin_index = 0
_round_robin_lock = threading.Lock()

def clean_json_str(s: str):
    s = re.sub(r'^```json|^```|```$', '', s, flags=re.MULTILINE).strip()
    s = re.sub(r'//.*', '', s)
    return json.loads(s)

def request_api_fast_vlm(input_prompt):
    global _round_robin_index

    ip = ''  
    port, model_name = '8000', "Qwen2.5-VL-72B-Instruct"


    url = f'http://{ip}:{port}/v1/chat/completions'

    payload = json.dumps({
        "model": model_name,
        "messages": input_prompt,
    })
    headers = {
        'Content-Type': 'application/json',
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=200)
        response.raise_for_status()
        resp_json = response.json()
        return {
            "status": True,
            "data": resp_json['choices'][0]['message']['content'],
            'model': model_name,
        }
    except Exception as e:
        return {
            "status": False,
            "data": 'request error ' + str(e),
            'model': model_name,
        }
    
def create_prompt_vlm(image_path, attribute):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
    attribute_dict = {key: "No" for key in attribute}
    attribute_dict[attribute[0]] = 'Yes'
    attribute_dict[attribute[-1]] = 'Yes'
    message = [
        {"role": "system", "content": f""" 
            Given an image, return a dictionary where each key is an attribute name and each value is "Yes" if the person in the image clearly possesses that attribute and it is visible, otherwise "No".

            Attribute set:
            {attribute}

            Example Output:
            {attribute_dict}

            All attributes should appear in the return dictionary. Output must be in valid JSON format. """
        },
        {   
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url":base64_qwen}},
                {"type": "text", "text": "Now output the dictionary for this image."},
                ],
        }
    ]
    return message

def read_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"error: cannot find '{file_path}'")
        return []

def process_and_write_line(idx, image_path, prompt, lock: threading.Lock, output_file: str):
    try:
        try_count = 3
        while try_count > 0:
            try:
                answer = request_api_fast_vlm(prompt)
                answer = clean_json_str(answer['data'])
                result_dict = {'image_path': image_path, 'answer': answer}
                break
            except Exception as e:
                result_dict = {'image_path': image_path, 'answer': {}}
                try_count -= 1

        json_line = json.dumps(result_dict, ensure_ascii=False)
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json_line + '\n')
                f.flush()
        print("write:", idx, image_path)
    except Exception as e:
        print(f"error: img_url:{image_path} : {e}")

if __name__ == "__main__":
    root = 'images/full_imagenet/imagenet_val'
    image_paths = []
    for i, sub_dir in enumerate(os.listdir(root)):
            for j, path in enumerate(os.listdir(os.path.join(root, sub_dir))):
                path = os.path.join(root, sub_dir, path)
                image_paths.append(path)

    attributes = []
    for key, value in json.load(open("images/full_imagenet/imagenet.json")).items():
        attributes += value
    print(len(attributes))
    data_per_part = len(attributes) // 10
    for part in range(6, 11):
        sub_attributes = attributes[part*data_per_part:(part+1)*data_per_part]
        file_write_lock = threading.Lock()
        output_file = 'images/full_imagenet/concept_part_%d_.jsonl'%part
        with ThreadPoolExecutor(max_workers=64) as executor:
            for img_idx, image_path in enumerate(image_paths):
                prompt = create_prompt_vlm(image_path, sub_attributes)
                executor.submit(process_and_write_line, img_idx, image_path, prompt, file_write_lock, output_file)
