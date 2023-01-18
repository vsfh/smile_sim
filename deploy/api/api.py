import requests
import json
import base64
import urllib
import time
from PIL import Image
from io import BytesIO

with open("./token_devt.json", "r") as f:
    token = json.loads(f.read())
    user_group = token["user_group"]
    user_id = token["user_id"]
    zh_token = token["zh_token"]

def smile(img_path):
    # 修改文件路径
    with open(img_path, "rb") as f:
        data = f.read()

    resp = requests.put(f"https://file-server.dev.chohotech.com/scratch/{user_group}/{user_id}/upload?" +
                        "postfix=jpg", # 必须指定 postfix, 即文件后缀名
                        data, headers={"X-ZH-TOKEN": zh_token})
    resp.raise_for_status()
    path = "/".join(urllib.parse.urlparse(resp.url).path.lstrip("/").split("/")[3:])
    urn = f"urn:zhfile:o:s:{user_group}:{user_id}:{path}"
    print("文件指针:", urn)
    url = "https://zhmle-workflow-api.dev.chohotech.com/workflow/run" # API 地址改变

    payload = json.dumps({
    "spec_group": "smile",
    "spec_name": "smile-sim-lip-preserve",
    "spec_version": "1.0-snapshot",
    "user_group": user_group,
    "user_id": user_id,
    "input_data": {
        "image": urn,
    },
    "metadata":{"workflow_scheduler":True},
    })
    headers = {
    "Content-Type": "application/json",
    "X-ZH-TOKEN": zh_token
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
    create_result = response.json()
    run_id = create_result['run_id']
    print(run_id)
    url = f"https://zhmle-workflow-api.dev.chohotech.com/workflow/run/{run_id}"
    # response = requests.request("GET", url, headers=headers)
    # response.json()


    start_time = time.time()
    while time.time()-start_time < 300: # 最多等5分钟
        time.sleep(3) # 轮询间隔
        response = requests.request("GET", url, headers=headers)
        result = response.json()
        if result['completed'] or result['failed']:
            break
    print(time.time()-start_time)

    url = f"https://zhmle-workflow-api.dev.chohotech.com/workflow/data/{run_id}"
    response = requests.request("GET", url, headers=headers)

    result = response.json()
    # print(result['result']['image'])
    resp = requests.get("https://file-server.dev.chohotech.com" + f"/file/download?" + urllib.parse.urlencode({
                            "urn": result['result']['image']}),
                            headers={"X-ZH-TOKEN": zh_token})
    mesh_bytes = resp.content

    a = img_path.split('/')[-1].split('.')[0]
    image = Image.open(BytesIO(mesh_bytes)).convert("RGB")
    image.save(f'./result/{a}.jpg')
    
import os
path = '/home/disk/data/test/smile/11.8.2022'
path = '/mnt/share/shenfeihong/data/test_03_26'
path = '/home/meta/sfh/data/smile/40photo'
for file in os.listdir(path):
    if not os.path.isfile(os.path.join('./result', file.split('.')[0]+'.jpg')):
        print(file)
        smile(os.path.join(path, file))