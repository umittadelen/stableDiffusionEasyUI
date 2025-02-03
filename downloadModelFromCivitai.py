import json, requests, os

gconfig = {
    "defaultModelPath": "./models"
}

def askForAPIKey():
    key = input("Enter API key: ").strip()
    with open("civitai-api.key", "w") as f:
        f.write(key)
    return key

def getModeldata(model_id, version_id):
    url = f"https://civitai.com/api/v1/models/{model_id}?token={gconfig["CAI_TOKEN"]}"

    response = requests.get(url)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        model_data = response.json()
        modelVersion = next((m for m in model_data["modelVersions"] if m["id"] == version_id), model_data["modelVersions"][0])
        modelVersion["base_id"] = model_id
        return modelVersion
    else:
        return False

def shortenModelData(modelData):
    try:
        tmp = {
            modelData.get("files", [{}])[0].get("name", "").split(".")[0].replace("_", " ") or "unknown": {
                "path": f"./{modelData.get('files', [{}])[0].get('name', '')}" or None,
                "cfg": 7,
                "disabled": False,
                "type": modelData.get("baseModel", None),
                "link": modelData.get("files", [{}])[0].get("downloadUrl", None),
            },
            "name": modelData["files"][0]["name"].split(".")[0].replace("_", " ") if "files" in modelData else None,
            "files": {
                "path": f"./{modelData["files"][0]["name"].split(".")[0]}/{modelData["files"][0]["name"]}" if "files" in modelData else None,
                "name": modelData["files"][0]["name"] if "files" in modelData else None,
                "downloadUrl": modelData["files"][0]["downloadUrl"] if "files" in modelData else None

            },
            "images": {
                "url": modelData["images"][0]["url"] if "images" in modelData else None,
                "width": modelData["images"][0]["width"] if "images" in modelData else None,
                "height": modelData["images"][0]["height"] if "images" in modelData else None
            },
            "base_id": int(modelData["base_id"]) if "base_id" in modelData else None
        }
        print(tmp)
    except Exception as e:
        print(f"Error: {e}")
    return tmp

def checkForModelFiles(modelData, folderPath=gconfig["defaultModelPath"]):
    print(f"Model: {folderPath}/{modelData['files']['name'].replace(".safetensors","")}/{modelData['files']['name']}")
    if os.path.exists(f"{folderPath}/{modelData['files']['name'].replace(".safetensors","")}"):
        return True
    else:
        return False

def downloadModel(modelData, token, folderPath=gconfig["defaultModelPath"]):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    if not os.path.exists(f"{folderPath}/{modelData['files']['name'].replace(".safetensors","")}"):
        if not os.path.exists(f"{folderPath}/{modelData['files']['name'].replace(".safetensors","")}/{modelData['files']['name']}"):
            downloadWithTool(modelData["files"]["downloadUrl"].split('?')[0]+f"?token={token}", folderPath, f"/{modelData['files']['name'].replace('.safetensors', '')}/")
        with open(f"{folderPath}/{modelData["files"]["path"]}.json", "w") as file:
            json.dump(modelData, file, indent=4)
    else:
        return

def downloadWithTool(link, rootFolderPath=gconfig["defaultModelPath"], additionFolder=""):
    import shutil, subprocess
    def is_command_available(cmd):
            return shutil.which(cmd) is not None
    if is_command_available("aria2c"):
        print("Using aria2c for download.")
        subprocess.run(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {rootFolderPath + additionFolder.replace("./","")} {link}", shell=True)
    elif is_command_available("wget"):
        print("Using wget for download.")
        subprocess.run(f"wget -q -c -P {rootFolderPath + additionFolder.replace("./","")} '{link}'", shell=True)
    elif is_command_available("curl"):
        print("Using curl for download.")
        subprocess.run(f"curl -L -o \"{rootFolderPath + additionFolder.replace("./","")}\" \"{link}\"", shell=True)

def downloadModelFromCivitai(modelID, versionID):
    gconfig["downloading"] = True
    try:
        gconfig["CAI_TOKEN"] = (open(f'./civitai-api.key', 'r').read().strip() 
            if os.path.exists(f'./civitai-api.key')
            else askForAPIKey())

        model_data = getModeldata(modelID, versionID)
        print(model_data)
        print(model_data["files"][0]["downloadUrl"])
        print(f"{modelID}@{versionID}")
    
        # Ensure the model data was retrieved successfully
        if not model_data:
            print("Failed to retrieve model data.")
            return
        
        modelData = shortenModelData(model_data)
        
        # Check if model_data.json exists and initialize it as an empty list if not
        if not os.path.exists("./static/json/models.json"):
            with open("./static/json/models.json", "w") as file:
                file.write("[]")  # Initialize as an empty list
        
        with open("./static/json/models.json", "r+") as file:
            data = json.load(file)

            # Ensure data is a list
            if not isinstance(data, list):
                data = []  # Reinitialize as an empty list if not a list
            
            # Check if the model is already in the data
            model_exists = any(existing_model['name'] == modelData['name'] for existing_model in data)
            
            if model_exists:
                if checkForModelFiles(modelData):
                    print("Model already exists and files are present.")
                    return
                else:
                    print("Model already exists but files are missing. Downloading again.")
                    downloadModel(modelData, gconfig["CAI_TOKEN"])
                    return
            else:
                print("Model does not exist. Downloading...")
                downloadModel(modelData, gconfig["CAI_TOKEN"])
                data.append(modelData)  # Add the new model data to the list

            # Write the updated list to the file
            with open("./static/json/models.json", "w") as file:
                json.dump(data, file, indent=4)
    finally:
        gconfig["downloading"] = False
