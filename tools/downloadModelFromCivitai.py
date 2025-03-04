import json, requests, os

gconfig = {
    "defaultModelPath": "./models/"
}

def askForAPIKey():
    key = input("Enter API key: ").strip()
    with open("civitai-api.key", "w") as f:
        f.write(key)
    return key

def getModeldata(model_id, version_id):
    url = f"https://civitai.com/api/v1/models/{model_id}?token={gconfig['CAI_TOKEN']}"

    response = requests.get(url)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        model_data = response.json()
        modelVersion = next((m for m in model_data["modelVersions"] if m["id"] == version_id), model_data["modelVersions"][0])
        modelVersion["model_id"] = model_id
        return modelVersion
    else:
        return False

def shortenModelData(modelData):
    try:
        open("output.json", "w").write(json.dumps(modelData, indent=4))
        model_filename = next(
            (v.get("name", "") for v in modelData.get("files", [])
            if str(modelData.get("id", "")) in v.get("downloadUrl", "")
            and v.get("type", "Model") == "Model"
            and v.get("metadata", {}).get("format", "") == "SafeTensor"), 
            ""
        )

        tmp = {
            "type": modelData.get("baseModel", "SDXL"),
            "disabled": False,
            "cfg": 7,
            "name": model_filename,
            "path": f"{gconfig["defaultModelPath"]}{model_filename.split('.')[0]}/{model_filename}" if "files" in modelData else "",
            "images": {
                "url": modelData["images"][0]["url"] if "images" in modelData else "",
                "width": modelData["images"][0]["width"] if "images" in modelData else "",
                "height": modelData["images"][0]["height"] if "images" in modelData else ""
            },
            "trained_words": modelData.get("trainedWords", [""])[0] if modelData.get("trainedWords") else "",
            "nsfw": modelData.get("model", {}).get("nsfw", False),
            "model_id": modelData.get("model_id", ""),
            "model_version_id": modelData.get("id", ""),
            "download_url": modelData.get("downloadUrl", "")
        }
        print(tmp)
    except Exception as e:
        print(f"Error: {e}")
    return tmp

def checkForModelFiles(modelData, folderPath=gconfig["defaultModelPath"]):
    model_name = modelData['files']['name']
    model_folder = model_name.split('.')[0]
    print(f"Model: {folderPath}/{model_folder}/{model_name}")
    if os.path.exists(f"{folderPath}/{model_folder}"):
        return True
    else:
        return False

def downloadModel(modelData, token, folderPath=gconfig["defaultModelPath"]):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    model_name = modelData['name']
    model_folder = model_name.split('.')[0]
    if not os.path.exists(f"{folderPath}/{model_folder}"):
        if not os.path.exists(f"{folderPath}/{model_folder}/{model_name}"):
            print("Starting to download...")
            downloadWithTool(modelData["download_url"].split('?')[0] + f"?token={token}", folderPath, f"/{model_folder}/")
        with open(f"{folderPath}/{model_folder}/{model_name}.json", "w") as file:
            json.dump(modelData, file, indent=4)
    else:
        if not os.path.isfile(f"./{folderPath}/{model_folder}/{model_name}.json"):
            print("creaing config file")
            with open(f"{folderPath}/{model_folder}/{model_name}.json", "w") as file:
                json.dump(modelData, file, indent=4)
        return

def downloadWithTool(link, rootFolderPath=gconfig["defaultModelPath"], additionFolder=""):
    print(f"Downloading {link}")
    import shutil, subprocess
    def is_command_available(cmd):
            return shutil.which(cmd) is not None
    if is_command_available("aria2c"):
        print("Using aria2c for download.")
        subprocess.run(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {rootFolderPath + additionFolder.replace('./','')} {link}", shell=True)
    elif is_command_available("wget"):
        print("Using wget for download.")
        subprocess.run(f"wget -q -c -P {rootFolderPath + additionFolder.replace('./','')} '{link}'", shell=True)
    elif is_command_available("curl"):
        print("Using curl for download.")
        subprocess.run(f"curl -L -o \"{rootFolderPath + additionFolder.replace('./','')}\" \"{link}\"", shell=True)

def downloadModelFromCivitai(modelID, versionID):
    gconfig["downloading"] = True
    try:
        gconfig["CAI_TOKEN"] = (open(f'./civitai-api.key', 'r').read().strip() 
            if os.path.exists(f'./civitai-api.key')
            else askForAPIKey())

        model_data = getModeldata(modelID, versionID)
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
    finally:
        gconfig["downloading"] = False

if __name__ == "__main__":
    downloadModelFromCivitai("34469", "480978")