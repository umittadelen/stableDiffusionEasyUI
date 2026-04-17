import json, requests, os
from urllib.parse import urljoin, urlparse

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

    try:
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch model data: {e}")
        return False

    model_data = response.json()
    modelVersion = next((m for m in model_data["modelVersions"] if m["id"] == version_id), model_data["modelVersions"][0])
    modelVersion["model_id"] = model_id
    modelVersion["model_name"] = model_data.get("name", "")
    return modelVersion

def shortenModelData(modelData):
    tmp = {}
    try:
        model_filename = next(
            (v.get("name", "") for v in modelData.get("files", [])
            if str(modelData.get("id", "")) in v.get("downloadUrl", "")
            and v.get("type", "Model") == "Model"
            and v.get("metadata", {}).get("format", "") == "SafeTensor"),
            ""
        )

        if not model_filename and modelData.get("files"):
            model_filename = modelData.get("files", [{}])[0].get("name", "")

        first_image = (modelData.get("images") or [{}])[0]
        display_name = (
            modelData.get("model_name")
            or modelData.get("display_name")
            or modelData.get("_display_name")
            or modelData.get("name", "")
            or model_filename.split('.')[0]
        )

        tmp = {
            "type": modelData.get("baseModel", "SDXL"),
            "disabled": False,
            "cfg": 7,
            "name": model_filename,
            "display_name": display_name,
            "path": f"{gconfig['defaultModelPath']}{model_filename.split('.')[0]}/{model_filename}" if model_filename else "",
            "images": {
                "url": first_image.get("url", ""),
                "width": first_image.get("width", ""),
                "height": first_image.get("height", "")
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
    model_name = modelData.get('name', '')
    if not model_name:
        return False

    model_folder = model_name.split('.')[0]
    model_path = os.path.join(folderPath, model_folder, model_name)
    print(f"Model: {model_path}")
    return os.path.isfile(model_path)

def downloadModel(modelData, token, folderPath=gconfig["defaultModelPath"]):
    os.makedirs(folderPath, exist_ok=True)

    model_name = modelData['name']
    model_folder = model_name.split('.')[0]
    model_dir = os.path.join(folderPath, model_folder)
    model_path = os.path.join(model_dir, model_name)
    model_json_path = f"{model_path}.json"

    os.makedirs(model_dir, exist_ok=True)

    if not os.path.isfile(model_path):
        print("Starting to download...")
        downloadWithTool(modelData["download_url"].split('?')[0], folderPath, f"/{model_folder}/", token, model_name)

    with open(model_json_path, "w", encoding="utf-8") as file:
        json.dump(modelData, file, indent=4)

def _get_auth_headers(url, token=""):
    host = (urlparse(url).netloc or "").lower()
    if token and ("civitai.com" in host or "civitai.red" in host):
        return {"Authorization": f"Bearer {token}"}
    return {}


def resolve_download_url(link, token="", max_redirects=5):
    """Resolve civitai redirects to a fully qualified downloadable URL.
    Some responses return scheme-relative locations like //civitai.com/..., which
    must be normalized before passing them to aria2c/wget/curl."""
    session = requests.Session()
    current_url = link

    for _ in range(max_redirects):
        response = session.get(
            current_url,
            headers=_get_auth_headers(current_url, token),
            allow_redirects=False,
            stream=True,
            timeout=30,
        )

        if response.status_code not in (301, 302, 303, 307, 308):
            return current_url

        next_url = response.headers.get("Location", current_url)
        current_url = urljoin(current_url, next_url)
        print(f"Resolved to: {current_url}")

    return current_url


def downloadWithTool(link, rootFolderPath=gconfig["defaultModelPath"], additionFolder="", token="", filename=""):
    link = resolve_download_url(link, token)
    print(f"Downloading {link}")
    import shutil, subprocess

    def is_command_available(cmd):
        return shutil.which(cmd) is not None

    dest = rootFolderPath + additionFolder.replace('./', '')
    os.makedirs(dest, exist_ok=True)
    filename = filename or os.path.basename(urlparse(link).path) or "downloaded_model.safetensors"
    output_path = os.path.join(dest, filename)

    if is_command_available("aria2c"):
        print("Using aria2c for download.")
        cmd = [
            "aria2c", "--console-log-level=error", "--summary-interval=10",
            "-c", "-x", "16", "-k", "1M", "-s", "16", "-d", dest, "-o", filename, link
        ]
    elif is_command_available("wget"):
        print("Using wget for download.")
        cmd = ["wget", "-q", "-c", "-O", output_path, link]
    elif is_command_available("curl"):
        print("Using curl for download.")
        cmd = ["curl", "-L", "-C", "-", "-o", output_path, link]
    else:
        print("No external downloader found, falling back to requests.")
        with requests.get(link, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        return output_path

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed with exit code {result.returncode}")

    return output_path

def downloadModelFromCivitai(modelID, versionID):
    gconfig["downloading"] = True
    try:
        gconfig["CAI_TOKEN"] = (open(f'./civitai-api.key', 'r').read().strip() 
            if os.path.exists(f'./civitai-api.key')
            else askForAPIKey())

        model_data = getModeldata(modelID, versionID)
        print(f"{modelID}@{versionID}")

        if __name__ != "__main__":
            # Ensure the model data was retrieved successfully
            if not model_data:
                print("Failed to retrieve model data.")
                return
            
            modelData = shortenModelData(model_data)
            
            # Check if model_data.json exists and initialize it as an empty list if not
            if not os.path.exists("./static/json/models.json"):
                with open("./static/json/models.json", "w") as file:
                    file.write("[]")  # Initialize as an empty list

            with open("./static/json/models.json", "r+", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []

                if not isinstance(data, list):
                    data = []

                existing_index = next(
                    (i for i, existing_model in enumerate(data) if existing_model.get('name') == modelData['name']),
                    -1
                )

                if existing_index != -1:
                    if checkForModelFiles(modelData):
                        print("Model already exists and files are present. Refreshing metadata.")
                    else:
                        print("Model already exists but files are missing. Downloading again.")

                    downloadModel(modelData, gconfig["CAI_TOKEN"])
                    data[existing_index] = modelData
                else:
                    print("Model does not exist. Downloading...")
                    downloadModel(modelData, gconfig["CAI_TOKEN"])
                    data.append(modelData)

                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()
    finally:
        gconfig["downloading"] = False
        gconfig["status"] = "Finished Downloading"

if __name__ == "__main__":
    downloadModelFromCivitai("1025125", "1506467")