const existingImages = new Map(); // Store existing images with their seeds as keys
let isGeneratingNewImages = false;
let pendingUpdates = false;
let isCleared = false;
const customConfirm = new CustomConfirm();

function loadFormData() {
    fetch('/load_form_data')
    .then(response => response.json())
    .then(data => {
        const form = document.getElementById('generateForm');
        for (const [key, value] of Object.entries(data)) {
            const field = form.elements[key];

            if (field && ['TEXTAREA', 'SELECT', 'INPUT'].includes(field.tagName)) {
                if (field.type === 'file') {
                    continue;
                }
                if (field.tagName === 'TEXTAREA') {
                    field.textContent = value;
                } else if (field.tagName === 'SELECT') {
                    Array.from(field.options).forEach(option => {
                        option.selected = option.value === String(value);
                    });
                } else {
                    field.value = value;
                }
            }
        }
    })
    .catch(error => console.error('Error loading form data:', error));
}

function populateModels(data, select) {
    data.forEach(item => {
        const option = document.createElement('option');
        option.value = item.path;
        option.dataset.cfg = item.cfg || 7;
        option.dataset.type = item.type || "SDXL";
        option.dataset.src = item.images.url;
        option.textContent = item.name.split(".")[0];

        if (item.disabled) {
            option.disabled = true;
        }

        select.appendChild(option);
    });
}

function populateExamplePrompts(data, select) {
    data.examples.forEach(prompt => {
        const option = document.createElement('option');
        option.value = prompt;
        option.textContent = prompt;
        select.appendChild(option);
    });
}

function populateExampleSizes(data, select) {
    Object.entries(data).forEach(([sizeName, sizeDimensions]) => {
        const option = document.createElement('option');
        option.value = sizeDimensions.join('x'); // Format as "width x height"
        option.textContent = sizeName; // Display the size name
        select.appendChild(option);
    });
}

function populateSchedulers(data, select) {
    data.schedulers.forEach(scheduler => {
        const option = document.createElement('option');
        option.value = scheduler;
        option.textContent = scheduler;
        select.appendChild(option);
    });
}

fetch("/scan_model_configs")
.then(response => response.json())
.then(data => {
    loadJsonAndPopulateSelect(data, 'model', populateModels);
})
.catch(error => {
    loadJsonAndPopulateSelect('/static/json/models.json', 'model', populateModels);
    console.error("Error:", error);
})

loadJsonAndPopulateSelect('/static/json/examplePrompts.json', 'example_prompt', populateExamplePrompts);
loadJsonAndPopulateSelect('/static/json/dimensions.json', 'example_size', populateExampleSizes);
loadJsonAndPopulateSelect('/static/json/schedulers.json', 'scheduler', populateSchedulers);

loadFormData();

function submitButtonOnClick(event) {
    event.preventDefault();

    // Clear existing images for new generation
    isGeneratingNewImages = true;
    existingImages.clear();
    document.getElementById('images').innerHTML = '';
    const formElement = document.getElementById('generateForm');
    saveFormData();

    // Prepare data for the server
    const formData = new FormData(formElement); // Use the form element here
    fetch('/generate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        isGeneratingNewImages = false;
        console.log('Server response:', data);
    })
    .catch(error => {
        console.error('Error:', error);
        isGeneratingNewImages = false;
    });
}

function saveFormData() {
    const formDataToSave = {};
    const formElement = document.getElementById("generateForm");

    Array.from(formElement.elements).forEach(field => {
        if (field.name && ['TEXTAREA', 'SELECT', 'INPUT'].includes(field.tagName)) {
            let value = field.value;

            // Convert to number if it's a valid number
            if (!isNaN(value) && value.trim() !== "") {
                value = Number(value);
            }

            formDataToSave[field.name] = value;
        }
    });

    fetch('/save_form_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formDataToSave)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Form data saved:', data);
    })
    .catch(error => {
        console.error('Error saving form data:', error);
    });
}

async function resetFormButtonOnClick(event) {
    const isConfirmed = await customConfirm.createConfirm('Are you sure you want to reset the form?<br>this cannot be undone!',
        [
            { text: 'Reset', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

    if (isConfirmed) {
        fetch('/reset_form_data')
        .then(response => response.json())
        .then(data => {
            console.log('Form cache has been reset:', data);
            location.reload();
        })
        .catch(error => {
            console.error('Error resetting form cache:', error);
        });
    }
};

setInterval(() => {
    if (document.visibilityState === 'visible' && !pendingUpdates && !isCleared) {
        pendingUpdates = true;
        console.log('Fetching status...');

        fetch('/status', { cache: 'no-store' })
            .then((response) => response.json())
            .then((data) => {
                document.getElementById('all').style.display = 'flex';

                updateProgressBars(data);

                processImageUpdates(data.images, data.images_reverse || false);

                updateImageScales();

                if (data.images.length < existingImages.size) {
                    existingImages.clear();
                    document.getElementById('images').innerHTML = '';
                }
            })
            .catch(() => {
                updateProgressBars({}, "Error fetching status");
                document.getElementById('all').style.display = 'none';
            })
            .finally(() => {
                pendingUpdates = false;
                isCleared = false;
            });
    }
}, 2500);

document.addEventListener('contextmenu', function (event) {
    // block <div id="status" oncontextmenu="toggleStatus()"></div> an dchilds
    if (event.target.closest('#status')) {
        event.preventDefault();
        console.log('Context menu prevented');
        return;
    }

    if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT' || event.target.tagName === 'TEXTAREA') {
        console.log('Context menu allowed for input/textarea');
        return;
    }
    else {
        event.preventDefault();
        console.log('Context menu prevented');

        console.log('Creating custom confirm');
        customConfirm.createConfirm('Quick Actions', [
            { text: 'Clear Images', value: () => clearButtonOnClick(event) },
            { text: 'Stop Generation', value: () => stopButtonOnClick(event) },
            { text: 'Get Metadata', value: () => window.open(`metadata`) }
        ], true);
    }
});

function updateProgressBars(data, error = "") {
    const progressText = document.getElementById('progress');
    const dynamicProgressBar = document.getElementById('dynamic-progress-bar');
    const alldynamicProgressBar = document.getElementById('all-dynamic-progress-bar');

    const setProgress = (element, value) => element.style.width = `calc(${value}%)`;
    const resetProgress = () => {
        dynamicProgressBar.style.width = '0%';
        alldynamicProgressBar.style.width = '0%';
    };

    if (error) {
        progressText.innerHTML = error;
        resetProgress();
        return;
    }

    if (Number.isInteger(data.imgprogress)) {
        setProgress(dynamicProgressBar, data.imgprogress);
        progressText.innerHTML = `Progress: ${data.imgprogress}% Remaining: ${data.remainingimages}`;
    }
    else if (data.imgprogress.endsWith("Generation Stopped\n")) {
        resetProgress();
        progressText.innerHTML = 'Generation Stopped';
    }
    else if (typeof data.imgprogress === 'string') {
        resetProgress();
        progressText.innerHTML = data.imgprogress.slice(-200).replace(/\n/g, '<br>');
    }
    else if (data.imgprogress.trim() === "") {
        resetProgress();
        progressText.innerHTML = 'Status: idle';
    }

    if (Number.isInteger(data.allpercentage)) {
        setProgress(alldynamicProgressBar, data.allpercentage);
    } else {
        setProgress(alldynamicProgressBar, 0);
    }
}

async function getTokenCount(inElementID, outElementId) {
    const textarea = document.getElementById(inElementID);
    const text = textarea.value;
    const response = await fetch('/clip_token_count', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ text })
    });
    const result = await response.json();
    document.getElementById(outElementId).innerHTML = `${result.CLIPTokenCount}/${result.MaxTokens}`;
}

function processImageUpdates(images, reverse) {
    const imagesDiv = document.getElementById('images');

    if (reverse) {
        images.reverse()
        var editedImagesList = images;
    } else {
        var editedImagesList = images;
    }

    editedImagesList.forEach((imgData, index) => {
        const key = imgData.seed;

        if (existingImages.has(key)) {
            const existingImg = existingImages.get(key).querySelector('img');
            if (existingImg.src !== imgData.img) {
                existingImg.src = imgData.img+"?r=1";
            }
        } else {
            const wrapper = document.createElement('div');
            wrapper.className = 'image-wrapper';

            const img = document.createElement('img');
            img.src = imgData.img+"?r=1";
            img.loading = "lazy";
            img.onclick = () => openLink("image/" + imgData.img.split('/').pop());

            wrapper.appendChild(img);

            if (reverse && index === 0) {
                imagesDiv.insertBefore(wrapper, imagesDiv.firstChild);
            } else {
                imagesDiv.appendChild(wrapper);
            }
            existingImages.set(key, wrapper);
        }

        if ((reverse && index === 0) || (!reverse && index === images.length - 1)) {
            const imgToUpdate = existingImages.get(key)?.querySelector('img');
            if (imgToUpdate) {
                imgToUpdate.src = `${imgData.img}?timestamp=${Date.now()}&r=1`; // Force refresh
            }
        }
    });
}

function toggleStatus(event) {
    event.preventDefault();
    const statusBox = document.getElementById('status');
    statusBox.classList.toggle('minimized');
}

dragElement(document.getElementById("status"));

function dragElement(elmnt) {
  var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  if (document.getElementById(elmnt.id + "header")) {
    // if present, the header is where you move the DIV from:
    document.getElementById(elmnt.id + "header").onmousedown = dragMouseDown;
  } else {
    // otherwise, move the DIV from anywhere inside the DIV:
    elmnt.onmousedown = dragMouseDown;
  }

  function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    // get the mouse cursor position at startup:
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag;
  }

  function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    // calculate the new cursor position:
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    // set the element's new position:
    elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
    elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
  }

  function closeDragElement() {
    // stop moving when mouse button is released:
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

function openLink(link) {
    window.open(link.split('?')[0]);
}

function savePrompt() {
    const prompt = document.getElementById('prompt').value;
    const negativePrompt = document.getElementById('negative_prompt').value;

    fetch('/save_prompt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `prompt=${encodeURIComponent(prompt)}&negative_prompt=${encodeURIComponent(negativePrompt)}`
    })
        .then(response => response.json())
        .then(data => {
            console.log(data.status);
        })
        .catch(error => console.error('Error saving prompt:', error));
}

document.addEventListener('visibilitychange', function () {
    const state =
        document.visibilityState === 'visible'
            ? 'Vis'
            : document.visibilityState === 'hidden'
            ? 'Hid'
            : document.visibilityState === 'prerender'
            ? 'Pre'
            : 'Unk';
    document.title = `Image Generator (${state})`;
    if (state === 'Vis') {
        existingImages.clear();
        document.getElementById('images').innerHTML = '';
    }
});

function stopButtonOnClick(event) {
    fetch('/stop', {
        method: 'POST'
    })
    .catch(error => console.error('Error stopping generation:', error));
};

async function restartButtonOnClick(event) {
    const isConfirmed = await customConfirm.createConfirm(
        'Are you sure you want to restart the server?\nIt will reset all variables and has a chance to fail restarting.',
        [
            { text: 'Restart', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

    if (isConfirmed) {
        isCleared = true;
        fetch('/restart', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('all').style.display = 'none';
        })
        .catch(error => console.error('Error stopping generation:', error));
    }
};

async function clearButtonOnClick(event) {
    const isConfirmed = await customConfirm.createConfirm(
        'Are you sure you want to clear all images?',
        [
            { text: 'Clear', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

    if (isConfirmed) {
        fetch('/clear', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            existingImages.clear();
            document.getElementById('images').innerHTML = '';
        })
        .catch(error => console.error('Error Clearing Images:', error));
    }
};

async function resetValueOnRightClick(event) {
    const isConfirmed = await customConfirm.createConfirm(
        'Are you sure you want to reset this value?',
        [
            { text: 'Reset', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

    if (isConfirmed) {
        event.target.value = "";
    }
};

document.getElementById("prompt").addEventListener("contextmenu", async function(event) {
    if (event.ctrlKey) {
        event.preventDefault();
        await resetValueOnRightClick(event);
    }
});

function updateImageScales() {
    const images = document.querySelectorAll('#images img');
    const value = Number(document.getElementById('img_display_input').value); // Convert value to a number
    images.forEach(img => {
        img.style.width = `${100 / value - 4}vw`;
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const imgInput = document.getElementById('img_input_div');

    dropArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('dragover');
    });

    dropArea.addEventListener('drop', (event) => {
        event.preventDefault();
        dropArea.classList.remove('dragover');

        if (event.target.id !== 'img_input_div') {
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                console.log('Dropped file:', file);
                handleImageDrop(file);
            } else {
                alert('Please drop an image file.');
            }
        }
    });

    imgInput.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropArea.classList.add('dragover');
    });

    imgInput.addEventListener('dragleave', () => {
        dropArea.classList.remove('dragover');
    });

    imgInput.addEventListener('drop', (event) => {
        event.preventDefault();
        dropArea.classList.remove('dragover');

        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imgInput.querySelector('#img_input_img').files = dataTransfer.files;
        } else {
            alert('Please drop an image file.');
        }
    });
});

function handleImageDrop(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
        const uint8Array = new Uint8Array(event.target.result);
        const metadata = extractPngTextChunks(uint8Array);
        console.log('Extracted metadata:', metadata);
        if (metadata) {
            updateFormFields(metadata);
        } else {
            alert("No metadata found in this PNG image.");
        }
    };
    reader.readAsArrayBuffer(file);
}

function extractPngTextChunks(uint8Array) {
    const textChunks = {};
    const pngSignature = [137, 80, 78, 71, 13, 10, 26, 10];
    let position = 8;  // Skip PNG header

    if (!pngSignature.every((byte, index) => uint8Array[index] === byte)) {
        console.error('Not a valid PNG file');
        return null;
    }

    while (position < uint8Array.length) {
        const length = new DataView(uint8Array.buffer).getUint32(position, false);
        const chunkType = String.fromCharCode(...uint8Array.slice(position + 4, position + 8));
        const chunkData = uint8Array.slice(position + 8, position + 8 + length);

        if (chunkType === 'tEXt') {
            const textData = new TextDecoder().decode(chunkData);
            const [key, value] = textData.split('\u0000');
            textChunks[key] = value;
        }

        position += 12 + length;
    }

    return textChunks;
}

function updateFormFields(metadata) {
    const form = document.getElementById('generateForm');
    const fieldMapping = {
        'prompt': 'Prompt',
        'negative_prompt': 'NegativePrompt',
        'width': 'Width',
        'height': 'Height',
        'cfg_scale': 'CFGScale',
        'strength': 'Strength',
        'custom_seed': 'Seed',
        'sampling_steps': 'SamplingSteps',
        'model': 'Model',
        'scheduler': 'Scheduler',
    };

    Object.entries(fieldMapping).forEach(([formField, metaKey]) => {
        if (metadata[metaKey] && metadata[metaKey] !== 'N/A') {
            form.elements[formField].value = metadata[metaKey];
            form.elements[formField].dispatchEvent(new Event('change'));
        }
    });
    console.log('Form fields updated');
}

const select = document.getElementById("model");
const preview = document.getElementById("model-preview");

const showPreview = (event) => {
    const imageUrl = select.options[select.selectedIndex].dataset.src;
    if (imageUrl) {
        preview.src = imageUrl;
        preview.style.display = "block";
        preview.style.height = "50vh";
        preview.style.maxWidth = "70%";
        preview.style.width = "auto";
        preview.style.objectFit = "contain";
        preview.style.position = "absolute";
        updatePosition(event);
    }
};

const updatePosition = (event) => {
    const { pageX: x, pageY: y } = event.touches ? event.touches[0] : event;
    preview.style.left = `${x - preview.getBoundingClientRect().width / 2}px`;
    preview.style.top = `${y + 10}px`;
};

const hidePreview = () => preview.style.display = "none";

// For both mouse and touch events
select.addEventListener("mouseover", showPreview);
select.addEventListener("mousemove", updatePosition);
select.addEventListener("mouseleave", hidePreview);
select.addEventListener("touchstart", showPreview);
select.addEventListener("touchmove", updatePosition);
select.addEventListener("touchend", hidePreview);


//TODO handle prompt example change

const promptSelectElement = document.getElementById('example_prompt');
const promptTextareaElement = document.getElementById('prompt');

// Add an event listener for the 'change' event on the select element
promptSelectElement.addEventListener('change', function() {
    promptTextareaElement.value = promptSelectElement.value; // Update textarea with the selected prompt
});

//TODO handle model change

const modelSelectElement = document.getElementById('model');
const cfgInputElement = document.getElementById('cfg_scale');
const modelTypeInputElement = document.getElementById('model_type');

// Add an event listener for the 'change' event on the select element
modelSelectElement.addEventListener('change', function() {
    cfgInputElement.value = modelSelectElement.options[modelSelectElement.selectedIndex].dataset.cfg || 7;
    modelTypeInputElement.value = modelSelectElement.options[modelSelectElement.selectedIndex].dataset.type || "SDXL";
});

//TODO handle pre dimension change

// Assuming the following elements exist in your HTML
const exampleSizeSelectElement = document.getElementById('example_size');
const widthInputElement = document.getElementById('width');
const heightInputElement = document.getElementById('height');

// Add an event listener for the 'change' event on the example_size select element
exampleSizeSelectElement.addEventListener('change', function() {
    const selectedOption = exampleSizeSelectElement.options[exampleSizeSelectElement.selectedIndex];

    if (selectedOption) {
        // Parse the selected value to get width and height
        const dimensions = selectedOption.value.split('x'); // Assuming the value format is "widthxheight"
        if (dimensions.length === 2) {
            widthInputElement.value = dimensions[0]; // Set width
            heightInputElement.value = dimensions[1]; // Set height
        }
    }
});

// Optionally, set the initial values based on the first option in the select
if (exampleSizeSelectElement.options.length > 0) {
    const initialDimensions = exampleSizeSelectElement.options[0].value.split('x');
    widthInputElement.value = initialDimensions[0];
    heightInputElement.value = initialDimensions[1];
}