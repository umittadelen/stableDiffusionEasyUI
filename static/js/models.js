
const customConfirm = new CustomConfirm();

// Fetch and display installed models
async function loadModels() {
    try {
        const response = await fetch('/scan_model_configs');
        if (!response.ok) throw new Error('Error loading models');
        const models = await response.json();
        renderModelList(models);
    } catch (error) {
        document.getElementById('models-list').innerHTML = `<div class="div-border">Error loading models: ${error.message}</div>`;
    }
}

function renderModelList(models) {
    const listDiv = document.getElementById('models-list');
    if (!models || models.length === 0) {
        listDiv.innerHTML = '<div class="div-border">No models found.</div>';
        return;
    }
    listDiv.innerHTML = '';
    models.forEach((model, idx) => {
        const card = document.createElement('div');
        card.className = 'div-border model-card-flex';
        card.style.cursor = 'pointer';
        let imgHtml = '';
        // Only show image if model.images.url exists
        if (model.images && model.images.url) {
            imgHtml = `<div class="model-card-img-wrap"><img src="${model.images.url}" alt="model image" class="model-card-img"></div>`;
        } else {
            imgHtml = `<div class="model-card-img-wrap"></div>`;
        }
        const textHtml = `<div class="model-card-text"><strong>${model.name || model.id || 'Unnamed Model'}</strong><br><span style="font-size:0.9em;opacity:0.7;">${model.type || ''}</span></div>`;
        card.innerHTML = imgHtml + textHtml;
        card.onclick = () => showModelDetails(model, idx);
        listDiv.appendChild(card);
    });
}


let currentModel = null;
let currentModelIdx = null;

function showModelDetails(model, idx) {
    currentModel = model;
    currentModelIdx = idx;
    document.getElementById('model-details-overlay').style.display = 'flex';
    document.getElementById('edit-model-name').value = model.name || '';
    document.getElementById('edit-model-type').value = model.type || '';
    document.getElementById('edit-model-path').value = model.path || '';
    document.getElementById('edit-model-description').value = model.description || '';
    // Center the delete button and add margin (in case of dynamic reload)
    const deleteBtn = document.querySelector('#model-details-form .modal-delete-btn');
    if (deleteBtn) {
        deleteBtn.style.display = 'block';
        deleteBtn.style.margin = '24px auto 0 auto';
        deleteBtn.style.textAlign = 'center';
    }
}

function closeModelDetails() {
    document.getElementById('model-details-overlay').style.display = 'none';
}

// Save model details (dummy, needs backend support)
function saveModelDetails(event) {
    event.preventDefault();
    // Gather edited model details from the form
    const updatedModel = {
        ...currentModel,
        name: document.getElementById('edit-model-name').value,
        type: document.getElementById('edit-model-type').value,
        path: document.getElementById('edit-model-path').value,
        description: document.getElementById('edit-model-description').value
    };

    // Save the updated model config (replace in list and POST whole list)
    fetch('/scan_model_configs')
        .then(r => r.json())
        .then(models => {
            models[currentModelIdx] = updatedModel;
            return fetch('/save_model_configs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(models, null, 2)
            });
        })
        .then(r => {
            if (!r.ok) throw new Error('Failed to save model details');
            closeModelDetails();
            loadModels();
        })
        .catch(error => alert('Error saving model details: ' + error.message));
}

// Delete selected model from details panel
async function deleteSelectedModel() {
    if (!currentModel) return;
    const isConfirmed = await customConfirm.createConfirm(
        `Are you sure you want to delete model <b>${currentModel.name || currentModel.id}</b>?`,
        [ { text: 'Delete', value: true }, { text: 'Cancel', value: false } ],
        false
    );
    if (isConfirmed) {
        fetch('/delete_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `model_name=${encodeURIComponent(currentModel.name || currentModel.id)}`
        })
        .then(r => r.json())
        .then(() => {
            closeModelDetails();
            loadModels();
        })
        .catch(error => alert('Error deleting model: ' + error.message));
    }
}

// Download model form submission
function submitModel(event) {
    event.preventDefault();
    const statusDiv = document.getElementById('model-download-status');
    statusDiv.textContent = 'Starting model download...';
    localStorage.setItem('modelDownloadStatus', 'Starting model download...');
    const formData = new FormData(event.target);
    fetch('/addmodel', {
        method: 'POST',
        body: formData
    })
    .then((response) => response.json())
    .then((data) => {
        loadModels();
        statusDiv.textContent = 'Model download started! (Check progress in the console or logs)';
        localStorage.setItem('modelDownloadStatus', 'Model download started! (Check progress in the console or logs)');
        setTimeout(() => {
            statusDiv.textContent = '';
            localStorage.removeItem('modelDownloadStatus');
        }, 6000);
    })
    .catch((error) => {
        statusDiv.textContent = 'Error: ' + error.message;
        localStorage.setItem('modelDownloadStatus', 'Error: ' + error.message);
    });
}



document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    // Restore model download status from localStorage if present
    const statusDiv = document.getElementById('model-download-status');
    const savedStatus = localStorage.getItem('modelDownloadStatus');
    if (statusDiv && savedStatus) {
        statusDiv.textContent = savedStatus;
        // Optionally clear after a timeout if not an error
        if (!savedStatus.startsWith('Error:')) {
            setTimeout(() => {
                statusDiv.textContent = '';
                localStorage.removeItem('modelDownloadStatus');
            }, 6000);
        }
    }
});