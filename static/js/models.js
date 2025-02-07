const customConfirm = new CustomConfirm();

// Function to handle form submission
function submitModel(event) {
    event.preventDefault(); // Prevent default form submission

    // Use event.target to get the form element
    const formData = new FormData(event.target);
    fetch('/addmodel', {
        method: 'POST',
        body: formData
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// Load JSON file into the editor
function loadJson() {
    fetch('/scan_model_configs')
        .then(response => {
            if (!response.ok) {
                throw new Error('Error loading JSON file');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('json-editor').value = JSON.stringify(data, null, 4);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('json-status').innerHTML = `Error loading JSON: ${error.message}`;
        });
}

// Save the JSON content
function saveJson() {
    const jsonContent = document.getElementById('json-editor').value;

    fetch('/save_model_configs', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: jsonContent
    })
    .then((response) => {
        if (!response.ok) {
            throw new Error('Failed to save JSON');
        }
        return response.json();
    })
    .then((data) => {
        console.log('Success:', data);
        document.getElementById('json-status').innerHTML = 'JSON saved successfully!';
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('json-status').innerHTML = `Error saving JSON: ${error.message}`;
    });
}

// Validate JSON input
function validateJson() {
    const jsonContent = document.getElementById('json-editor').value;
    try {
        JSON.parse(jsonContent);
        document.getElementById('json-status').innerHTML = 'Valid JSON';
    } catch {
        document.getElementById('json-status').innerHTML = 'Invalid JSON';
    }
}

// Delete Model
async function deleteModel() {
    const isConfirmed = await customConfirm.createConfirm(
        'Are you sure you want to delete the model?',
        [
            { text: 'Delete', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

    if (isConfirmed) {
        const modelName = document.getElementById('deleteModelId').value;
        fetch('/delete_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `model_name=${modelName}`
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error('Failed to delete model');
            }
            return response.json();
        })
        .then((data) => {
            console.log('Success:', data);
            document.getElementById('delete-status').innerHTML = 'Model deleted successfully!';
        })
        .catch((error) => {
            console.error('Error:', error);
            document.getElementById('delete-status').innerHTML = `Error deleting model: ${error.message}`;
        });
    }
}

// Load JSON when the page loads
document.addEventListener('DOMContentLoaded', loadJson);