function populateThemes(data, select) {
    Object.entries(data).forEach(([themeName, themeTones]) => {
        const option = document.createElement('option');
        // Convert the theme tones object to a JSON string for the option value
        option.value = JSON.stringify(themeTones);
        option.textContent = themeName;
        select.appendChild(option);
    });
}

document.addEventListener('DOMContentLoaded', async function() {
    await loadJsonAndPopulateSelect('/static/json/themes.json', 'theme', populateThemes);
    await fetch("/load_settings")
        .then(response => response.json())
        .then(data => {
            if (data) {
                
                document.getElementById("theme").value = JSON.stringify(data.theme) || '{"tone_1":"240, 240, 240","tone_2":"240, 218, 218","tone_3":"240, 163, 163"}';
                document.getElementById("attention-slicing").value = data.enable_attention_slicing || "False";
                document.getElementById("xformers").value = data.enable_xformers_memory_efficient_attention || "False";
                document.getElementById("cpu-offload").value = data.enable_model_cpu_offload || "False";
                document.getElementById("sequential-cpu").value = data.enable_sequential_cpu_offload || "False";
                document.getElementById("long-clip").value = data.use_long_clip || "False";
                document.getElementById("show-latents").value = data.show_latents || "False";
                document.getElementById("load-previous-data").value = data.load_previous_data || "False";
                document.getElementById("use-multi-prompt").value = data.use_multi_prompt || "False";
                document.getElementById("multi-prompt-separator").value = data.multi_prompt_separator || "§";
                document.getElementById("host").value = data.host || "localhost";
                document.getElementById("port").value = data.port || "8080";
            }
        })
        .catch(error => console.error('Error loading settings:', error));
});

function saveSettings(event) {
    event.preventDefault();

    const settings = {};

    // Handle theme separately
    const themeValue = document.getElementById('theme').value;
    if (themeValue) {
        try {
            const theme = JSON.parse(themeValue);
            settings.theme = theme; // Store theme if valid
        } catch (error) {
            console.error('Error parsing theme:', error);
        }
    }
    else

    // Handle other settings
    settings.enable_attention_slicing = document.getElementById("attention-slicing").value;
    settings.enable_xformers_memory_efficient_attention = document.getElementById("xformers").value;
    settings.enable_model_cpu_offload = document.getElementById("cpu-offload").value;
    settings.enable_sequential_cpu_offload = document.getElementById("sequential-cpu").value;
    settings.use_long_clip = document.getElementById("long-clip").value;
    settings.show_latents = document.getElementById("show-latents").value;
    settings.load_previous_data = document.getElementById("load-previous-data").value;
    settings.use_multi_prompt = document.getElementById("use-multi-prompt").value;
    settings.multi_prompt_separator = document.getElementById("multi-prompt-separator").value.replace(/\\n/g, '\n').replace(/\\r/g, '\r').replace(/\\t/g, '\t');
    settings.host = document.getElementById("host").value;
    settings.port = document.getElementById("port").value;

    fetch('/save_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.status);
    })
    .catch(error => console.error('Error:', error));
}

document.getElementById('theme').addEventListener('change', function () {
    try {
        const theme = JSON.parse(this.value);

        // Set CSS variables
        document.documentElement.style.setProperty('--tone1', theme.tone_1);
        document.documentElement.style.setProperty('--tone2', theme.tone_2);
        document.documentElement.style.setProperty('--tone3', theme.tone_3);
    } catch (error) {
        console.error('Error applying theme:', error);
    }
});
