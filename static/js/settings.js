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
                document.getElementById("attention-slicing").value = data.enable_attention_slicing ? "True" : "False";
                document.getElementById("xformers").value = data.enable_xformers_memory_efficient_attention ? "True" : "False";
                document.getElementById("cpu-offload").value = data.enable_model_cpu_offload ? "True" : "False";
                document.getElementById("sequential-cpu").value = data.enable_sequential_cpu_offload ? "True" : "False";
                document.getElementById("long-clip").value = data.use_long_clip ? "True" : "False";
                document.getElementById("long-clip-model").value = data.long_clip_model || "zer0int/LongCLIP-GmP-ViT-L-14";
                document.getElementById("fallback-vae-model").value = data.fallback_vae_model || "clip-vae";
                document.getElementById("default-clip-model").value = data.default_clip_model || "clip-vit-l-14-clip";
                document.getElementById("fallback-tokenizer-model").value = data.fallback_tokenizer_model || "openai/clip-vit-base-patch16";
                document.getElementById("image-size").value = data.image_size || "100";
                document.getElementById("update-interwal").value = data.update_interwal || "2500";
                document.getElementById("update-page-in-background").value = data.update_page_in_background ? "True" : "False";
                document.getElementById("show-model-preview").value = data.show_model_preview ? "True" : "False";
                document.getElementById("show-latents").value = data.show_latents ? "True" : "False";
                document.getElementById("load-previous-data").value = data.load_previous_data ? "True" : "False";
                document.getElementById("reset-on-new-request").value = data.reset_on_new_request ? "True" : "False";
                document.getElementById("reverse-image-order").value = data.reverse_image_order ? "True" : "False";
                document.getElementById("use-multi-prompt").value = data.use_multi_prompt ? "True" : "False";
                document.getElementById("multi-prompt-separator").value = JSON.stringify(data.multi_prompt_separator).slice(1, -1) || "§";
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
    settings.enable_attention_slicing = document.getElementById("attention-slicing").value === "True";
    settings.enable_xformers_memory_efficient_attention = document.getElementById("xformers").value === "True";
    settings.enable_model_cpu_offload = document.getElementById("cpu-offload").value === "True";
    settings.enable_sequential_cpu_offload = document.getElementById("sequential-cpu").value === "True";
    settings.use_long_clip = document.getElementById("long-clip").value === "True";
    settings.long_clip_model = document.getElementById("long-clip-model").value;
    settings.fallback_vae_model = document.getElementById("fallback-vae-model").value;
    settings.default_clip_model = document.getElementById("default-clip-model").value;
    settings.fallback_tokenizer_model = document.getElementById("fallback-tokenizer-model").value;
    settings.image_size = document.getElementById("image-size").value;
    settings.update_interwal = document.getElementById("update-interwal").value;
    settings.update_page_in_background = document.getElementById("update-page-in-background").value === "True";
    settings.show_model_preview = document.getElementById("show-model-preview").value === "True";
    settings.show_latents = document.getElementById("show-latents").value === "True";
    settings.load_previous_data = document.getElementById("load-previous-data").value === "True";
    settings.reset_on_new_request = document.getElementById("reset-on-new-request").value === "True";
    settings.reverse_image_order = document.getElementById("reverse-image-order").value === "True";
    settings.use_multi_prompt = document.getElementById("use-multi-prompt").value === "True";
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
