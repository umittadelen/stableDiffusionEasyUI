async function loadTheme() {
    let savedData = "";

    try {
        const response = await fetch("/load_settings");
        const data = await response.json();

        // Fallback to default theme if no theme is found
        savedData = data.theme || {
            "tone_1": "18, 18, 24",
            "tone_2": "32, 34, 48",
            "tone_3": "200, 210, 230"
        };

        // Apply the saved theme to the document
        document.documentElement.style.setProperty('--tone1', savedData.tone_1, 'important');
        document.documentElement.style.setProperty('--tone2', savedData.tone_2, 'important');
        document.documentElement.style.setProperty('--tone3', savedData.tone_3, 'important');
    } catch (error) {
        console.error('Error loading settings:', error);

        // Use default theme in case of error
        const defaultTheme = {
            "tone_1": "18, 18, 24",
            "tone_2": "32, 34, 48",
            "tone_3": "200, 210, 230"
        };

        document.documentElement.style.setProperty('--tone1', defaultTheme.tone_1, 'important');
        document.documentElement.style.setProperty('--tone2', defaultTheme.tone_2, 'important');
        document.documentElement.style.setProperty('--tone3', defaultTheme.tone_3, 'important');
    }
}

window.onload = function () {
    loadTheme();
};
