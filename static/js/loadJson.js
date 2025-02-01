// Reusable function to load JSON and populate a select element
async function loadJsonAndPopulateSelect(location, selectId, dataHandler) {
    try {
        const response = await fetch(location);
        if (!response.ok) {
            throw new Error(`Error loading ${location}`);
        }
        const data = await response.json();
        dataHandler(data, document.getElementById(selectId));
        return data;
    } catch (error) {
        console.error('Error:', error);
        return {};
    }
}