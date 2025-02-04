function isString(value) {
	return typeof value === 'string' || value instanceof String;
}

// Reusable function to load JSON and populate a select element
async function loadJsonAndPopulateSelect(location, selectId, dataHandler) {
    try {
        if (isString(location)) {
            const response = await fetch(location);
            if (!response.ok) {
                throw new Error(`Error loading ${location}`);
            }
            const data = await response.json();
            dataHandler(data, document.getElementById(selectId));
            return data;
        }
        else {
            dataHandler(location, document.getElementById(selectId));
            return location;
        }
        
    } catch (error) {
        console.error('Error:', error);
        return {};
    }
}