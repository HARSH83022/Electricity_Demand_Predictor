var map = L.map('map').setView([28.6139, 77.2090], 12);
// Set up the OpenStreetMap tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);
// Approximate boundary of New Delhi using polygon coordinates
var newDelhiBoundary = [
    [28.8131, 77.1075],
    [28.8578, 77.2093],
    [28.5145, 77.3445],
    [28.5149, 77.0940],
    [28.7097, 77.0021],
    [28.8131, 77.1075]
];
// Create the New Delhi boundary polygon and add it to the map
var boundaryPolygon = L.polygon(newDelhiBoundary, {
    color: 'blue',
    fillColor: '#3388ff',
    fillOpacity: 0.4
}).addTo(map);
var marker;
// Function to check if a point is inside New Delhi boundary
function isInsideDelhi(latLng) {
    return boundaryPolygon.getBounds().contains(latLng);
}
// Function to update region using reverse geocoding
function updateRegion(latLng) {
    if (isInsideDelhi(latLng)) {
        fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${latLng.lat}&lon=${latLng.lng}`)
            .then(response => response.json())
            .then(data => {
                var region = data.address.suburb || data.address.neighbourhood || data.address.city || data.address.town || "Unknown Region";
                document.getElementById('region').value = region; // Update the region field
                document.getElementById('sendData').disabled = false;  // Enable button after successful region fetch
            })
            .catch(error => {
                console.error('Error in reverse geocoding:', error);
                document.getElementById('region').value = "Unknown Region";
            });
    } else {
        alert('Selected location is outside Delhi boundary.');
        document.getElementById('region').value = "Restricted Area"; // Mark as restricted
        document.getElementById('sendData').disabled = true;  // Disable button if outside Delhi
    }
}
