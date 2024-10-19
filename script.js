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
map.on('click', function (e) {
    var latLng = e.latlng;
    if (isInsideDelhi(latLng)) {
        // If a marker exists, move it; if not, create one
        if (!marker) {
            marker = L.marker(latLng, { draggable: true }).addTo(map);
        } else {
            marker.setLatLng(latLng);
        }
        // Update latitude and longitude input fields
        document.getElementById('latitude').value = latLng.lat.toFixed(6);
        document.getElementById('longitude').value = latLng.lng.toFixed(6);
        // Update region based on the clicked location
        updateRegion(latLng);
        // If the marker is dragged, update the region dynamically
        marker.on('dragend', function () {
            var latLng = marker.getLatLng();
            document.getElementById('latitude').value = latLng.lat.toFixed(6);
            document.getElementById('longitude').value = latLng.lng.toFixed(6);
            // Update region after dragging
            updateRegion(latLng);
        });
    } else {
        if (marker) {
            map.removeLayer(marker);
            marker = null;
        }
        document.getElementById('latitude').value = "";
        document.getElementById('longitude').value = "";
        document.getElementById('region').value = "Restricted Area"; // Mark as restricted
        document.getElementById('sendData').disabled = true;  // Disable the button
        alert('Selected location is outside Delhi boundary.');
    }
});
function fetchWeather(latitude, longitude, date) {
    const apiKey = 'YOUR_OPENWEATHERMAP_API_KEY';  // Replace with your OpenWeatherMap API key
    const timestamp = Math.floor(new Date(date).getTime() / 1000);  // Convert date to UNIX timestamp
    const weatherUrl = `https://api.openweathermap.org/data/2.5/onecall/timemachine?lat=${latitude}&lon=${longitude}&dt=${timestamp}&appid=${apiKey}&units=metric`;

    fetch(weatherUrl)
        .then(response => response.json())
        .then(data => {
            const temp = data.current.temp;
            const weather = data.current.weather[0].description;
            document.getElementById('predicted-weather').innerText = `Temperature: ${temp}°C, Weather: ${weather}`;
        })
        .catch(error => {
            console.error('Error fetching weather data:', error);
            document.getElementById('predicted-weather').innerText = 'Weather data unavailable';
        });
}
// Function to check if the selected date is a holiday
function checkHoliday(date) {
    const apiKey = 'YOUR_CALENDARIFIC_API_KEY';  // Replace with your Calendarific API key
    const holidayUrl = `https://calendarific.com/api/v2/holidays?&api_key=${apiKey}&country=IN&year=${new Date(date).getFullYear()}&month=${new Date(date).getMonth() + 1}&day=${new Date(date).getDate()}`;

    fetch(holidayUrl)
        .then(response => response.json())
        .then(data => {
            const holidays = data.response.holidays;
            if (holidays.length > 0) {
                document.getElementById('holiday-info').innerText = `Holiday: ${holidays[0].name}`;
            } else {
                document.getElementById('holiday-info').innerText = 'No holiday on this day';
            }
        })
        .catch(error => {
            console.error('Error fetching holiday data:', error);
            document.getElementById('holiday-info').innerText = 'Holiday data unavailable';
        });
}
document.getElementById('sendData').addEventListener('click', function () {
    var latitude = document.getElementById('latitude').value;
    var longitude = document.getElementById('longitude').value;
    var region = document.getElementById('region').value;
    var date = document.getElementById('date').value;

    // Ensure the location is within Delhi boundary before proceeding
    if (region !== "Restricted Area") {
        // Fetch weather data
        fetchWeather(latitude, longitude, date);
        // Check if the selected date is a holiday
        checkHoliday(date);
        // Simulate load prediction calculation
        setTimeout(function () {
            document.getElementById('predicted-load').innerText = (Math.random() * 2000).toFixed(2) + ' MWh';
        }, 1000);
    } else {
        alert('Cannot fetch data for restricted area.');
    }
});
