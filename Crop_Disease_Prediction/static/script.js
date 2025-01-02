// Handle real-time image capture from webcam
const translations = {
    en: {
        title: "Crop Disease Detection",
        home: "Home",
        about: "About",
        heroTitle: "Empowering Agriculture with Smart Disease Detection",
        heroText: "Detect crop diseases easily and improve your harvest with AI-powered insights.",
        uploadTitle: "Upload Image",
        detectButton: "Detect Disease",
        cameraTitle: "Or Capture Image in Real-Time",
        startCamera: "Start Camera",
        captureImage: "Capture Image",
        footerText: "Crop Disease Detection © 2024 | All rights reserved.",
        resultsTitle: "Detection Results",
        goBack: "Go Back",
        cropLabel: "Crop:",
        diseaseLabel: "Disease:",
        confidenceLabel: "Confidence:",
        symptomsLabel: "Symptoms:",
        curesLabel: "Cures:"
    },
    hi: {
        title: "फसल रोग पहचान",
        home: "होम",
        about: "के बारे में",
        heroTitle: "स्मार्ट रोग पहचान के साथ कृषि को सशक्त बनाना",
        heroText: "कृत्रिम बुद्धिमत्ता संचालित अंतर्दृष्टि के साथ फसल रोगों का आसानी से पता लगाएं और अपनी फसल में सुधार करें।",
        uploadTitle: "छवि अपलोड करें",
        detectButton: "रोग का पता लगाएं",
        cameraTitle: "या वास्तविक समय में छवि कैप्चर करें",
        startCamera: "कैमरा प्रारंभ करें",
        captureImage: "छवि कैप्चर करें",
        footerText: "फसल रोग पहचान © 2024 | सभी अधिकार सुरक्षित।",
        resultsTitle: "पहचान परिणाम",
        goBack: "वापस जाएं",
        cropLabel: "फसल:",
        diseaseLabel: "रोग:",
        confidenceLabel: "विश्वसनीयता:",
        symptomsLabel: "लक्षण:",
        curesLabel: "उपचार:"
    }
};

function translateContent(lang) {
    for (const key in translations[lang]) {
        const elements = document.querySelectorAll(`[data-translate="${key}"]`);
        elements.forEach(element => {
            element.textContent = translations[lang][key];
        });
    }
    // Update the hidden language input field in the form (if present)
    // const languageInput = document.getElementById('selectedLanguage');
    // if (languageInput) {
    //     languageInput.value = lang;
    // }
    document.getElementById('selectedLanguage').value = lang; //*** Update hidden input value
}



document.getElementById('language').addEventListener('change', function () {
    translateContent(this.value);
});

function startCamera() {
    const video = document.getElementById('videoInput');
    const captureButton = document.getElementById('captureButton');
    const startCameraButton = document.getElementById('startCameraButton');
    const canvas = document.getElementById('canvasOutput');
    const context = canvas.getContext('2d');

    // Access webcam when "Start Camera" is clicked
    startCameraButton.addEventListener('click', () => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                video.srcObject = stream;
                video.style.display = "block"; // Show video element
                captureButton.style.display = "inline-block"; // Show capture button
                startCameraButton.style.display = "none"; // Hide start button
                video.play();
            })
            .catch((err) => {
                console.error("Error accessing webcam: ", err);
            });
    });

    // Capture the image when "Capture Image" is clicked
    captureButton.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Compress and upload the captured image
        canvas.toBlob(uploadImage, 'image/jpeg', 0.7); // Compress to 70% quality
    });
}

// Upload image captured or selected by the user
function uploadImage(blob) {
    const formData = new FormData();
    formData.append('image', blob, 'captured.jpg');

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (response.ok) return response.text();
            else throw new Error("Failed to upload");
        })
        .then(data => {
            document.body.innerHTML = data; // Replace the current page with the results page
        })
        .catch(err => {
            console.error("Error uploading image: ", err);
            alert("An error occurred while processing the image. Please try again.");
        });
}

// Auto-start functionality removed; camera will only start when the button is clicked
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('videoInput')) {
        startCamera();
    // if (document.getElementById('language')) { // Call this on pages with the language selector
    //     translateContent(document.getElementById('language').value); // Initialize translations 
    //     }
    }
    let initialLanguage = "en";
    if ("{{ selected_language }}" !== "") {  // Use Jinja2 template engine syntax
        initialLanguage = "{{ selected_language }}";
    }

    const languageSelect = document.getElementById('language');
    if (languageSelect) {
        languageSelect.value = initialLanguage;  // Set selected option correctly
    }
    translateContent(initialLanguage);
   
    
  //Call after setting value
});


// let initialLanguage = "en"; // Default
//     const languageSelect = document.getElementById('language');

//     //Check if language is explicitly set (on result.html for example)
//     if("{{selected_language}}" != ""){
//         initialLanguage = "{{ selected_language }}";
//     }

//     if (languageSelect) {
//         languageSelect.value = initialLanguage; // Set dropdown to correct language
//     }
//     translateContent(initialLanguage);