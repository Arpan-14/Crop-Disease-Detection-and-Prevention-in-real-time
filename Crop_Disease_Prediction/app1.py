from flask import Flask, render_template, request, jsonify
import os
import torch
from PIL import Image
from torchvision import transforms, models as mod
import torch.nn as nn

# Initialize the Flask app
app = Flask(__name__)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model paths
model_paths = {
    'Rice': 'Models/Rice.pth',
    'Corn': 'Models/Corn.pth',
    'Wheat': 'Models/Wheat.pt'
}

# Define classes
classes = {
    'Rice': ['Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast'],
    'Corn': ['Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight'],
    'Wheat': ['Wheat___Yellow_Rust', 'Wheat___Healthy', 'Wheat___Brown_Rust']
}

translations_ch = {  # Chhattisgarh context
    'en': {  # English
        'Rice': 'Rice',
        'Corn': 'Corn',
        'Wheat': 'Wheat',
        'Rice___Brown_Spot': 'Rice - Brown Spot',  # Example with hyphen
        'Rice___Healthy': 'Rice - Healthy',
        'Rice___Leaf_Blast': 'Rice - Leaf Blast',
        'Rice___Neck_Blast': 'Rice - Neck Blast',
        'Corn___Common_Rust': 'Corn - Common Rust',
        'Corn___Gray_Leaf_Spot': 'Corn - Gray Leaf Spot',
        'Corn___Healthy': 'Corn - Healthy',
        'Corn___Northern_Leaf_Blight': 'Corn - Northern Leaf Blight',
        'Wheat___Yellow_Rust': 'Wheat - Yellow Rust',
        'Wheat___Healthy': 'Wheat - Healthy',
        'Wheat___Brown_Rust': 'Wheat - Brown Rust'

    },
    'hi': {  # Hindi
        'Rice': 'चावल',
        'Corn': 'मक्का',
        'Wheat': 'गेहूं',
        'Rice___Brown_Spot': 'चावल - भूरा धब्बा',
        'Rice___Healthy': 'चावल - स्वस्थ',
        'Rice___Leaf_Blast': 'चावल - पत्ता विस्फोट',
        'Rice___Neck_Blast': 'चावल - गर्दन विस्फोट',
        'Corn___Common_Rust': 'मक्का - साधारण जंग',
        'Corn___Gray_Leaf_Spot': 'मक्का - ग्रे लीफ स्पॉट',
        'Corn___Healthy': 'मक्का - स्वस्थ',
        'Corn___Northern_Leaf_Blight': 'मक्का - उत्तरी पत्ता अंगमारी',
        'Wheat___Yellow_Rust': 'गेहूं - पीला जंग',
        'Wheat___Healthy': 'गेहूं - स्वस्थ',
        'Wheat___Brown_Rust': 'गेहूं - भूरा जंग'

    }

}

# Symptoms and cures data
# symptoms_and_cures = {
#     "Rice___Brown_Spot": {"symptoms": "Small brown spots with gray centers on leaves.", "cures": "Use fungicides, rotate crops, and ensure good drainage."},
#     "Rice___Healthy": {"symptoms": "No visible symptoms.", "cures": "Maintain optimal conditions for rice growth."},
#     "Rice___Leaf_Blast": {"symptoms": "Diamond-shaped lesions on leaves.", "cures": "Apply resistant varieties and fungicides."},
#     "Rice___Neck_Blast": {"symptoms": "Infection at the neck node causing neck rot.", "cures": "Use fungicides and manage nitrogen levels."},
#     "Corn___Common_Rust": {"symptoms": "Orange pustules on leaves.", "cures": "Apply fungicides and plant resistant varieties."},
#     "Corn___Gray_Leaf_Spot": {"symptoms": "Small gray lesions on leaves.", "cures": "Improve air circulation and use fungicides."},
#     "Corn___Healthy": {"symptoms": "No visible symptoms.", "cures": "Maintain proper field hygiene."},
#     "Corn___Northern_Leaf_Blight": {"symptoms": "Cigar-shaped gray lesions on leaves.", "cures": "Apply fungicides and use resistant varieties."},
#     "Wheat___Yellow_Rust": {"symptoms": "Yellow streaks on leaves.", "cures": "Use fungicides and resistant cultivars."},
#     "Wheat___Healthy": {"symptoms": "No visible symptoms.", "cures": "Maintain good agricultural practices."},
#     "Wheat___Brown_Rust": {"symptoms": "Brown pustules on leaves.", "cures": "Use resistant varieties and fungicides."}
# }

symptoms_and_cures_ch = {
    "Rice___Brown_Spot": {
        "symptoms_en": [
            "Small brown spots with gray centers on leaves.",
            "Lesions enlarge and become oval or round.",
            "Spots may merge to form large blotches.",
            "Severe infection can cause leaf drying and yield loss.",
            "Disease is favored by high humidity and warm temperatures."
        ],
        "cures_en": [
            "Use resistant varieties like MTU 1010 and Indira Baras.",
            "Apply fungicides like Carbendazim or Mancozeb.",
            "Practice crop rotation with non-host crops.",
            "Ensure proper field sanitation by removing infected debris.",
            "Maintain balanced nitrogen fertilization."
        ],
        "symptoms_hi": [
            "पत्तियों पर भूरे रंग के धब्बे जिनके केंद्र में धूसर रंग होता है।",
            "धब्बे बड़े होकर अंडाकार या गोलाकार हो जाते हैं।",
            "धब्बे मिलकर बड़े धब्बे बना सकते हैं।",
            "गंभीर संक्रमण पत्तियों के सूखने और उपज के नुकसान का कारण बन सकता है।",
            "उच्च नमी और गर्म तापमान से रोग बढ़ता है।"
        ],
        "cures_hi": [
            "एमटीयू 1010 और इंदिरा बरस जैसी प्रतिरोधी किस्मों का प्रयोग करें।",
            "कार्बेन्डाजिम या मैंकोज़ेब जैसे कवकनाशी का प्रयोग करें।",
            "गैर-मेजबान फसलों के साथ फसल चक्र अपनाएं।",
            "संक्रमित मलबे को हटाकर उचित खेत स्वच्छता सुनिश्चित करें।",
            "संतुलित नाइट्रोजन उर्वरीकरण बनाए रखें।"
        ]
    },
    "Rice___Healthy": {
        "symptoms_en": [
            "Leaves are green and healthy.",
            "No visible signs of disease or stress.",
            "Plants are vigorous and growing well.",
            "Healthy root system.",
            "Expected yield potential."
        ],
        "cures_en": [
             "Maintain good agricultural practices.",
            "Monitor regularly for any signs of pests or diseases.",
            "Ensure adequate water and nutrient supply.",
            "Practice crop rotation to maintain soil health.",
            "Consult with agricultural experts for best practices."


        ],
        "symptoms_hi": [
            "पत्तियां हरी और स्वस्थ हैं।",
            "बीमारी या तनाव के कोई लक्षण दिखाई नहीं दे रहे हैं।",
            "पौधे जोरदार और अच्छी तरह से बढ़ रहे हैं।",
            "जड़ प्रणाली स्वस्थ है।",
            "उपज की अपेक्षित क्षमता।"

        ],
        "cures_hi": [
            "अच्छे कृषि व्यवहार बनाए रखें।",
            "कीटों या बीमारियों के किसी भी लक्षण के लिए नियमित रूप से निगरानी करें।",
            "पर्याप्त पानी और पोषक तत्वों की आपूर्ति सुनिश्चित करें।",
            "मिट्टी के स्वास्थ्य को बनाए रखने के लिए फसल चक्र अपनाएं।",
            "सर्वोत्तम कृषि पद्धतियों के लिए कृषि विशेषज्ञों से परामर्श लें।"
        ]
    },
    "Rice___Leaf_Blast": {
        "symptoms_en": [
            "Diamond-shaped lesions on leaves.",
            "Lesions may be grayish-white or bluish-green.",
            "Lesions can enlarge and coalesce, causing leaf blight.",
            "Severe infection can affect panicles and reduce yield.",
            "Disease is favored by high humidity, cool nights, and warm days."
        ],
        "cures_en": [
            "Use resistant varieties like Tetep and Kalinga III.",
            "Apply fungicides like Tricyclazole or Propiconazole.",
            "Avoid excessive nitrogen fertilization.",
            "Manage planting density to improve air circulation.",
            "Remove and destroy infected plant debris."


        ],
        "symptoms_hi": [
            "पत्तियों पर हीरे के आकार के घाव।",
            "घाव भूरे-सफेद या नीले-हरे रंग के हो सकते हैं।",
            "घाव बड़े हो सकते हैं और आपस में मिल सकते हैं, जिससे पत्तियां झुलस जाती हैं।",
            "गंभीर संक्रमण फूलों को प्रभावित कर सकता है और उपज को कम कर सकता है।",
            "उच्च नमी, ठंडी रातें और गर्म दिन इस रोग को बढ़ावा देते हैं।"
        ],
        "cures_hi": [
            "टेटेप और कलिंग III जैसी प्रतिरोधी किस्मों का प्रयोग करें।",
            "ट्राइसाइक्लाज़ोल या प्रोपिकोनाज़ोल जैसे कवकनाशी का प्रयोग करें।",
            "अत्यधिक नाइट्रोजन उर्वरीकरण से बचें।",
            "हवा के संचार को बेहतर बनाने के लिए पौधों के घनत्व का प्रबंधन करें।",
            "संक्रमित पौधों के मलबे को हटा दें और नष्ट कर दें।"

        ]
    },
   "Rice___Neck_Blast": {
        "symptoms_en": [
             "Grayish-brown lesions at the neck node of the panicle.",
            "Lesions can girdle the neck, causing the panicle to droop and break.",
            "Infected panicles produce empty or partially filled grains.",
            "Disease can cause significant yield losses, especially during grain filling.",
            "High humidity and moderate temperatures favor disease development."
        ],
        "cures_en": [
            "Plant resistant or tolerant rice varieties.",
            "Apply recommended fungicides at the appropriate growth stages.",
            "Avoid excessive nitrogen application during the susceptible stage.",
            "Ensure proper drainage and avoid waterlogging in the field.",
            "Practice crop rotation and destroy infected residue."
        ],
        "symptoms_hi": [
            "धान के फूल के नीचे के भाग पर भूरे रंग के धब्बे।",
            "रोग फूल के तने के चारों ओर फैल सकता है, जिससे फूल गिर सकता है या टूट सकता है।",
            "संक्रमित फूलों में खाली या आंशिक रूप से भरे हुए दाने होते हैं।",
            "यह रोग उपज को काफी नुकसान पहुंचा सकता है, खासकर अनाज भरने के समय।",
            "उच्च नमी और मध्यम तापमान रोग को बढ़ावा देते हैं।"

        ],
        "cures_hi": [
            "प्रतिरोधी या सहनशील धान की किस्में लगाएं।",
            "अनुशंसित कवकनाशी को पौधे के विकास के उपयुक्त चरणों में प्रयोग करें।",
            "संवेदनशील अवस्था के दौरान अत्यधिक नाइट्रोजन प्रयोग से बचें।",
            "खेत में उचित जल निकासी सुनिश्चित करें और जलभराव से बचें।",
            "फसल चक्र अपनाएं और संक्रमित अवशेषों को नष्ट करें।"
        ]
    },
        "Corn___Common_Rust": {
        "symptoms_en": [
           "Small, circular to elongated, reddish-brown pustules on leaves.",
            "Pustules may appear on both leaf surfaces.",
            "As the disease progresses, pustules turn darker and become more numerous.",
            "Severe infections can cause leaf yellowing and premature death.",
            "The fungus survives on infected corn debris and spreads by wind."

        ],
        "cures_en": [
            "Plant resistant or tolerant hybrids whenever available.",
            "Apply fungicides containing active ingredients like tebuconazole or trifloxystrobin.",
            "Rotate crops to non-host species like soybeans or other legumes.",
            "Time planting to avoid periods of high disease pressure.",
            "Destroy infected crop residue after harvest."

        ],
        "symptoms_hi": [
            "पत्तियों पर छोटे, गोल से लेकर लंबे, लाल-भूरे रंग के फुंसी।",
            "फुंसी पत्ती के दोनों तरफ दिखाई दे सकते हैं।",
            "जैसे-जैसे रोग बढ़ता है, फुंसी का रंग गहरा होता जाता है और उनकी संख्या बढ़ती जाती है।",
            "गंभीर संक्रमण से पत्तियां पीली पड़ सकती हैं और समय से पहले मर सकती हैं।",
            "कवक संक्रमित मक्का के मलबे पर जीवित रहता है और हवा से फैलता है।"

        ],
        "cures_hi": [
            "जब भी उपलब्ध हो, प्रतिरोधी या सहनशील संकर किस्में लगाएं।",
            "टेबुकोनाज़ोल या ट्राइफ्लॉक्सीस्ट्रोबिन जैसे सक्रिय तत्वों वाले कवकनाशी का प्रयोग करें।",
            "सोयाबीन या अन्य फलियों जैसी गैर-मेजबान प्रजातियों के साथ फसल चक्र अपनाएं।",
            "रोपण के लिए ऐसा समय चुनें जब बीमारी का दबाव कम हो।",
            "कटाई के बाद संक्रमित फसल के अवशेषों को नष्ट कर दें।"

        ]
    },
        "Corn___Gray_Leaf_Spot": {
        "symptoms_en": [
            "Small, tan to gray lesions with reddish-brown margins on leaves.",
            "Lesions are rectangular or oval and may be surrounded by a yellow halo.",
            "Lesions can coalesce to form large, blighted areas on the leaves.",
            "Disease development is favored by warm temperatures and high humidity.",
            "The fungus overwinters in corn residue and spreads by wind and rain splash."

        ],
        "cures_en": [
            "Plant resistant hybrids if available in your area.",
            "Fungicide application may be necessary, especially in areas with high disease pressure.",
            "Rotate to non-host crops for at least two years to reduce fungal inoculum.",
            "Till under crop debris after harvest to promote decomposition and reduce overwintering of the fungus.",
            "Avoid excessive nitrogen fertilization, as it can promote lush growth and increase susceptibility to gray leaf spot."

        ],
        "symptoms_hi": [
            "पत्तियों पर हल्के भूरे से धूसर रंग के धब्बे, लाल-भूरे किनारों के साथ।",
            "धब्बे आयताकार या अंडाकार होते हैं और उनके चारों ओर एक पीला घेरा हो सकता है।",
            "धब्बे मिलकर पत्तियों पर बड़े, झुलसे हुए क्षेत्र बना सकते हैं।",
            "गर्म तापमान और उच्च नमी रोग के विकास के लिए अनुकूल होती है।",
            "कवक मक्का के अवशेषों में सर्दियां बिताता है और हवा और बारिश की छींटों से फैलता है।"


        ],
        "cures_hi": [
            "यदि आपके क्षेत्र में उपलब्ध हो तो प्रतिरोधी संकर किस्में लगाएं।",
            "कवकनाशी का प्रयोग आवश्यक हो सकता है, खासकर उन क्षेत्रों में जहां रोग का दबाव अधिक होता है।",
            "कवक संक्रमण को कम करने के लिए कम से कम दो साल तक गैर-मेजबान फसलों के साथ फसल चक्र अपनाएं।",
            "कवक के अपघटन को बढ़ावा देने और सर्दियों में उसके जीवित रहने को कम करने के लिए कटाई के बाद फसल के मलबे को मिट्टी में मिला दें।",
            "अत्यधिक नाइट्रोजन उर्वरीकरण से बचें, क्योंकि इससे पौधे का विकास तेजी से होता है और धूसर पत्ती धब्बे के प्रति संवेदनशीलता बढ़ जाती है।"
        ]
    },
        "Corn___Healthy": {
        "symptoms_en": [
             "Leaves are a healthy green color.",
            "No visible lesions, spots, or discoloration.",
            "Stalks are strong and upright.",
            "Ears develop normally.",
            "Plant exhibits vigorous growth."

        ],
        "cures_en": [
            "Maintain optimal soil health through proper fertilization and pH management.",
            "Scout regularly for pests and diseases and take appropriate action if necessary.",
            "Ensure adequate but not excessive watering, avoiding waterlogging.",
            "Choose appropriate corn hybrids for your specific growing conditions and climate.",
            "Follow recommended planting densities to ensure good air circulation and light penetration."


        ],
        "symptoms_hi": [
            "पत्तियां स्वस्थ हरे रंग की होती हैं।",
            "कोई दृश्यमान घाव, धब्बे या रंग परिवर्तन नहीं।",
            "तने मजबूत और सीधे होते हैं।",
            "भुट्टे सामान्य रूप से विकसित होते हैं।",
            "पौधा जोरदार विकास दर्शाता है।"


        ],
        "cures_hi": [
           "उचित उर्वरीकरण और pH प्रबंधन के माध्यम से इष्टतम मिट्टी स्वास्थ्य बनाए रखें।",
            "कीटों और बीमारियों के लिए नियमित रूप से जांच करें और यदि आवश्यक हो तो उचित कार्रवाई करें।",
            "पर्याप्त लेकिन अत्यधिक पानी न दें, जलभराव से बचें।",
            "अपनी विशिष्ट बढ़ती परिस्थितियों और जलवायु के लिए उपयुक्त मक्का संकर चुनें।",
            "अच्छे वायु परिसंचरण और प्रकाश प्रवेश सुनिश्चित करने के लिए अनुशंसित रोपण घनत्व का पालन करें।"

        ]
    },
        "Corn___Northern_Leaf_Blight": {
        "symptoms_en": [
             "Long, elliptical or cigar-shaped lesions on leaves.",
            "Lesions are tan to gray in color.",
            "Lesions may have a dark brown border.",
            "Severe infections can lead to significant leaf blight and reduced yield.",
            "Disease is favored by wet, humid weather."
        ],
        "cures_en": [
             "Choose resistant or tolerant hybrids if they are locally available.",
            "Apply fungicides with active ingredients effective against northern corn leaf blight.",
            "Rotate crops with non-host species for at least two years.",
            "Manage crop residue by burying or chopping it after harvest to reduce fungal survival.",
            "Plant at recommended densities and ensure adequate spacing to promote good air circulation."



        ],
        "symptoms_hi": [
            "पत्तियों पर लम्बे, अण्डाकार या सिगार के आकार के धब्बे।",
            "धब्बे हल्के भूरे से धूसर रंग के होते हैं।",
            "धब्बों का रंग गहरे भूरे रंग की सीमा रेखा के साथ हो सकता है।",
            "गंभीर संक्रमण से पत्तियां झुलस सकती हैं और पैदावार कम हो सकती है।",
            "यह रोग गीले, आर्द्र मौसम में बढ़ता है।"


        ],
        "cures_hi": [
            "यदि स्थानीय रूप से उपलब्ध हो तो प्रतिरोधी या सहनशील संकर चुनें।",
            "उत्तरी मक्का पत्ती अंगमारी के खिलाफ प्रभावी सक्रिय तत्वों वाले कवकनाशी का प्रयोग करें।",
            "कम से कम दो साल के लिए गैर-मेजबान प्रजातियों के साथ फसल चक्र अपनाएं।",
            "कटाई के बाद फसल के अवशेषों को दफनाने या काटने से कवक के जीवित रहने को कम करें।",
            "अनुशंसित घनत्व पर पौधे लगाएं और अच्छे वायु परिसंचरण को बढ़ावा देने के लिए पर्याप्त दूरी सुनिश्चित करें।"

        ]
    },
 "Wheat___Yellow_Rust": {
        "symptoms_en": [
            "Yellow to orange pustules, or stripes, appear on the upper surfaces of leaves and leaf sheaths.",
            "Pustules contain yellow-orange spores of the fungus.",
            "Infected leaves may turn yellow and die prematurely.",
            "Symptoms are most visible during periods of high humidity.",
            "Yield losses can be significant in severe epidemics."
        ],
        "cures_en": [
            "Plant resistant or tolerant varieties.",
            "Use recommended seed treatments.",
            "Apply fungicides when disease symptoms first appear.",
            "Avoid late planting.",
            "Manage crop residue to reduce the carryover of inoculum from season to season."


        ],
        "symptoms_hi": [
             "पत्तियों और पत्ती के आवरण की ऊपरी सतहों पर पीले से नारंगी रंग के फुंसी या धारियां दिखाई देती हैं।",
            "फुंसियों में कवक के पीले-नारंगी रंग के बीजाणु होते हैं।",
            "संक्रमित पत्तियां पीली पड़ सकती हैं और समय से पहले मर सकती हैं।",
            "उच्च आर्द्रता की अवधि के दौरान लक्षण सबसे अधिक दिखाई देते हैं।",
            "गंभीर महामारी में पैदावार का नुकसान महत्वपूर्ण हो सकता है।"

        ],
        "cures_hi": [
             "प्रतिरोधी या सहनशील किस्में लगाएं।",
            "अनुशंसित बीज उपचार का प्रयोग करें।",
            "जब रोग के लक्षण पहली बार दिखाई दें तो कवकनाशी का प्रयोग करें।",
            "देर से बुवाई से बचें।",
            "मौसम दर मौसम इनोकुलम के हस्तांतरण को कम करने के लिए फसल के अवशेषों का प्रबंधन करें।"


        ]
    },
  "Wheat___Healthy": {
        "symptoms_en": [
            "Healthy wheat plants exhibit vigorous growth.",
            "Leaves are uniformly green, without spots, lesions, or discoloration.",
            "Stems are strong and upright.",
            "Heads emerge and develop normally.",
            "Grain filling occurs efficiently."
        ],
        "cures_en": [
            "Select wheat varieties adapted to local climate and soil conditions.",
            "Maintain adequate soil fertility through balanced fertilization.",
            "Monitor soil moisture and irrigate as needed to avoid water stress.",
            "Implement integrated pest management strategies to control insect pests and diseases.",
            "Rotate with non-cereal crops to improve soil health and break disease cycles."


        ],
        "symptoms_hi": [
            "स्वस्थ गेहूं के पौधे जोरदार विकास दर्शाते हैं।",
            "पत्तियां समान रूप से हरी होती हैं, बिना धब्बे, घाव या रंग परिवर्तन के।",
            "तने मजबूत और सीधे होते हैं।",
            "बाली सामान्य रूप से निकलती और विकसित होती है।",
            "अनाज अच्छी तरह से भरता है।"


        ],
        "cures_hi": [
            "स्थानीय जलवायु और मिट्टी की स्थिति के अनुकूल गेहूं की किस्में चुनें।",
            "संतुलित उर्वरीकरण के माध्यम से पर्याप्त मिट्टी की उर्वरता बनाए रखें।",
            "पानी के तनाव से बचने के लिए मिट्टी की नमी की निगरानी करें और आवश्यकतानुसार सिंचाई करें।",
            "कीटों और बीमारियों को नियंत्रित करने के लिए एकीकृत कीट प्रबंधन रणनीतियों को लागू करें।",
            "मिट्टी के स्वास्थ्य में सुधार और रोग चक्र को तोड़ने के लिए गैर-अनाज फसलों के साथ फसल चक्र अपनाएं।"

        ]
    },
    "Wheat___Brown_Rust": {
        "symptoms_en": [
            "Orange-brown pustules appear on the leaves, stems, and glumes of infected wheat plants.",
            "Pustules may be scattered or clustered together.",
            "Infected leaves may yellow and die prematurely.",
            "Rust spores are easily spread by wind over long distances.",
            "Brown rust can reduce grain yield and quality, particularly in warm, humid environments."


        ],
        "cures_en": [
            "Plant resistant or moderately resistant wheat varieties.",
            "Apply fungicides as soon as symptoms are detected.",
            "Follow integrated pest management strategies to minimize disease spread.",
            "Destroy volunteer wheat and other susceptible grasses that can harbor the fungus.",
            "Rotate crops to non-host species to break the disease cycle."


        ],
        "symptoms_hi": [
              "संक्रमित गेहूं के पौधों की पत्तियों, तनों और फूलों पर नारंगी-भूरे रंग के फुंसी दिखाई देते हैं।",
            "फुंसी बिखरे हुए या एक साथ गुच्छों में हो सकते हैं।",
            "संक्रमित पत्तियां पीली पड़ सकती हैं और समय से पहले मर सकती हैं।",
            "जंग के बीजाणु हवा द्वारा लंबी दूरी तक आसानी से फैल जाते हैं।",
            "भूरा जंग अनाज की उपज और गुणवत्ता को कम कर सकता है, खासकर गर्म, आर्द्र वातावरण में।"

        ],
        "cures_hi": [
              "प्रतिरोधी या मध्यम प्रतिरोधी गेहूं की किस्में लगाएं।",
            "लक्षणों का पता चलते ही कवकनाशी का प्रयोग करें।",
            "बीमारी के प्रसार को कम करने के लिए एकीकृत कीट प्रबंधन रणनीतियों का पालन करें।",
            "स्वयंसेवी गेहूं और अन्य संवेदनशील घासों को नष्ट करें जो कवक को आश्रय दे सकती हैं।",
            "रोग चक्र को तोड़ने के लिए गैर-मेजबान प्रजातियों के साथ फसल चक्र अपनाएं।"
        ]
    }
}

# Recreate model architectures  (This needs to match your saved models EXACTLY)
def create_model(num_classes):
    model = mod.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential( # Example: Adapt this for *each* model if they differ
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        # nn.Dropout(0.5),  # Include/exclude Dropout as needed to match your saved models
        nn.Linear(256, num_classes)
    )
    return model


# Load models
models = {}
for crop, path in model_paths.items():
    num_classes = len(classes[crop])
    model = create_model(num_classes)  # Ensure create_model matches your saved model architectures
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    models[crop] = model

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/')
# def result():
#     return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."})

    # Save the file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess the image
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Detect crop and disease
    predictions = []
    for crop, model in models.items():
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predictions.append((confidence.item(), crop, classes[crop][predicted.item()]))

    # Select the most confident prediction
    predictions.sort(reverse=True, key=lambda x: x[0])
    final_confidence, final_crop, final_disease = predictions[0]
    

    # Fetch symptoms and cures
    selected_language = request.form.get('language', 'en')  # Get language from form, default to English
    translated_crop = translations_ch[selected_language].get(final_crop, final_crop)
    translated_disease = translations_ch[selected_language].get(final_disease, final_disease)

    disease_info = symptoms_and_cures_ch.get(final_disease, {})
    symptoms_key = f"symptoms_{selected_language}"
    cures_key = f"cures_{selected_language}"

    symptoms = disease_info.get(symptoms_key, ["Unknown"])
    cures = disease_info.get(cures_key, ["Unknown"])
    

    # return render_template(
    #     'result.html',
    #     crop=final_crop,
    #     disease=final_disease,
    #     symptoms=disease_info.get("symptoms", "Unknown"),
    #     cures=disease_info.get("cures", "Unknown"),
    #     confidence=f"{final_confidence:.2f}",
    #     image_path=file_path
    # )
    return render_template(
        'result.html',
        crop=translated_crop,
        disease=translated_disease,
        symptoms=symptoms, # Pass the list of symptoms
        cures=cures, # Pass the list of cures
        confidence=f"{final_confidence:.2f}",
        image_path=file_path,
        selected_language = selected_language # Pass the language for use on result.html
    )

if __name__ == '__main__':
    app.run(debug=True)