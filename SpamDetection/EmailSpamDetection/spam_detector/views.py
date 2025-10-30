import joblib
import os
from django.shortcuts import render
from django.http import JsonResponse

# Load Model and Vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Custom Spam Rule Function
def custom_spam_rule(text):
    spam_keywords = ['join my group', 'earn money', 'click here', 'make money', 'free gift', 'limited offer']
    for keyword in spam_keywords:
        if keyword in text.lower():
            return 1  # Flag as Spam
    return 0  # Flag as Ham

def index(request):
    return render(request, 'index.html')

def classify_email(request):
    if request.method == 'POST':
        # Get email text from the form
        email_text = request.POST.get('email')
        print(f"Received email text: {email_text}")

        if email_text:
            # Apply custom spam rule before vectorizing
            if custom_spam_rule(email_text):
                result = 'Spam'
            else:
                # Vectorize and predict using model
                input_data = vectorizer.transform([email_text])
                prediction = model.predict(input_data)

                result = 'Spam' if prediction[0] == 1 else 'Ham'

            print(f"Prediction: {result}")
            return render(request, 'results.html', {'result': result, 'email': email_text})
        else:
            return render(request, 'index.html', {'error': 'Please enter some email text.'})

    return render(request, 'index.html')
