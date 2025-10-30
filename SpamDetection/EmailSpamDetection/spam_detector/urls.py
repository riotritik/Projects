# EmailSpamDetection/spam_detector/urls.py
from django.urls import path
from . import views

# Do NOT set app_name here (keep names global to match templates using {% url 'classify_email' %})
urlpatterns = [
    path('', views.index, name='home'),                    # homepage
    path('classify_email/', views.classify_email, name='classify_email'),
]
