from django import forms
from .models import BrainTumorDetector

class UploadForm(forms.ModelForm):
    class Meta:
        model = BrainTumorDetector
        fields = ('image',)