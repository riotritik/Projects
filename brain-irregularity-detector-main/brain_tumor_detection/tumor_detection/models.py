from django.db import models

# Create your models here.
class BrainTumorDetector(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    result=models.CharField(max_length=255, blank=True)
    
    def __str__(self):
        return self.image.name