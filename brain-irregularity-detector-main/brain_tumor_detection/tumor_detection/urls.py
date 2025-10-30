from django.urls import path
from . import views

urlpatterns = [
    path('',views.upload_image,name='upload_image'),
    path('result/',views.view_result, name='view_result'),
    path('dev.html/',views.view_dev,name='about_devops'),
    
]
