from django.urls import path
from django.views.generic.base import RedirectView
from . import views

urlpatterns = [
    path('video_feed/', views.video_feed, name='video_feed'),
    path('', RedirectView.as_view(url='video_feed/', permanent=True)),
]
