from django.urls import path
from . import views

app_name='hackathon'

urlpatterns=[
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
]