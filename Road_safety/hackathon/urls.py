from django.urls import path
from . import views

app_name='hackathon'

urlpatterns=[
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('home/', views.PoliceHome, name='PoliceHome'),
    path('logout/', views.logout, name='logout'),
    path('record/', views.record, name='record'),
    path('dumb/', views.dumb, name='dumb'),
    path('take_driver/', views.take_driver, name='take_driver'),
    path('mainserver/', views.mainserver, name='mainserver'),
]