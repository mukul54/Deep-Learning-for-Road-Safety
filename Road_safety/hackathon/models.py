from django.db import models

# Create your models here.

class Driver(models.Model):
    car_no = models.CharField(max_length=100, unique=True)
    lat = models.FloatField(default=0)
    lon = models.FloatField(default=0)
    taken = models.BooleanField(default=False)
    detected = models.BooleanField(default=False)
    
    def __str__(self):
        return self.car_no

    
class Police(models.Model):
    name = models.CharField(max_length=50)
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=50)
    confirm_password = models.CharField(max_length=50)
    availabality = models.BooleanField(default=True)
    authent = models.BooleanField(default=False)
    detected_car = models.ManyToManyField(Driver, related_name='choice_set', blank=True)
    captured_image = models.ImageField(blank=True)
    lat = models.FloatField(default=0)
    lon = models.FloatField(default=0)
    
    def __str__(self):
        return str(self.name) + ' (' + str(self.username) + ')'
    