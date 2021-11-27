from django.db import models
from django.db.models.fields import TextField
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse
import os


class Post(models.Model):
    # title = models.CharField(max_length=100)
    # file = models.FileField(null=True,blank=True,upload_to='Files')
    # content = models.TextField()
    # date_posted = models.DateTimeField(default=timezone.now)
    # author = models.ForeignKey(User, on_delete=models.CASCADE)

    Accented_Sentence = TextField()
    Accentless_Sentence = TextField()
    Date = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    fb_choices = (
        ("unsatisfied", "unsatisfied"),
        ("neutral", "neutral"),
        ("satisfied", "satisfied"),
    )
    Leave_Feedback = models.CharField(
        max_length=100, choices=fb_choices, null=True, blank=True)

    def __str__(self):
        return self.Accented_Sentence

    def extension(self):
        name, extension = os.path.splitext(self.file.name)
        return extension

    def get_absolute_url(self):
        return reverse('post-detail', kwargs={'pk': self.pk})
