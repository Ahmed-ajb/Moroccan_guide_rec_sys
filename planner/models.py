# planner/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver

class Trip(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trips')
    name = models.CharField(max_length=200, default="Mon Voyage")
    created_at = models.DateTimeField(auto_now_add=True)
    target_cities_input_str = models.TextField(blank=True, null=True)
    total_budget_str = models.CharField(max_length=50, blank=True, null=True)
    num_days_str = models.CharField(max_length=10, blank=True, null=True)
    min_hotel_rating_str = models.CharField(max_length=10, blank=True, null=True)
    activity_preferences_list_str = models.TextField(blank=True, null=True)
    ordered_cities_json = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"'{self.name}' par {self.user.username} ({self.created_at.strftime('%d-%m-%Y')})"

class TripDay(models.Model):
    trip = models.ForeignKey(Trip, on_delete=models.CASCADE, related_name='days')
    day_number = models.PositiveIntegerField()
    city_name = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        ordering = ['trip', 'day_number']
        unique_together = ('trip', 'day_number')

    def __str__(self):
        return f"Jour {self.day_number} ({self.city_name or 'N/A'}) - {self.trip.name}"

class DailyActivityItem(models.Model):
    ITEM_TYPES = [('hotel', 'Hôtel'), ('activity', 'Activité')]
    trip_day = models.ForeignKey(TripDay, on_delete=models.CASCADE, related_name='activity_items')
    order_in_day = models.PositiveIntegerField()
    item_type = models.CharField(max_length=20, choices=ITEM_TYPES)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    price_per_night = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    rating_general = models.FloatField(null=True, blank=True)
    budget_estime = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    activity_type_name = models.CharField(max_length=100, blank=True, null=True)
    duree_estimee = models.CharField(max_length=50, blank=True, null=True)
    astar_segment_to_next_json = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['trip_day', 'order_in_day']

    def __str__(self):
        return f"{self.order_in_day}. {self.name or 'Sans Nom'} (Jour {self.trip_day.day_number})"

class ActivityRating(models.Model):
    RATING_CHOICES = [(i, str(i)) for i in range(1, 6)]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activity_ratings')
    daily_activity_item = models.ForeignKey(DailyActivityItem, on_delete=models.CASCADE, related_name="user_ratings")
    rating = models.PositiveSmallIntegerField(choices=RATING_CHOICES)
    comment = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'daily_activity_item')
        ordering = ['-created_at']

    def __str__(self):
        item_name = self.daily_activity_item.name if self.daily_activity_item else "Item Inconnu"
        return f"Note de {self.user.username} pour {item_name}: {self.rating} étoiles"

# --- NOUVEAU MODÈLE UserProfile ---
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    GENDER_CHOICES = [('H', 'Homme'), ('F', 'Femme'), ('A', 'Autre'), ('N', 'Préfère ne pas spécifier')]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True, null=True, verbose_name="Genre")
    birth_date = models.DateField(null=True, blank=True, verbose_name="Date de naissance")
    city = models.CharField(max_length=100, blank=True, null=True, verbose_name="Ville de résidence")
    profession = models.CharField(max_length=100, blank=True, null=True, verbose_name="Profession")
    bio = models.TextField(blank=True, null=True, verbose_name="Petite biographie (optionnel)")

    def __str__(self):
        return f"Profil de {self.user.username}"

    @property
    def age(self):
        if self.birth_date:
            today = timezone.now().date()
            return today.year - self.birth_date.year - ((today.month, today.day) < (self.birth_date.month, self.birth_date.day))
        return None

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
    try: # Essayer de sauvegarder le profil existant si l'utilisateur est mis à jour
        instance.profile.save()
    except UserProfile.DoesNotExist: # Si le profil n'existe pas pour une raison quelconque (ne devrait pas arriver avec le if created)
         UserProfile.objects.create(user=instance)


# --- NOUVEAU MODÈLE JournalEntry ---
def user_trip_day_media_path(instance, filename):
    # Remplacer les caractères non sûrs dans le nom de fichier
    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in filename)
    return f'user_{instance.user.id}/trip_{instance.trip_day.trip.id}/day_{instance.trip_day.id}/{safe_filename}'

class JournalEntry(models.Model):
    trip_day = models.ForeignKey(TripDay, on_delete=models.CASCADE, related_name='journal_entries')
    user = models.ForeignKey(User, on_delete=models.CASCADE) # Lier à l'utilisateur pour la permission
    title = models.CharField(max_length=200, blank=True, null=True, help_text="Titre optionnel pour ce moment")
    text_content = models.TextField(blank=True, null=True, help_text="Vos notes ou description du moment.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    image = models.ImageField(upload_to=user_trip_day_media_path, blank=True, null=True, verbose_name="Photo souvenir")
    audio = models.FileField(upload_to=user_trip_day_media_path, blank=True, null=True, verbose_name="Enregistrement audio", help_text="Formats supportés: mp3, wav, ogg...")
    
    related_activity_item = models.ForeignKey(DailyActivityItem, on_delete=models.SET_NULL, null=True, blank=True, related_name="journal_moments", verbose_name="Lier à une activité/hôtel planifié")

    class Meta:
        ordering = ['trip_day', 'created_at']
        verbose_name = "Entrée de Journal"
        verbose_name_plural = "Entrées de Journal"

    def __str__(self):
        base = f"Entrée pour Jour {self.trip_day.day_number} du voyage '{self.trip_day.trip.name}'"
        if self.title:
            return f"{self.title} - {base}"
        return base