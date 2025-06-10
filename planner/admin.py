# planner/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import Trip, TripDay, DailyActivityItem, ActivityRating, UserProfile, JournalEntry

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profils Utilisateur'
    fk_name = 'user'
    fields = ('gender', 'birth_date', 'city', 'profession', 'bio') # Spécifier les champs

class CustomUserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'get_birth_date_profile')
    list_select_related = ('profile',)

    def get_birth_date_profile(self, instance):
        try:
            return instance.profile.birth_date
        except UserProfile.DoesNotExist:
            return None
    get_birth_date_profile.short_description = 'Date de Naissance (Profil)'

    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()
        return super().get_inline_instances(request, obj)

admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

@admin.register(Trip)
class TripAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'created_at', 'target_cities_input_str')
    list_filter = ('user', 'created_at')
    search_fields = ('name', 'user__username', 'target_cities_input_str')

@admin.register(TripDay)
class TripDayAdmin(admin.ModelAdmin):
    list_display = ('trip', 'day_number', 'city_name')
    list_filter = ('trip__user', 'city_name')
    search_fields = ('trip__name', 'city_name')

@admin.register(DailyActivityItem)
class DailyActivityItemAdmin(admin.ModelAdmin):
    list_display = ('name', 'item_type', 'trip_day', 'order_in_day')
    list_filter = ('item_type', 'trip_day__trip__user', 'trip_day__city_name')
    search_fields = ('name', 'description')

@admin.register(ActivityRating)
class ActivityRatingAdmin(admin.ModelAdmin):
    list_display = ('daily_activity_item_name', 'user', 'rating', 'created_at')
    list_filter = ('user', 'rating', 'created_at')
    search_fields = ('daily_activity_item__name', 'user__username', 'comment')

    def daily_activity_item_name(self, obj):
        return obj.daily_activity_item.name
    daily_activity_item_name.short_description = 'Élément Noté'

@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    list_display = ('title_or_default', 'trip_day_info', 'user', 'created_at', 'has_image', 'has_audio')
    list_filter = ('user', 'trip_day__trip__name', 'created_at')
    search_fields = ('title', 'text_content', 'user__username', 'trip_day__trip__name')
    readonly_fields = ('created_at', 'updated_at')

    def trip_day_info(self, obj):
        return f"Jour {obj.trip_day.day_number} - {obj.trip_day.trip.name}"
    trip_day_info.short_description = "Jour du Voyage"

    def title_or_default(self, obj):
        return obj.title if obj.title else f"Entrée du {obj.created_at.strftime('%d/%m/%Y')}"
    title_or_default.short_description = "Titre / Date"
    
    def has_image(self, obj):
        return bool(obj.image)
    has_image.boolean = True
    has_image.short_description = "Image ?"

    def has_audio(self, obj):
        return bool(obj.audio)
    has_audio.boolean = True
    has_audio.short_description = "Audio ?"