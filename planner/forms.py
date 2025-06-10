# planner/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .utils import load_and_preprocess_data, OSMNX_AVAILABLE # Assurez-vous que utils.py est fonctionnel
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Logique de chargement des CHOICES (identique, mais avec plus de logging/robustesse)
ALL_AVAILABLE_CITIES_CHOICES = [('loading', 'Chargement des villes...')]
ACTIVITY_TYPE_CHOICES = [('loading', 'Chargement des types...')]
try:
    activities_df_form, _, _ = load_and_preprocess_data()
    if activities_df_form is not None and not activities_df_form.empty:
        if "ville_normalisee" in activities_df_form.columns:
            unique_sorted_cities_form = sorted(list(set(c for c in activities_df_form["ville_normalisee"].unique() if pd.notna(c))))
            if unique_sorted_cities_form:
                ALL_AVAILABLE_CITIES_CHOICES = [(city, city) for city in unique_sorted_cities_form]
            else:
                logger.warning("Aucune ville unique trouvée après traitement pour les formulaires.")
                ALL_AVAILABLE_CITIES_CHOICES = [('', 'Aucune ville disponible')]
        else:
            logger.warning("Colonne 'ville_normalisee' manquante dans activities_df_form pour forms.py.")
            ALL_AVAILABLE_CITIES_CHOICES = [('', 'Erreur chargement villes (col manquante)')]

        if "type" in activities_df_form.columns:
            activity_type_options_form = sorted(list(set(t for t in activities_df_form["type"].unique() if pd.notna(t))))
            if activity_type_options_form:
                ACTIVITY_TYPE_CHOICES = [(atype, atype) for atype in activity_type_options_form]
            else:
                logger.warning("Aucun type d'activité unique trouvé après traitement pour les formulaires.")
                ACTIVITY_TYPE_CHOICES = [('', 'Aucun type disponible')]
        else:
            logger.warning("Colonne 'type' manquante dans activities_df_form pour forms.py.")
            ACTIVITY_TYPE_CHOICES = [('', 'Erreur chargement types (col manquante)')]
    else:
        logger.warning("activities_df_form est vide ou None dans forms.py. Les choix de formulaire seront limités.")
        ALL_AVAILABLE_CITIES_CHOICES = [('', 'Données indisponibles (villes)')]
        ACTIVITY_TYPE_CHOICES = [('', 'Données indisponibles (types)')]
except Exception as e:
    logger.error(f"ERREUR critique lors du chargement des données pour les choix de formulaires (forms.py): {e}", exc_info=True)
    ALL_AVAILABLE_CITIES_CHOICES = [('error', 'Erreur chargement')]
    ACTIVITY_TYPE_CHOICES = [('error', 'Erreur chargement')]


class TripPlannerForm(forms.Form):
    target_cities = forms.MultipleChoiceField(label="Villes à visiter ?", choices=ALL_AVAILABLE_CITIES_CHOICES, widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}), required=True, error_messages={'required': "Veuillez sélectionner au moins une ville."})
    num_days = forms.IntegerField(label="Nombre de jours ?", min_value=1, initial=1, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    total_budget = forms.FloatField(label="Budget total (MAD) ?", min_value=100.0, initial=1000.0, widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '100.0'}))
    min_hotel_rating = forms.FloatField(label="Rating hôtel min ?", min_value=0.0, max_value=10.0, initial=7.5, widget=forms.NumberInput(attrs={'class': 'form-control', 'step': "0.1"}))
    activity_prefs = forms.MultipleChoiceField(label="Préférences type d'activité ?", choices=ACTIVITY_TYPE_CHOICES, widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}), required=False)
    use_astar_routes_planning = forms.BooleanField(label="Utiliser A* pour itinéraires journaliers", help_text="(Plus lent, nécessite OSMnx)", required=False, initial=True, widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-assigner les choices car ils pourraient avoir été mis à jour si load_and_preprocess_data a un cache
        self.fields['target_cities'].choices = ALL_AVAILABLE_CITIES_CHOICES
        self.fields['activity_prefs'].choices = ACTIVITY_TYPE_CHOICES
        
        if not OSMNX_AVAILABLE:
            self.fields['use_astar_routes_planning'].widget.attrs['disabled'] = True
            self.fields['use_astar_routes_planning'].help_text = "OSMnx non détecté. Option désactivée."
            self.fields['use_astar_routes_planning'].initial = False# planner/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, JournalEntry, DailyActivityItem # AJOUTER JournalEntry, DailyActivityItem
from .utils import load_and_preprocess_data, OSMNX_AVAILABLE
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ... (Logique de chargement des CHOICES pour TripPlannerForm - INCHANGÉE) ...
# ... (Classe TripPlannerForm - INCHANGÉE) ...
# ... (Classe SignUpForm - INCHANGÉE, mais s'assure qu'elle sauvegarde bien le profil) ...
# ... (Classe RatingForm - INCHANGÉE) ...

# --- NOUVEAU FORMULAIRE POUR LES ENTRÉES DE JOURNAL ---
class JournalEntryForm(forms.ModelForm):
    class Meta:
        model = JournalEntry
        fields = ['title', 'text_content', 'image', 'audio', 'related_activity_item']
        widgets = {
            'text_content': forms.Textarea(attrs={'rows': 5, 'placeholder': 'Décrivez votre moment, vos impressions, anecdotes...'}),
            'image': forms.ClearableFileInput(attrs={'accept': 'image/*'}),
            'audio': forms.ClearableFileInput(attrs={'accept': 'audio/*'}),
        }
        labels = {
            'title': "Titre du moment (optionnel)",
            'text_content': "Vos notes et souvenirs",
            'image': "Ajouter une photo (optionnel)",
            'audio': "Ajouter un enregistrement audio (optionnel)",
            'related_activity_item': "Lier à une activité/hôtel du plan (optionnel)"
        }
        help_texts = { 'audio': "Formats courants : MP3, WAV, OGG." }

    def __init__(self, *args, **kwargs):
        trip_day_obj = kwargs.pop('trip_day', None)
        super().__init__(*args, **kwargs)
        if trip_day_obj:
            self.fields['related_activity_item'].queryset = DailyActivityItem.objects.filter(trip_day=trip_day_obj)
            self.fields['related_activity_item'].empty_label = "--- Non lié à un item spécifique ---"
            self.fields['related_activity_item'].label_from_instance = lambda obj: f"{obj.order_in_day}. {obj.name}"
        else:
            self.fields['related_activity_item'].queryset = DailyActivityItem.objects.none()
            if self.is_bound and 'use_astar_routes_planning' in self.data:
                mutable_data = self.data.copy(); mutable_data['use_astar_routes_planning'] = False; self.data = mutable_data

class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, required=True, help_text='Requis. Entrez une adresse email valide.')
    first_name = forms.CharField(max_length=30, required=False, label="Prénom")
    last_name = forms.CharField(max_length=150, required=False, label="Nom de famille")
    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('first_name', 'last_name', 'email',)

class RatingForm(forms.Form):
    RATING_CHOICES = [(i, str(i)) for i in range(1, 6)]
    rating = forms.ChoiceField(choices=RATING_CHOICES, widget=forms.RadioSelect(attrs={'class': 'form-check-input-rating'}), label="Votre note", required=True, error_messages={'required': "Veuillez sélectionner une note."})
    comment = forms.CharField(widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control', 'placeholder': 'Laissez un commentaire...'}), required=False, label="Votre commentaire (optionnel)")