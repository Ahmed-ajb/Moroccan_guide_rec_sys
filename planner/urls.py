# planner/urls.py
from django.urls import path
from . import views # Importer les vues de l'application planner

app_name = 'planner' # Définit un namespace pour ces URLs

urlpatterns = [
    # Page d'accueil / Vitrine
    path('', views.home_showcase_view, name='home_showcase'),

    # Page de planification principale
    path('plan/', views.plan_trip_view, name='plan_trip'),

    # Téléchargement du PDF du plan généré (stocké en session)
    path('download-pdf/', views.download_pdf_view, name='download_pdf'),

    # Authentification (inscription personnalisée)
    path('signup/', views.signup_view, name='signup'),
    # Les URLs de connexion, déconnexion, changement de MDP etc., sont gérées par 'django.contrib.auth.urls'
    # inclus dans le urls.py principal du projet.

    # Voyages de l'utilisateur
    path('my-trips/', views.my_trips_view, name='my_trips'),
    path('trip/<int:trip_id>/', views.trip_detail_view, name='trip_detail'),

    # Notation des éléments d'un voyage
    path('rate-item/<int:item_id>/', views.rate_item_view, name='rate_item'),

    # Journal de voyage (Moments)
    path('trip-day/<int:trip_day_id>/add-journal/', views.add_journal_entry_view, name='add_journal_entry'),
    path('journal-entry/<int:entry_id>/edit/', views.edit_journal_entry_view, name='edit_journal_entry'),
    # Vous pourriez aussi ajouter une vue pour supprimer une entrée de journal :
    # path('journal-entry/<int:entry_id>/delete/', views.delete_journal_entry_view, name='delete_journal_entry'),

    # Rapport de voyage
    path('trip/<int:trip_id>/report/', views.trip_report_view, name='trip_report'),
]