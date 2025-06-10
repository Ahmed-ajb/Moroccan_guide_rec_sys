# planner/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, Http404
from io import BytesIO
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.urls import reverse
from django.utils import timezone
import pandas as pd
import logging
import traceback

# Imports depuis vos autres fichiers d'application
from .forms import TripPlannerForm, SignUpForm, RatingForm, JournalEntryForm
from .models import Trip, TripDay, DailyActivityItem, ActivityRating, JournalEntry, UserProfile
from .utils import (
    load_and_preprocess_data,
    plan_trip_django,
    OSMNX_AVAILABLE
)
from .folium_utils import generate_trip_map_folium_django
from .reportlab_utils import generate_schedule_content_objects_django, generate_trip_pdf_django

logger = logging.getLogger(__name__)

# --- VUE VITRINE ---
def home_showcase_view(request):
    logger.debug("--- Entrée dans home_showcase_view ---")
    showcase_activities_by_city = {}
    activities_df = None
    
    try:
        activities_df, _, _ = load_and_preprocess_data()
        if activities_df is None:
            logger.error("load_and_preprocess_data a retourné None pour activities_df dans home_showcase_view.")
            activities_df = pd.DataFrame()
        elif not isinstance(activities_df, pd.DataFrame):
            logger.error(f"load_and_preprocess_data a retourné un type inattendu pour activities_df: {type(activities_df)} dans home_showcase_view.")
            activities_df = pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Erreur critique lors du chargement des données pour home_showcase_view: {e}", exc_info=True)
        messages.error(request, "Impossible de charger les données des activités pour la vitrine due à une erreur système.")
        activities_df = pd.DataFrame()

    if not activities_df.empty:
        logger.debug(f"Activités chargées pour la vitrine, nombre de lignes: {len(activities_df)}")
        required_cols = ['ville_normalisee', 'nom', 'description', 'type']
        
        df_for_showcase = activities_df.copy()
        
        if 'image_url' not in df_for_showcase.columns:
            df_for_showcase['image_url'] = 'https://via.placeholder.com/350x200.png?text=Activit%C3%A9'
            logger.debug("Colonne 'image_url' ajoutée avec placeholder à df_for_showcase.")
        else:
            df_for_showcase['image_url'] = df_for_showcase['image_url'].fillna('https://via.placeholder.com/350x200.png?text=Activit%C3%A9')
        
        for col in required_cols:
            if col not in df_for_showcase.columns:
                df_for_showcase[col] = "Donnée manquante"
                logger.warning(f"Colonne '{col}' manquante pour la vitrine dans df_for_showcase, utilisation de 'Donnée manquante'.")
        
        try:
            if 'ville_normalisee' in df_for_showcase.columns and 'nom' in df_for_showcase.columns :
                grouped = df_for_showcase.groupby('ville_normalisee')
                for city, group in grouped:
                    cols_for_dict = required_cols + ['image_url']
                    existing_cols_for_dict = [c for c in cols_for_dict if c in group.columns]
                    
                    if not group.empty:
                        unique_activities = group.drop_duplicates(subset=['nom']).head(3)
                        if not unique_activities.empty:
                            showcase_activities_by_city[city] = unique_activities[existing_cols_for_dict].to_dict('records')
                        else: logger.debug(f"Aucune activité unique trouvée pour la ville {city} après drop_duplicates/head.")
                    else: logger.debug(f"Groupe vide pour la ville {city} dans la vitrine.")
            else:
                 logger.warning("Colonnes 'ville_normalisee' ou 'nom' manquantes pour grouper les activités de la vitrine.")
        except Exception as e_showcase_prep:
            logger.error(f"Erreur lors de la préparation des données de vitrine: {e_showcase_prep}", exc_info=True)
            messages.error(request, "Une erreur s'est produite lors de la préparation des activités à afficher.")
    else:
        logger.info("Aucune donnée d'activité (activities_df vide ou None) pour la vitrine après le chargement.")
        if not list(messages.get_messages(request)):
             messages.info(request, "Aucune activité à afficher pour le moment sur la page d'accueil.")

    context = {
        'showcase_activities_by_city': showcase_activities_by_city,
        'page_title': "Découvrez le Maroc avec Nous !",
    }
    logger.debug(f"Contexte pour home_showcase.html: {len(showcase_activities_by_city)} villes avec activités.")
    return render(request, 'planner/home_showcase.html', context)

# --- VUE D'INSCRIPTION ---
def signup_view(request):
    if request.user.is_authenticated:
        return redirect(reverse('planner:home_showcase'))
    
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save() # La méthode save() customisée dans SignUpForm s'occupe du profil
            login(request, user)
            messages.success(request, "Inscription réussie ! Vous êtes maintenant connecté.")
            return redirect(reverse('planner:home_showcase'))
        else:
            logger.warning(f"Formulaire d'inscription invalide: {form.errors.as_json(escape_html=True)}")
            # Les erreurs de formulaire sont gérées par le template avec {{ form.errors }} ou {{ field.errors }}
            # On peut ajouter un message global si on le souhaite
            messages.error(request, "Le formulaire d'inscription contient des erreurs. Veuillez vérifier les champs.")
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form, 'page_title': "Créer un Compte"})

# --- VUE DE PLANIFICATION ---
def plan_trip_view(request):
    form = TripPlannerForm() 
    trip_plan_result, trip_params_for_pdf, ordered_cities_for_map, folium_map_html, schedule_md = None, {}, [], None, ""
    num_days_from_form = 0

    try:
        activities_df, hotels_df, city_coords_map = load_and_preprocess_data()
        if activities_df is None: activities_df = pd.DataFrame() 
        if hotels_df is None: hotels_df = pd.DataFrame()
        if city_coords_map is None: city_coords_map = {}
        
        if activities_df.empty:
             logger.warning("plan_trip_view: Données d'activités vides ou non chargées après load_and_preprocess_data.")
    except Exception as e:
        logger.critical(f"Erreur majeure lors du chargement initial des données pour plan_trip_view: {e}", exc_info=True)
        messages.error(request, f"Erreur système majeure lors du chargement des données. Planification impossible pour le moment.")
        activities_df, hotels_df, city_coords_map = pd.DataFrame(), pd.DataFrame(), {}

    if request.method == 'POST':
        form = TripPlannerForm(request.POST) 
        logger.debug(f"Formulaire de planification soumis. Données POST: {request.POST}")
        if form.is_valid():
            logger.info("Formulaire de planification est VALIDE.")
            try:
                target_cities = form.cleaned_data['target_cities']
                num_days = form.cleaned_data['num_days']
                num_days_from_form = num_days
                total_budget = form.cleaned_data['total_budget']
                min_hotel_rating = form.cleaned_data['min_hotel_rating']
                activity_prefs_list = form.cleaned_data['activity_prefs']
                use_astar = form.cleaned_data.get('use_astar_routes_planning', False) and OSMNX_AVAILABLE

                if activities_df.empty and (hotels_df.empty if hotels_df is not None else True) : 
                    messages.error(request, "Les données de base (activités et hôtels) sont manquantes. Impossible de planifier.")
                else:
                    logger.debug(f"Appel de plan_trip_django avec: villes={target_cities}, budget={total_budget}, jours={num_days}")
                    trip_plan_result, trip_params_for_pdf, ordered_cities_for_map = plan_trip_django(
                        target_cities_list=target_cities, total_budget_str=str(total_budget),
                        num_days_str=str(num_days), min_hotel_rating_str=str(min_hotel_rating),
                        activity_preferences_str=", ".join(activity_prefs_list),
                        activities_df_global=activities_df, hotels_df_global=hotels_df,
                        city_coords_map_global=city_coords_map, use_astar_for_planning=use_astar
                    )
                    logger.debug(f"Retour de plan_trip_django. trip_plan_result est None: {trip_plan_result is None}. Params PDF: {trip_params_for_pdf}")

                    if trip_plan_result:
                        messages.success(request, "🎉 Votre plan de voyage personnalisé est prêt !")
                        should_map_use_astar = use_astar and OSMNX_AVAILABLE
                        folium_map_obj = generate_trip_map_folium_django(
                            ordered_cities_for_map, trip_plan_result, city_coords_map,
                            num_days, use_astar_on_map=should_map_use_astar, osmnx_available_flag=OSMNX_AVAILABLE
                        )
                        if folium_map_obj: folium_map_html = folium_map_obj._repr_html_()
                        schedule_md, _ = generate_schedule_content_objects_django(trip_plan_result, num_days)

                        if request.user.is_authenticated:
                            try:
                                trip_name_parts = [city_plan_item['ville'] for city_plan_item in trip_plan_result[:2] if 'ville' in city_plan_item and city_plan_item['ville']]
                                trip_default_name = f"Voyage à {', '.join(trip_name_parts)}" if trip_name_parts else "Mon Nouveau Voyage"
                                if len(trip_plan_result) > 2 and trip_name_parts : trip_default_name += "..."
                                trip_default_name += f" ({timezone.now().strftime('%d/%m/%y')})"

                                logger.debug(f"Début sauvegarde pour: {trip_default_name}")
                                new_trip = Trip.objects.create(
                                    user=request.user, name=trip_default_name,
                                    target_cities_input_str=", ".join(target_cities),
                                    total_budget_str=str(total_budget), num_days_str=str(num_days),
                                    min_hotel_rating_str=str(min_hotel_rating),
                                    activity_preferences_list_str=", ".join(activity_prefs_list),
                                    ordered_cities_json=ordered_cities_for_map if ordered_cities_for_map else []
                                )
                                logger.info(f"Trip '{new_trip.name}' (ID: {new_trip.id}) créé pour {request.user.username}")

                                day_counter_in_trip = 0
                                for city_plan_item in trip_plan_result: 
                                    daily_opt_act_lists = city_plan_item.get('activites_par_jour_optimisees', [])
                                    daily_astar_seg_lists = city_plan_item.get('itineraire_segments_par_jour', [])
                                    
                                    for day_idx_city, daily_points_list in enumerate(daily_opt_act_lists): 
                                        if not daily_points_list: 
                                            logger.warning(f"Jour {day_counter_in_trip + 1} pour {city_plan_item.get('ville')} n'a pas de points. TripDay non créé pour ce jour vide.")
                                            continue 
                                        
                                        day_counter_in_trip += 1
                                        trip_day_obj = TripDay.objects.create(
                                            trip=new_trip, 
                                            day_number=day_counter_in_trip, 
                                            city_name=city_plan_item.get('ville', "Ville Inconnue")
                                        )
                                        
                                        astar_segs_for_this_day = daily_astar_seg_lists[day_idx_city] if day_idx_city < len(daily_astar_seg_lists) else []
                                        
                                        for order_idx, point_dict in enumerate(daily_points_list): 
                                            is_hotel = point_dict.get('type') == 'hotel' or point_dict.get('price_per_night') is not None
                                            seg_next = astar_segs_for_this_day[order_idx] if order_idx < len(astar_segs_for_this_day) else None
                                            
                                            item_name = point_dict.get('nom')
                                            if not item_name or pd.isna(item_name):
                                                item_name = f"{'Hôtel Etape' if is_hotel else 'Activité Etape'} #{order_idx + 1} (Nom Auto)"
                                                logger.warning(f"Point sans nom valide lors de la sauvegarde. Nom par défaut: '{item_name}'. Données du point: {point_dict}")

                                            daily_item_data_to_create = {
                                                'trip_day': trip_day_obj, 'order_in_day': order_idx + 1,
                                                'item_type': 'hotel' if is_hotel else 'activity', 'name': item_name,
                                                'description': point_dict.get('description'),
                                                'latitude': point_dict.get('latitude'), 'longitude': point_dict.get('longitude'),
                                                'price_per_night': point_dict.get('price_per_night') if is_hotel else None,
                                                'rating_general': point_dict.get('rating'),
                                                'budget_estime': point_dict.get('budget_estime') if not is_hotel else None,
                                                'activity_type_name': point_dict.get('type') if not is_hotel else None,
                                                'duree_estimee': point_dict.get('duree_estimee'),
                                                'astar_segment_to_next_json': seg_next if seg_next else None
                                            }
                                            logger.debug(f"Données pour DailyActivityItem.create: {daily_item_data_to_create}")
                                            DailyActivityItem.objects.create(**daily_item_data_to_create)
                                        logger.debug(f"{len(daily_points_list)} DailyActivityItems créés pour TripDay ID {trip_day_obj.id}")
                                messages.info(request, f"Voyage '{new_trip.name}' sauvegardé avec succès dans 'Mes Voyages'.")
                            except Exception as e_save:
                                logger.error(f"ERREUR DÉTAILLÉE lors de la sauvegarde du voyage pour {request.user.username}: {e_save}", exc_info=True)
                                messages.error(request, f"Une erreur technique est survenue lors de la sauvegarde de votre voyage: {e_save}")
                        else:
                            messages.info(request, "Connectez-vous ou inscrivez-vous pour sauvegarder ce voyage !")

                        request.session['trip_plan_for_pdf'] = trip_plan_result
                        request.session['trip_params_for_pdf'] = trip_params_for_pdf
                        request.session['num_days_for_pdf'] = num_days_from_form
                        request.session.modified = True
                    elif trip_params_for_pdf and "error" in trip_params_for_pdf:
                        messages.error(request, f"Erreur de planification : {trip_params_for_pdf['error']}")
                    else:
                        messages.warning(request, "La planification n'a pas retourné de résultats. Essayez d'ajuster vos critères.")
            except Exception as e:
                logger.error(f"Erreur majeure dans la soumission POST de plan_trip_view: {e}", exc_info=True)
                messages.error(request, f"Une erreur système majeure est survenue: {e}")
        else:
            logger.warning(f"Formulaire de planification soumis mais invalide: {form.errors.as_json(escape_html=True)}")
            messages.error(request, "Formulaire invalide. Veuillez corriger les erreurs indiquées.")
    
    context = {'form': form, 'trip_plan_result': trip_plan_result, 'trip_params': trip_params_for_pdf,
               'ordered_cities': ordered_cities_for_map, 'folium_map_html': folium_map_html,
               'schedule_md': schedule_md, 'OSMNX_AVAILABLE': OSMNX_AVAILABLE}
    return render(request, 'planner/plan_trip.html', context)

# --- VUE MES VOYAGES ---
@login_required
def my_trips_view(request):
    user_trips = Trip.objects.filter(user=request.user).order_by('-created_at')
    context = {'user_trips': user_trips, 'page_title': "Mes Voyages Sauvegardés"}
    return render(request, 'planner/my_trips.html', context)

# --- VUE DÉTAIL VOYAGE ---
@login_required
def trip_detail_view(request, trip_id):
    trip = get_object_or_404(Trip, pk=trip_id, user=request.user)
    days_with_items_and_journal = [] 

    for day_obj in trip.days.all().order_by('day_number'):
        items_for_day = []
        for item_db in day_obj.activity_items.all().order_by('order_in_day'):
            user_rating_obj = ActivityRating.objects.filter(user=request.user, daily_activity_item=item_db).first()
            items_for_day.append({
                'db_item': item_db,
                'user_has_rated': user_rating_obj is not None,
                'user_rating': user_rating_obj.rating if user_rating_obj else None
            })
        
        journal_entries_for_this_day = day_obj.journal_entries.filter(user=request.user).order_by('created_at')
        
        days_with_items_and_journal.append({
            'day_obj': day_obj,
            'items': items_for_day,
            'journal_entries': journal_entries_for_this_day
        })

    context = {
        'trip': trip,
        'days_with_items_and_journal': days_with_items_and_journal,
        'page_title': f"Détails du Voyage: {trip.name}",
    }
    return render(request, 'planner/trip_detail.html', context)

# --- VUES POUR LE JOURNAL DE VOYAGE ---
@login_required
def add_journal_entry_view(request, trip_day_id):
    trip_day = get_object_or_404(TripDay, pk=trip_day_id, trip__user=request.user)
    
    if request.method == 'POST':
        form = JournalEntryForm(request.POST, request.FILES, trip_day=trip_day)
        if form.is_valid():
            entry = form.save(commit=False)
            entry.trip_day = trip_day
            entry.user = request.user
            entry.save()
            messages.success(request, "Nouveau moment ajouté à votre journal de voyage !")
            return redirect(reverse('planner:trip_detail', kwargs={'trip_id': trip_day.trip.id}) + f"#day-{trip_day.day_number}")
        else:
            messages.error(request, "Veuillez corriger les erreurs dans le formulaire du moment.")
    else:
        form = JournalEntryForm(trip_day=trip_day)

    context = {
        'form': form,
        'trip_day': trip_day,
        'page_title': f"Ajouter un Moment - Jour {trip_day.day_number} ({trip_day.city_name or 'N/A'})"
    }
    return render(request, 'planner/add_edit_journal_entry.html', context)

@login_required
def edit_journal_entry_view(request, entry_id):
    entry = get_object_or_404(JournalEntry, pk=entry_id, user=request.user)
    trip_day = entry.trip_day

    if request.method == 'POST':
        form = JournalEntryForm(request.POST, request.FILES, instance=entry, trip_day=trip_day)
        if form.is_valid():
            form.save()
            messages.success(request, "Moment mis à jour avec succès !")
            return redirect(reverse('planner:trip_detail', kwargs={'trip_id': trip_day.trip.id}) + f"#day-{trip_day.day_number}")
        else:
            messages.error(request, "Veuillez corriger les erreurs lors de la modification du moment.")
    else:
        form = JournalEntryForm(instance=entry, trip_day=trip_day)
    
    context = {
        'form': form,
        'entry': entry, 
        'trip_day': trip_day,
        'page_title': f"Modifier Moment: {entry.title or 'Entrée sans titre'}"
    }
    return render(request, 'planner/add_edit_journal_entry.html', context)

# --- VUE NOTATION ITEM ---
@login_required
def rate_item_view(request, item_id):
    item_to_rate = get_object_or_404(DailyActivityItem, pk=item_id)
    if item_to_rate.trip_day.trip.user != request.user:
        messages.error(request, "Action non autorisée.")
        return redirect(reverse('planner:home_showcase'))

    existing_rating = ActivityRating.objects.filter(user=request.user, daily_activity_item=item_to_rate).first()
    if request.method == 'POST':
        form = RatingForm(request.POST)
        if form.is_valid():
            if existing_rating:
                existing_rating.rating = form.cleaned_data['rating']
                existing_rating.comment = form.cleaned_data['comment']
                existing_rating.save()
                messages.success(request, "Note mise à jour avec succès.")
            else:
                ActivityRating.objects.create(
                    user=request.user,
                    daily_activity_item=item_to_rate,
                    rating=form.cleaned_data['rating'],
                    comment=form.cleaned_data['comment']
                )
                messages.success(request, "Merci pour votre note !")
            return redirect(reverse('planner:trip_detail', kwargs={'trip_id': item_to_rate.trip_day.trip.id}))
        else:
            logger.warning(f"Formulaire de notation invalide: {form.errors.as_json(escape_html=True)}")
            messages.error(request, "Le formulaire de notation contient des erreurs.")
    else:
        initial_data = {'rating': existing_rating.rating, 'comment': existing_rating.comment} if existing_rating else {}
        form = RatingForm(initial=initial_data)
    
    context = {'form': form, 'item_to_rate': item_to_rate, 'page_title': f"Noter: {item_to_rate.name}"}
    return render(request, 'planner/rate_item.html', context)

# --- VUE RAPPORT DE VOYAGE ---
@login_required
def trip_report_view(request, trip_id):
    trip = get_object_or_404(Trip, pk=trip_id, user=request.user)
    
    days_report_data = []
    for day_obj in trip.days.all().order_by('day_number'):
        planned_items_data = []
        for item_db in day_obj.activity_items.all().order_by('order_in_day'):
            user_rating = ActivityRating.objects.filter(user=request.user, daily_activity_item=item_db).first()
            planned_items_data.append({
                'item': item_db,
                'user_rating': user_rating.rating if user_rating else None,
                'user_comment': user_rating.comment if user_rating else None,
            })
        
        journal_entries_for_day = day_obj.journal_entries.filter(user=request.user).order_by('created_at')
        
        days_report_data.append({
            'day_obj': day_obj,
            'planned_items': planned_items_data,
            'journal_entries': journal_entries_for_day
        })

    context = {
        'trip': trip,
        'days_report_data': days_report_data,
        'page_title': f"Rapport du Voyage: {trip.name}"
    }
    return render(request, 'planner/trip_report.html', context)

# --- VUE TÉLÉCHARGEMENT PDF ---
def download_pdf_view(request):
    trip_plan_result = request.session.get('trip_plan_for_pdf')
    trip_params = request.session.get('trip_params_for_pdf')
    num_days_for_pdf = request.session.get('num_days_for_pdf')

    if not trip_plan_result or not trip_params or num_days_for_pdf is None:
        messages.error(request, "Les données du plan de voyage pour le PDF sont manquantes ou ont expiré. Veuillez re-planifier un voyage.")
        return redirect(reverse('planner:plan_trip')) 

    try:
        _, schedule_pdf_obj_list_for_download = generate_schedule_content_objects_django(trip_plan_result, num_days_for_pdf)
    except Exception as e_schedule:
        logger.error(f"Erreur lors de la génération des objets d'emploi du temps pour le PDF: {e_schedule}", exc_info=True)
        messages.error(request, "Une erreur est survenue lors de la préparation du contenu de l'emploi du temps pour le PDF.")
        return redirect(reverse('planner:plan_trip')) 

    buffer = BytesIO()
    try:
        generate_trip_pdf_django(buffer, trip_plan_result, trip_params, schedule_pdf_obj_list_for_download)
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="plan_voyage_maroc_complet.pdf"'
        return response
    except Exception as e_pdf:
        logger.error(f"Erreur critique lors de la génération du PDF final: {e_pdf}", exc_info=True)
        messages.error(request, "Une erreur majeure est survenue lors de la génération du PDF. Veuillez réessayer.")
        return redirect(reverse('planner:plan_trip'))