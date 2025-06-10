# planner/utils.py
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from itertools import permutations
from geopy.distance import geodesic
import os
import pickle
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# --- Utilisation des chemins de settings.py ---
ACTIVITIES_JSON_PATH = settings.DATA_DIR / 'activities.json'
HOTELS_JSON_PATH = settings.DATA_DIR / 'hotels_with_real_coordinates_vf_v2.json'
MODEL_CACHE_DIR_DJANGO = settings.MODEL_CACHE_DIR_DJANGO
GRAPHS_CACHE_DIR_DJANGO = settings.GRAPHS_CACHE_DIR_DJANGO

OSMNX_AVAILABLE = False
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
    logger.info("Bibliothèques osmnx et networkx chargées avec succès.")
except ImportError:
    logger.warning("Les bibliothèques osmnx et/ou networkx ne sont pas installées. Les itinéraires détaillés sur carte ne seront pas disponibles.")

for cache_dir_path in [MODEL_CACHE_DIR_DJANGO, GRAPHS_CACHE_DIR_DJANGO]:
    if cache_dir_path == GRAPHS_CACHE_DIR_DJANGO and not OSMNX_AVAILABLE: continue
    if not os.path.exists(cache_dir_path):
        try: os.makedirs(cache_dir_path); logger.info(f"Dossier de cache créé: {cache_dir_path}")
        except OSError as e: logger.error(f"Impossible de créer le dossier de cache {cache_dir_path}: {e}")

CITY_NAME_MAPPING = {
    "Marrakech": ["marakech", "marrakesh"], "Fès": ["fez", "fes", "fes el bali", "fès el bali"],
    "Casablanca": ["casa"], "Meknès": ["meknes", "meknès el bali"], "Rabat": [], "Agadir": [],
    "Chefchaouen": ["chefchaouene", "chaouen"], "Essaouira": ["mogador"], "Ouarzazate": [],
    "Tangier": ["tanger", "tanger-assilah"], "Merzouga (Erg Chebbi)": ["merzouga", "erg chebbi"]
}
MANUAL_CITY_COORDINATES = {
    "Marrakech": {"latitude": 31.6295, "longitude": -7.9811}, "Fès": {"latitude": 34.0181, "longitude": -5.0078},
    "Casablanca": {"latitude": 33.5731, "longitude": -7.5898}, "Meknès": {"latitude": 33.8935, "longitude": -5.5473},
    "Rabat": {"latitude": 34.0209, "longitude": -6.8417}, "Agadir": {"latitude": 30.4202, "longitude": -9.5981},
    "Chefchaouen": {"latitude": 35.1688, "longitude": -5.2636}, "Essaouira": {"latitude": 31.5085, "longitude": -9.7595},
    "Ouarzazate": {"latitude": 30.9189, "longitude": -6.8934}, "Tangier": {"latitude": 35.7595, "longitude": -5.8330},
    "Merzouga (Erg Chebbi)": {"latitude": 31.0983, "longitude": -4.0119}
}

# --- Fonctions Utilitaires ---
def normalize_city_name(city_name, city_mapping):
    if not city_name or pd.isna(city_name): return None
    city_name_lower = str(city_name).strip().lower()
    for canonical, variations in city_mapping.items():
        if city_name_lower == canonical.lower() or city_name_lower in [v.lower() for v in variations]:
            return canonical
    return str(city_name).strip().capitalize()

def extract_city_from_hotel_location(location_str, canonical_activity_cities_list, city_name_mapping):
    if not location_str or pd.isna(location_str): return None
    location_lower = str(location_str).lower()
    for activity_city_canonical_name in canonical_activity_cities_list:
        if activity_city_canonical_name.lower() in location_lower: return activity_city_canonical_name
        if activity_city_canonical_name in city_name_mapping:
            for variation in city_name_mapping[activity_city_canonical_name]:
                if variation.lower() in location_lower: return activity_city_canonical_name
    parts = location_lower.split(" in ")
    candidate_city_str = parts[-1].split(",")[0].strip() if len(parts) > 1 else location_lower.split(",")[-1].strip()
    if candidate_city_str:
        normalized_candidate = normalize_city_name(candidate_city_str, city_name_mapping)
        if normalized_candidate in canonical_activity_cities_list: return normalized_candidate
        for canonical, variations in city_name_mapping.items():
            if normalized_candidate.lower() == canonical.lower() or normalized_candidate.lower() in [v.lower() for v in variations]:
                if canonical in canonical_activity_cities_list: return canonical
    for canonical_city in city_name_mapping.keys():
        if canonical_city.lower() in location_lower and canonical_city in canonical_activity_cities_list: return canonical_city
        for variation in city_name_mapping[canonical_city]:
            if variation.lower() in location_lower and canonical_city in canonical_activity_cities_list: return canonical_city
    return None

# --- Chargement et Prétraitement des Données ---
_CACHED_PREPROCESSED_DATA = None
def load_and_preprocess_data():
    global _CACHED_PREPROCESSED_DATA
    if _CACHED_PREPROCESSED_DATA is not None:
        logger.debug("Utilisation des données prétraitées en cache (mémoire module).")
        return _CACHED_PREPROCESSED_DATA

    logger.info("--- DÉBUT load_and_preprocess_data ---")
    logger.info(f"Chemin settings.DATA_DIR: {settings.DATA_DIR}")
    logger.info(f"Calculated ACTIVITIES_JSON_PATH: {ACTIVITIES_JSON_PATH}")
    logger.info(f"Calculated HOTELS_JSON_PATH: {HOTELS_JSON_PATH}")
    logger.info(f"Existence activities.json? : {os.path.exists(ACTIVITIES_JSON_PATH)}")
    logger.info(f"Existence hotels.json? : {os.path.exists(HOTELS_JSON_PATH)}")

    city_coordinates_map_for_tsp = {}
    activities_data_raw = []
    hotels_data_raw_list = []

    default_activity_cols = ['nom', 'type', 'budget_estime', 'description', 'duree_estimee', 'ville_normalisee', 'latitude', 'longitude', 'image_url', 'rating']
    default_hotel_cols = ['name', 'nom', 'type', 'ville_normalisee', 'rating', 'price_per_night', 'latitude', 'longitude', 'booking_link', 'description', 'image_url']

    try:
        with open(ACTIVITIES_JSON_PATH, 'r', encoding='utf-8') as f: activities_data_raw = json.load(f)
        logger.info(f"Fichier activities.json chargé. Type: {type(activities_data_raw)}. Nombre d'éléments (si liste): {len(activities_data_raw) if isinstance(activities_data_raw, list) else 'N/A'}")
    except FileNotFoundError:
        logger.error(f"ERREUR CRITIQUE: Fichier d'activités NON TROUVÉ: {ACTIVITIES_JSON_PATH}")
        activities_df = pd.DataFrame(columns=default_activity_cols)
        hotels_df = pd.DataFrame(columns=default_hotel_cols)
        _CACHED_PREPROCESSED_DATA = (activities_df, hotels_df, {})
        return _CACHED_PREPROCESSED_DATA
    except json.JSONDecodeError as e:
        logger.error(f"ERREUR CRITIQUE de décodage JSON dans {ACTIVITIES_JSON_PATH}: {e}", exc_info=True)
        raise

    try:
        with open(HOTELS_JSON_PATH, 'r', encoding='utf-8') as f:
            hotels_data_json = json.load(f)
            hotels_data_raw_list = hotels_data_json.get("hotels", [])
        logger.info(f"Fichier hotels.json chargé. Clé 'hotels' trouvée. Nombre d'hôtels bruts: {len(hotels_data_raw_list)}")
    except FileNotFoundError:
        logger.warning(f"Fichier d'hôtels NON TROUVÉ: {HOTELS_JSON_PATH}. Le processus continue sans données d'hôtels.")
        hotels_data_raw_list = []
    except json.JSONDecodeError as e:
        logger.error(f"ERREUR CRITIQUE de décodage JSON dans {HOTELS_JSON_PATH}: {e}", exc_info=True)
        raise
    except AttributeError as e:
        logger.error(f"ERREUR: Le fichier hotels.json ne semble pas avoir la structure attendue (objet avec clé 'hotels'). Erreur: {e}", exc_info=True)
        hotels_data_raw_list = []

    activities_list = []
    if not isinstance(activities_data_raw, list): activities_data_raw = []
    for city_data_act in activities_data_raw:
        raw_city_name_act = city_data_act.get("ville")
        normalized_city_name_act = normalize_city_name(raw_city_name_act, CITY_NAME_MAPPING)
        if not normalized_city_name_act: continue
        for activity in city_data_act.get("activites", []):
            if not isinstance(activity, dict):
                logger.warning(f"Élément d'activité non-dictionnaire ignoré pour {normalized_city_name_act}: {activity}")
                continue
            activity_entry = activity.copy()
            activity_entry["ville_normalisee"] = normalized_city_name_act
            coords_dict = activity_entry.get("coordonnees", {})
            activity_entry["latitude"] = pd.to_numeric(coords_dict.get("latitude"), errors='coerce')
            activity_entry["longitude"] = pd.to_numeric(coords_dict.get("longitude"), errors='coerce')
            if "tarif_recommande" in activity_entry and pd.notna(activity_entry["tarif_recommande"]):
                activity_entry["budget_estime"] = pd.to_numeric(activity_entry["tarif_recommande"], errors='coerce')
            elif "budget_estime" in activity_entry:
                activity_entry["budget_estime"] = pd.to_numeric(activity_entry["budget_estime"], errors='coerce')
            else: activity_entry["budget_estime"] = np.nan
            activity_entry['nom'] = activity_entry.get('nom', f"Activité à {normalized_city_name_act}")
            activity_entry['type'] = activity_entry.get('type', 'Inconnu')
            activity_entry['description'] = activity_entry.get('description', '')
            activity_entry['duree_estimee'] = activity_entry.get('duree_estimee', 'N/A')
            activity_entry['image_url'] = activity_entry.get('image_url')
            activity_entry['rating'] = pd.to_numeric(activity_entry.get('rating'), errors='coerce')
            activities_list.append(activity_entry)
    
    if activities_list:
        activities_df = pd.DataFrame(activities_list)
        if "budget_estime" in activities_df.columns: activities_df = activities_df.dropna(subset=["budget_estime"])
        if "type" in activities_df.columns: activities_df["type"] = activities_df["type"].fillna("Inconnu").astype(str)
        else: activities_df["type"] = "Inconnu"
        if "nom" not in activities_df.columns: activities_df["nom"] = "Activité sans nom"
    else:
        activities_df = pd.DataFrame(columns=default_activity_cols)
    logger.info(f"Nombre d'activités après traitement initial: {len(activities_df)}")

    canonical_activity_cities_list = sorted(activities_df["ville_normalisee"].unique().tolist()) if not activities_df.empty and "ville_normalisee" in activities_df.columns else []

    hotels_df_processed_list = []
    if hotels_data_raw_list:
        for hotel_raw in hotels_data_raw_list:
            if not isinstance(hotel_raw, dict):
                logger.warning(f"Élément d'hôtel non-dictionnaire ignoré: {hotel_raw}")
                continue
            hotel_entry = hotel_raw.copy()
            hotel_entry['name'] = hotel_entry.get('name', "Hôtel Inconnu")
            hotel_entry['nom'] = hotel_entry['name'] 
            hotel_entry['type'] = 'hotel'
            hotel_entry['description'] = hotel_entry.get('description', '')
            hotel_entry['booking_link'] = hotel_entry.get('booking_link')
            hotel_entry['image_url'] = hotel_entry.get('image_url')
            loc_str = hotel_entry.get("location")
            norm_city = extract_city_from_hotel_location(loc_str, canonical_activity_cities_list, CITY_NAME_MAPPING)
            hotel_entry["ville_normalisee"] = norm_city if norm_city else normalize_city_name(loc_str, CITY_NAME_MAPPING)
            for col_num in ["rating", "price_per_night", "latitude", "longitude"]:
                hotel_entry[col_num] = pd.to_numeric(hotel_entry.get(col_num), errors='coerce')
            hotels_df_processed_list.append(hotel_entry)

    if hotels_df_processed_list:
        hotels_df = pd.DataFrame(hotels_df_processed_list)
        if not hotels_df.empty and 'ville_normalisee' in hotels_df.columns:
            if "rating" in hotels_df.columns and not hotels_df["rating"].isnull().all():
                hotels_df["rating"] = hotels_df.groupby("ville_normalisee", group_keys=False)["rating"].apply(lambda x: x.fillna(x.median()))
                hotels_df["rating"].fillna(hotels_df["rating"].median(skipna=True), inplace=True)
            elif "rating" in hotels_df.columns: hotels_df["rating"].fillna(7.0, inplace=True)
            else: hotels_df["rating"] = 7.0
            if "name" not in hotels_df.columns: hotels_df["name"] = "Hôtel sans nom"
            cols_to_check = ["ville_normalisee", "price_per_night", "latitude", "longitude", "rating", "name"]
            existing_cols = [col for col in cols_to_check if col in hotels_df.columns]
            if existing_cols: hotels_df = hotels_df.dropna(subset=existing_cols) # Peut vider le df si des données cruciales manquent
            
            if not hotels_df.empty: # Vérifier après dropna
                city_coords_from_hotels = hotels_df.groupby("ville_normalisee")[["latitude", "longitude"]].mean().reset_index()
                for _, row in city_coords_from_hotels.iterrows():
                    if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
                        city_coordinates_map_for_tsp[row["ville_normalisee"]] = {"latitude": row["latitude"], "longitude": row["longitude"]}
    else: hotels_df = pd.DataFrame(columns=default_hotel_cols)
    logger.info(f"Nombre d'hôtels après traitement initial: {len(hotels_df)}")

    for city_name_act in canonical_activity_cities_list:
        if city_name_act not in city_coordinates_map_for_tsp and city_name_act in MANUAL_CITY_COORDINATES:
            city_coordinates_map_for_tsp[city_name_act] = MANUAL_CITY_COORDINATES[city_name_act]
        if not activities_df.empty and "ville_normalisee" in activities_df.columns and \
           "latitude" in activities_df.columns and "longitude" in activities_df.columns and \
           city_name_act in city_coordinates_map_for_tsp:
            city_coords_tsp = city_coordinates_map_for_tsp[city_name_act]
            mask_act_no_coords = (activities_df["ville_normalisee"] == city_name_act) & (activities_df["latitude"].isnull() | activities_df["longitude"].isnull())
            if mask_act_no_coords.any():
                activities_df.loc[mask_act_no_coords, "latitude"] = city_coords_tsp["latitude"]
                activities_df.loc[mask_act_no_coords, "longitude"] = city_coords_tsp["longitude"]
        elif not activities_df.empty: logger.warning(f"Coordonnées ville '{city_name_act}' non trouvées ou DF activités incomplet pour imputer.")

    logger.info(f"Fin - Données prêtes: {len(activities_df)} activités, {len(hotels_df)} hôtels.")
    _CACHED_PREPROCESSED_DATA = (activities_df, hotels_df, city_coordinates_map_for_tsp)
    return activities_df, hotels_df, city_coordinates_map_for_tsp

# --- Fonctions de Modèle ML (Complètes) ---
_CACHED_MODELS_COMPONENTS = {}
def get_model_components_from_file_django(city_name, model_type):
    filename = os.path.join(MODEL_CACHE_DIR_DJANGO, f"{city_name}_{model_type}_components.pkl")
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f: return pickle.load(f)
        except Exception as e: logger.warning(f"Erreur chargement cache modèle {model_type} pour {city_name}: {e}", exc_info=True)
    return None

def save_model_components_to_file_django(city_name, model_type, components):
    filename = os.path.join(MODEL_CACHE_DIR_DJANGO, f"{city_name}_{model_type}_components.pkl")
    try:
        with open(filename, 'wb') as f: pickle.dump(components, f)
        logger.info(f"Cache modèle {model_type} pour {city_name} sauvegardé.")
    except Exception as e: logger.warning(f"Erreur sauvegarde cache modèle {model_type} pour {city_name}: {e}", exc_info=True)

def get_hotel_recommender_model_django(city_name_key, hotels_in_city_df_for_model):
    if hotels_in_city_df_for_model.empty: logger.warning(f"Aucun hôtel pour modèle {city_name_key}."); return None
    actual_city_name = hotels_in_city_df_for_model["ville_normalisee"].iloc[0]
    cache_key = f"hotel_model_{actual_city_name}"
    if cache_key in _CACHED_MODELS_COMPONENTS:
        logger.debug(f"Modèle hôtel pour {actual_city_name} depuis cache mémoire.")
        return _CACHED_MODELS_COMPONENTS[cache_key]
    cached_components_file = get_model_components_from_file_django(actual_city_name, "hotel")
    if cached_components_file and len(cached_components_file) == 4:
        logger.info(f"Modèle hôtel pour {actual_city_name} depuis cache fichier.")
        _CACHED_MODELS_COMPONENTS[cache_key] = cached_components_file; return cached_components_file
    
    logger.info(f"Entraînement du modèle hôtel pour {actual_city_name}...")
    features_df = hotels_in_city_df_for_model[["price_per_night", "rating"]].copy()
    features_df["rating_inverted"] = 10.0 - features_df["rating"]
    scaler = MinMaxScaler()
    feature_columns_for_scaler = ["price_per_night", "rating_inverted"]
    if not all(col in features_df.columns for col in feature_columns_for_scaler) or features_df[feature_columns_for_scaler].isnull().all().all():
        logger.warning(f"Données hôtel insuffisantes/NaN pour {actual_city_name}."); return None
    features_df_cleaned = features_df.dropna(subset=feature_columns_for_scaler).copy()
    if features_df_cleaned.empty: logger.warning(f"Plus de données hôtel après NaN pour {actual_city_name}."); return None
    scaled_features_values = scaler.fit_transform(features_df_cleaned[feature_columns_for_scaler])
    n_neighbors = min(len(features_df_cleaned), 10)
    if n_neighbors == 0: logger.warning(f"Pas assez de voisins hôtel pour {actual_city_name}."); return None
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(scaled_features_values)
    components_to_save = (model, scaler, feature_columns_for_scaler, features_df_cleaned.index)
    save_model_components_to_file_django(actual_city_name, "hotel", components_to_save)
    _CACHED_MODELS_COMPONENTS[cache_key] = components_to_save
    return components_to_save

def get_activity_recommender_model_django(city_name_key, activities_in_city_df_for_model):
    if activities_in_city_df_for_model.empty: logger.warning(f"Aucune activité pour modèle {city_name_key}."); return None
    actual_city_name = activities_in_city_df_for_model["ville_normalisee"].iloc[0]
    cache_key = f"activity_model_{actual_city_name}"
    if cache_key in _CACHED_MODELS_COMPONENTS:
        logger.debug(f"Modèle activité pour {actual_city_name} depuis cache mémoire.")
        return _CACHED_MODELS_COMPONENTS[cache_key]
    components_file = get_model_components_from_file_django(actual_city_name, "activity")
    if components_file and len(components_file) == 2 : # Model et preprocessor
        logger.info(f"Modèle activité pour {actual_city_name} depuis cache fichier.")
        _CACHED_MODELS_COMPONENTS[cache_key] = components_file; return components_file
    
    logger.info(f"Entraînement du modèle activité pour {actual_city_name}...")
    activities_df_model_copy = activities_in_city_df_for_model.copy()
    if 'tarif_recommande' in activities_df_model_copy.columns and 'budget_estime' not in activities_df_model_copy.columns:
        activities_df_model_copy = activities_df_model_copy.rename(columns={'tarif_recommande': 'budget_estime'})
    elif 'tarif_recommande' in activities_df_model_copy.columns and 'budget_estime' in activities_df_model_copy.columns:
        activities_df_model_copy['budget_estime'] = activities_df_model_copy['tarif_recommande'].fillna(activities_df_model_copy['budget_estime'])
    numerical_features, categorical_features = ["budget_estime"], ["type"]
    if not all(col in activities_df_model_copy.columns for col in numerical_features + categorical_features): logger.warning(f"Colonnes manquantes activité pour {actual_city_name}."); return None
    activities_df_model_copy[numerical_features] = activities_df_model_copy[numerical_features].apply(pd.to_numeric, errors='coerce')
    activities_df_model_copy.dropna(subset=numerical_features, inplace=True)
    if activities_df_model_copy.empty: logger.warning(f"Aucune activité budget valide pour {actual_city_name}."); return None
    preprocessor = ColumnTransformer(transformers=[('num', MinMaxScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)], remainder='drop')
    try: transformed_features = preprocessor.fit_transform(activities_df_model_copy)
    except Exception as e: logger.warning(f"Erreur transformation activités pour {actual_city_name}: {e}", exc_info=True); return None
    n_neighbors = min(len(activities_df_model_copy), 15)
    if n_neighbors == 0 or transformed_features.shape[0] == 0: logger.warning(f"Pas assez données/voisins activité pour {actual_city_name}."); return None
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(transformed_features)
    components = (model, preprocessor)
    save_model_components_to_file_django(actual_city_name, "activity", components)
    _CACHED_MODELS_COMPONENTS[cache_key] = components
    return components

# --- Fonctions TSP (Complètes) ---
def calculate_total_distance(path, city_coordinates_map):
    total_dist = 0
    for i in range(len(path) - 1):
        city1_name, city2_name = path[i], path[i+1]
        coord1_dict, coord2_dict = city_coordinates_map.get(city1_name), city_coordinates_map.get(city2_name)
        if coord1_dict and coord2_dict and pd.notna(coord1_dict.get('latitude')) and pd.notna(coord1_dict.get('longitude')) and pd.notna(coord2_dict.get('latitude')) and pd.notna(coord2_dict.get('longitude')):
            try: total_dist += geodesic((coord1_dict['latitude'], coord1_dict['longitude']), (coord2_dict['latitude'], coord2_dict['longitude'])).km
            except ValueError: total_dist += float('inf')
        else: total_dist += float('inf')
    return total_dist

def find_optimal_path_greedy(start_city, other_cities_list, city_coordinates_map):
    if not city_coordinates_map.get(start_city) or pd.isna(city_coordinates_map.get(start_city, {}).get('latitude')):
        cities_with_coords = [c for c in other_cities_list if city_coordinates_map.get(c) and pd.notna(city_coordinates_map.get(c, {}).get('latitude'))]
        cities_without_coords = [c for c in other_cities_list if c not in cities_with_coords]
        if not cities_with_coords: return [start_city] + other_cities_list
        new_start_city = cities_with_coords[0]; remaining_for_greedy = cities_with_coords[1:]
        path_segment = find_optimal_path_greedy(new_start_city, remaining_for_greedy, city_coordinates_map)
        return path_segment + cities_without_coords + [start_city]
    path, current_city_name, remaining_cities = [start_city], start_city, list(other_cities_list)
    while remaining_cities:
        next_city_candidate, min_dist_to_candidate = None, float('inf')
        current_coord_dict = city_coordinates_map.get(current_city_name)
        if not current_coord_dict or pd.isna(current_coord_dict.get('latitude')): # Sécurité
            logger.warning(f"find_optimal_path_greedy: Coordonnées manquantes pour la ville actuelle {current_city_name}. Tentative de continuer.")
            # Tenter de prendre la prochaine ville avec des coordonnées dans remaining_cities
            next_city_with_coords = next((c for c in remaining_cities if city_coordinates_map.get(c) and pd.notna(city_coordinates_map.get(c, {}).get('latitude'))), None)
            if next_city_with_coords:
                path.append(next_city_with_coords)
                current_city_name = next_city_with_coords
                remaining_cities.remove(next_city_with_coords)
                continue
            else: # Plus aucune ville avec coordonnées
                path.extend(sorted(remaining_cities)); break 
        current_coord_tuple = (current_coord_dict['latitude'], current_coord_dict['longitude'])
        found_next_candidate_with_coords = False; temp_remaining_cities = list(remaining_cities)
        for city_candidate_name in temp_remaining_cities:
            candidate_coord_dict = city_coordinates_map.get(city_candidate_name)
            if candidate_coord_dict and pd.notna(candidate_coord_dict.get('latitude')):
                found_next_candidate_with_coords = True
                try:
                    dist = geodesic(current_coord_tuple, (candidate_coord_dict['latitude'], candidate_coord_dict['longitude'])).km
                    if dist < min_dist_to_candidate: min_dist_to_candidate = dist; next_city_candidate = city_candidate_name
                except ValueError: pass
        if next_city_candidate: path.append(next_city_candidate); current_city_name = next_city_candidate; remaining_cities.remove(next_city_candidate)
        elif found_next_candidate_with_coords :
            path.extend(sorted(r_city for r_city in remaining_cities if city_coordinates_map.get(r_city) and pd.notna(city_coordinates_map.get(r_city, {}).get('latitude'))))
            path.extend(sorted(r_city for r_city in remaining_cities if not (city_coordinates_map.get(r_city) and pd.notna(city_coordinates_map.get(r_city, {}).get('latitude')))))
            break
        else: path.extend(sorted(remaining_cities)); break
    return path

def find_optimal_path_permutations(cities_to_visit_list, city_coordinates_map):
    if not cities_to_visit_list: return []
    cities_with_coords_for_path = [city for city in cities_to_visit_list if city_coordinates_map.get(city) and pd.notna(city_coordinates_map[city].get('latitude')) and pd.notna(city_coordinates_map[city].get('longitude'))]
    cities_without_coords_for_path = [city for city in cities_to_visit_list if city not in cities_with_coords_for_path]
    if not cities_with_coords_for_path: return cities_to_visit_list # Retourne l'ordre initial si aucune ville avec coords
    if len(cities_with_coords_for_path) == 1: return cities_with_coords_for_path + cities_without_coords_for_path
    start_city = cities_with_coords_for_path[0]; other_cities_for_permutation = cities_with_coords_for_path[1:]
    best_path_segment = [start_city] + other_cities_for_permutation; min_total_distance = calculate_total_distance(best_path_segment, city_coordinates_map)
    if len(other_cities_for_permutation) > 6: # Limite (N-1)! permutations
        logger.warning(f"Trop de villes ({len(cities_with_coords_for_path)}) pour optimisation par permutations inter-villes. Approche gloutonne utilisée.");
        greedy_path = find_optimal_path_greedy(start_city, other_cities_for_permutation, city_coordinates_map)
        return greedy_path + cities_without_coords_for_path
    for p in permutations(other_cities_for_permutation):
        current_path_segment = [start_city] + list(p); dist = calculate_total_distance(current_path_segment, city_coordinates_map)
        if dist < min_total_distance: min_total_distance = dist; best_path_segment = current_path_segment
    return best_path_segment + cities_without_coords_for_path

# --- Fonctions OSMnx et A* (Complètes) ---
_CACHED_CITY_GRAPHS = {}
if OSMNX_AVAILABLE:
    def get_city_drive_graph_django(city_name_query, country="Morocco"):
        cache_key = f"graph_{city_name_query.replace(' ', '_').lower()}"
        if cache_key in _CACHED_CITY_GRAPHS: logger.debug(f"Graphe {city_name_query} depuis cache mémoire."); return _CACHED_CITY_GRAPHS[cache_key]
        graph_filename_base = city_name_query.split(',')[0].replace(' ', '_').replace('(', '').replace(')', '').lower()
        graph_filename = os.path.join(GRAPHS_CACHE_DIR_DJANGO, f"{graph_filename_base}_drive.graphml")
        if os.path.exists(graph_filename):
            try: G = ox.load_graphml(graph_filename); _CACHED_CITY_GRAPHS[cache_key] = G; logger.info(f"Graphe {city_name_query} chargé depuis fichier."); return G
            except Exception as e: logger.warning(f"Impossible charger cache graphe {graph_filename}: {e}. Re-téléchargement.")
        logger.info(f"Téléchargement réseau routier pour {city_name_query}, {country}...")
        try:
            G = ox.graph_from_place(f"{city_name_query}, {country}", network_type="drive", retain_all=False, truncate_by_edge=True)
            ox.save_graphml(G, filepath=graph_filename); logger.info(f"Graphe pour {city_name_query} sauvegardé.")
            _CACHED_CITY_GRAPHS[cache_key] = G; return G
        except Exception as e: logger.error(f"Impossible télécharger/traiter graphe pour {city_name_query}: {e}", exc_info=True); return None

    def get_astar_route_coords_from_graph_django(graph, start_point_data, end_point_data):
        if graph is None: return None, 0, 0
        start_nom = start_point_data.get('nom', 'Départ_inconnu')
        end_nom = end_point_data.get('nom', 'Arrivée_inconnue')
        start_coords = (start_point_data.get('latitude'), start_point_data.get('longitude'))
        end_coords = (end_point_data.get('latitude'), end_point_data.get('longitude'))
        if not (pd.notna(start_coords[0]) and pd.notna(start_coords[1]) and pd.notna(end_coords[0]) and pd.notna(end_coords[1])):
            logger.warning(f"Coordonnées manquantes pour A* entre '{start_nom}' et '{end_nom}'.")
            return None, 0, 0
        try:
            start_node = ox.nearest_nodes(graph, X=start_coords[1], Y=start_coords[0])
            end_node = ox.nearest_nodes(graph, X=end_coords[1], Y=end_coords[0])
            route_nodes = nx.astar_path(graph, start_node, end_node, weight="length")
            route_coords_for_polyline = [(graph.nodes[node_id]['y'], graph.nodes[node_id]['x']) for node_id in route_nodes]
            route_length_m = 0
            for i in range(len(route_nodes) - 1):
                u, v = route_nodes[i], route_nodes[i+1]
                edge_data = graph.get_edge_data(u,v) # Peut retourner un dict si multigraph
                if edge_data: 
                    if 0 in edge_data: # Cas le plus simple, non-multigraph ou première arête
                        route_length_m += edge_data[0].get('length', 0)
                    else: # MultiGraph, prendre la plus courte
                        lengths = [d.get('length', float('inf')) for d in edge_data.values()]
                        route_length_m += min(lengths) if lengths else 0
            route_length_km = route_length_m / 1000.0
            avg_speed_kmh = 25; route_time_min = (route_length_km / avg_speed_kmh) * 60 if avg_speed_kmh > 0 else 0
            return route_coords_for_polyline, route_length_km, route_time_min
        except nx.NetworkXNoPath: logger.warning(f"Aucun chemin A* trouvé entre '{start_nom}' et '{end_nom}'."); return None, 0, 0
        except Exception as e: logger.error(f"Erreur calcul itinéraire A* entre '{start_nom}' et '{end_nom}': {e}", exc_info=True); return None, 0, 0

    def optimize_daily_activities_order_on_graph_django(city_graph, hotel_location_data, daily_activities_data):
        if not OSMNX_AVAILABLE or city_graph is None: return optimize_daily_activities_order_geodesic_django(hotel_location_data, daily_activities_data)
        if not daily_activities_data: return [hotel_location_data, hotel_location_data] if hotel_location_data else []
        points_to_visit = []
        start_end_point_data = None
        if hotel_location_data and pd.notna(hotel_location_data.get('latitude')) and pd.notna(hotel_location_data.get('longitude')):
            start_end_point_data = hotel_location_data; points_to_visit.append(hotel_location_data)
        valid_activities = [act for act in daily_activities_data if pd.notna(act.get('latitude')) and pd.notna(act.get('longitude'))]
        if not valid_activities: return [start_end_point_data, start_end_point_data] if start_end_point_data else []
        points_to_visit.extend(valid_activities)
        num_points = len(points_to_visit); dist_matrix = np.full((num_points, num_points), np.inf)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                _, dist_km, _ = get_astar_route_coords_from_graph_django(city_graph, points_to_visit[i], points_to_visit[j])
                if dist_km is not None and dist_km >= 0: dist_matrix[i, j] = dist_matrix[j, i] = dist_km if dist_km > 0 else 0.001 # Petite distance pour même noeud
        
        if start_end_point_data:
            activity_indices = list(range(1, num_points)) # Index des activités dans points_to_visit
            if not activity_indices: return [start_end_point_data, start_end_point_data] # Juste l'hôtel
            MAX_TSP_NODES_FOR_PERMUTATION = 6 
            if len(activity_indices) > MAX_TSP_NODES_FOR_PERMUTATION:
                logger.warning(f"Trop d'activités ({len(activity_indices)}) pour TSP A* par permutations. Utilisation heuristique gloutonne A*.")
                current_idx = 0; path_indices = [0]; remaining_indices = list(activity_indices)
                while remaining_indices:
                    next_best_idx = min(remaining_indices, key=lambda idx_loop: dist_matrix[current_idx, idx_loop])
                    if dist_matrix[current_idx, next_best_idx] == np.inf: logger.warning("TSP A* glouton: chemin infini rencontré."); break
                    path_indices.append(next_best_idx); remaining_indices.remove(next_best_idx); current_idx = next_best_idx
                if dist_matrix[current_idx, 0] != np.inf: path_indices.append(0) # Retour hôtel
                else: logger.warning("TSP A* glouton: Impossible de retourner à l'hôtel (chemin infini).")
                return [points_to_visit[i] for i in path_indices]

            best_perm_indices, min_total_dist = None, float('inf')
            for p in permutations(activity_indices):
                current_dist = dist_matrix[0, p[0]] # Hôtel -> Première activité
                for k_perm in range(len(p) - 1): current_dist += dist_matrix[p[k_perm], p[k_perm+1]] # Entre activités
                current_dist += dist_matrix[p[-1], 0] # Dernière activité -> Hôtel
                if current_dist < min_total_dist: min_total_dist = current_dist; best_perm_indices = p
            
            if best_perm_indices and min_total_dist != float('inf'):
                final_ordered_route_data = [start_end_point_data]
                final_ordered_route_data.extend([points_to_visit[idx] for idx in best_perm_indices])
                final_ordered_route_data.append(start_end_point_data)
                return final_ordered_route_data
            else:
                logger.warning("TSP A* par permutations n'a pas trouvé de chemin complet valide, retour à l'ordre géodésique.")
                return optimize_daily_activities_order_geodesic_django(hotel_location_data, daily_activities_data)
        else: # Pas d'hôtel, optimiser l'ordre des activités elles-mêmes (chemin ouvert)
            logger.info("Optimisation A* sans hôtel non implémentée comme circuit, retour à géodésique ouvert.")
            return optimize_daily_activities_order_geodesic_django(None, daily_activities_data)

def optimize_daily_activities_order_geodesic_django(hotel_location_data, daily_activities_data):
    if not daily_activities_data: return [hotel_location_data, hotel_location_data] if hotel_location_data else []
    start_end_node_info = None
    if hotel_location_data and pd.notna(hotel_location_data.get('latitude')) and pd.notna(hotel_location_data.get('longitude')):
        start_end_node_info = {'data': hotel_location_data, 'coords': (hotel_location_data['latitude'], hotel_location_data['longitude'])}
    activity_nodes = [{'data': act, 'coords': (act['latitude'], act['longitude'])} for act in daily_activities_data if pd.notna(act.get('latitude')) and pd.notna(act.get('longitude'))]
    if not activity_nodes: return [start_end_node_info['data'], start_end_node_info['data']] if start_end_node_info else []
    if not start_end_node_info: # Chemin ouvert
        if len(activity_nodes) <= 1: return [node['data'] for node in activity_nodes]
        path = [activity_nodes[0]]; remaining = activity_nodes[1:]
        while remaining: last_node = path[-1]; next_node = min(remaining, key=lambda node: geodesic(last_node['coords'], node['coords']).km); path.append(next_node); remaining.remove(next_node)
        return [node['data'] for node in path]
    if len(activity_nodes) == 1: return [start_end_node_info['data']] + [activity_nodes[0]['data']] + [start_end_node_info['data']]
    MAX_TSP_NODES_GEO = 6
    if len(activity_nodes) > MAX_TSP_NODES_GEO:
        logger.warning(f"Trop d'activités ({len(activity_nodes)}) pour TSP géodésique exact. Glouton utilisé.")
        current_path = [start_end_node_info]; remaining = list(activity_nodes)
        while remaining: last_node = current_path[-1]; next_node = min(remaining, key=lambda node: geodesic(last_node['coords'], node['coords']).km); current_path.append(next_node); remaining.remove(next_node)
        current_path.append(start_end_node_info)
        return [node['data'] for node in current_path]
    best_perm_data, min_dist_geo = [], float('inf')
    for p_nodes_geo in permutations(activity_nodes):
        current_dist_geo = geodesic(start_end_node_info['coords'], p_nodes_geo[0]['coords']).km + sum(geodesic(p_nodes_geo[i]['coords'], p_nodes_geo[i+1]['coords']).km for i in range(len(p_nodes_geo) - 1)) + geodesic(p_nodes_geo[-1]['coords'], start_end_node_info['coords']).km
        if current_dist_geo < min_dist_geo: min_dist_geo = current_dist_geo; best_perm_data = [node['data'] for node in p_nodes_geo]
    if best_perm_data and min_dist_geo != float('inf'): return [start_end_node_info['data']] + best_perm_data + [start_end_node_info['data']]
    logger.warning("TSP Géodésique : Aucune permutation valide (ou toutes inf), retour à glouton (fallback).")
    current_path = [start_end_node_info]; remaining = list(activity_nodes) # Glouton fallback
    while remaining: last_node = current_path[-1]; next_node = min(remaining, key=lambda node: geodesic(last_node['coords'], node['coords']).km); current_path.append(next_node); remaining.remove(next_node)
    current_path.append(start_end_node_info)
    return [node['data'] for node in current_path]

# --- Fonction de Recommandation par Ville (Version Complète avec robustesse) ---
def recommend_for_city_django(city_name, hotels_df_global, activities_df_global,
                             daily_budget_hotel, budget_activities_for_stay_in_city,
                             min_hotel_rating, activity_preferences, # activity_preferences est une liste
                             num_hotel_recs=1, num_activity_recs_per_day=3, num_days_in_city=1,
                             use_astar_intra_city=True):
    logger.debug(f"--- recommend_for_city_django pour : {city_name}, jours: {num_days_in_city} ---")
    recommendations = {"ville": city_name, "hotel": None,
                       "activites_par_jour_optimisees": [],
                       "budget_activites_depense": 0,
                       "num_activity_recs_per_day": num_activity_recs_per_day,
                       "itineraire_segments_par_jour": []
                       }
    current_hotel_selection = None

    if hotels_df_global is not None and not hotels_df_global.empty and 'ville_normalisee' in hotels_df_global.columns:
        hotels_in_city_df = hotels_df_global[hotels_df_global["ville_normalisee"] == city_name].copy()
        logger.debug(f"Recommandation Hôtel pour {city_name}: {len(hotels_in_city_df)} hôtels bruts trouvés.")
        if not hotels_in_city_df.empty:
            model_components_hotel = get_hotel_recommender_model_django(city_name, hotels_in_city_df)
            if model_components_hotel:
                model_h, scaler_h, feature_cols_h, original_indices_h = model_components_hotel
                valid_original_indices = [idx for idx in original_indices_h if idx in hotels_in_city_df.index]
                if valid_original_indices:
                    hotels_df_for_query = hotels_in_city_df.loc[valid_original_indices].copy()
                    if not hotels_df_for_query.empty:
                        query_data = [[daily_budget_hotel, 10.0 - min_hotel_rating]]
                        query_df_hotel_model_space = pd.DataFrame(query_data, columns=feature_cols_h)
                        try:
                            scaled_query_hotel = scaler_h.transform(query_df_hotel_model_space)
                            _, indices_from_model = model_h.kneighbors(scaled_query_hotel)
                            raw_recs_h = hotels_df_for_query.iloc[indices_from_model[0]].copy()
                            filtered_recs_h = raw_recs_h[(raw_recs_h["price_per_night"] <= daily_budget_hotel) & (raw_recs_h["rating"] >= min_hotel_rating)].copy()
                            if not filtered_recs_h.empty:
                                filtered_recs_h = filtered_recs_h.sort_values(by=["rating", "price_per_night"], ascending=[False, True])
                                recommended_hotels_list_raw = filtered_recs_h.head(num_hotel_recs).to_dict('records')
                                if recommended_hotels_list_raw:
                                    current_hotel_selection = recommended_hotels_list_raw[0].copy()
                                    current_hotel_selection['nom'] = current_hotel_selection.get('name', current_hotel_selection.get('nom', f"Hôtel suggéré à {city_name}"))
                                    current_hotel_selection['type'] = 'hotel' # Type explicite pour les hôtels
                                    current_hotel_selection['description'] = current_hotel_selection.get('description')
                                    current_hotel_selection['latitude'] = current_hotel_selection.get('latitude')
                                    current_hotel_selection['longitude'] = current_hotel_selection.get('longitude')
                                    current_hotel_selection['price_per_night'] = current_hotel_selection.get('price_per_night')
                                    current_hotel_selection['rating'] = current_hotel_selection.get('rating')
                                    current_hotel_selection['booking_link'] = current_hotel_selection.get('booking_link')
                                    current_hotel_selection['image_url'] = current_hotel_selection.get('image_url')
                                    recommendations["hotel"] = [current_hotel_selection] # Doit être une liste de dictionnaires
                                    logger.debug(f"Hôtel sélectionné pour {city_name}: {current_hotel_selection['nom']}")
                            else: logger.warning(f"Aucun hôtel après filtrage budget/rating pour {city_name}")
                        except Exception as e: logger.error(f"Erreur reco hôtel pour {city_name}: {e}", exc_info=True)
            else: logger.warning(f"Pas de modèle hôtel pour {city_name}")
    else: logger.warning(f"Pas de données hôtels globales pour {city_name} ou 'ville_normalisee' manquante.")

    all_selected_activities_for_stay = []
    total_spent_on_activities = 0
    if activities_df_global is not None and not activities_df_global.empty and 'ville_normalisee' in activities_df_global.columns:
        activities_in_city_df = activities_df_global[activities_df_global["ville_normalisee"] == city_name].copy()
        if 'budget_estime' not in activities_in_city_df.columns and 'tarif_recommande' in activities_in_city_df.columns:
            activities_in_city_df = activities_in_city_df.rename(columns={'tarif_recommande': 'budget_estime'})
        if 'budget_estime' in activities_in_city_df.columns :
            activities_in_city_df['budget_estime'] = pd.to_numeric(activities_in_city_df['budget_estime'], errors='coerce')
            activities_in_city_df.dropna(subset=['budget_estime'], inplace=True)
        
        if not activities_in_city_df.empty and 'budget_estime' in activities_in_city_df.columns and 'type' in activities_in_city_df.columns:
            model_components_activity = get_activity_recommender_model_django(city_name, activities_in_city_df)
            if model_components_activity:
                model_act, preprocessor_act = model_components_activity
                type_mode = activities_in_city_df["type"].mode(); query_type_act_default = type_mode[0] if not type_mode.empty else "Inconnu"
                query_type_act_final = query_type_act_default
                if activity_preferences:
                    valid_prefs = [p for p in activity_preferences if p in activities_in_city_df["type"].unique()]
                    if valid_prefs: query_type_act_final = valid_prefs[0]
                query_budget_per_activity = budget_activities_for_stay_in_city / (num_days_in_city * num_activity_recs_per_day + 1e-6)
                query_df_act = pd.DataFrame({"budget_estime": [query_budget_per_activity], "type": [query_type_act_final]})
                try:
                    scaled_query_act = preprocessor_act.transform(query_df_act)
                    _, indices_a = model_act.kneighbors(scaled_query_act)
                    valid_indices_a = [idx for idx in indices_a[0] if idx < len(activities_in_city_df)]
                    if valid_indices_a:
                        raw_recs_act = activities_in_city_df.iloc[valid_indices_a].copy()
                        filtered_recs_act_type = raw_recs_act
                        if activity_preferences:
                            temp_filtered = raw_recs_act[raw_recs_act["type"].isin(activity_preferences)]
                            if not temp_filtered.empty: filtered_recs_act_type = temp_filtered.copy()
                        if not filtered_recs_act_type.empty:
                            filtered_recs_act_final = filtered_recs_act_type[filtered_recs_act_type['budget_estime'] <= budget_activities_for_stay_in_city].copy()
                            if not filtered_recs_act_final.empty:
                                filtered_recs_act_final = filtered_recs_act_final.sort_values(by="budget_estime")
                                max_acts_total = num_activity_recs_per_day * num_days_in_city
                                for _, row_act_series in filtered_recs_act_final.iterrows():
                                    if len(all_selected_activities_for_stay) < max_acts_total and total_spent_on_activities + row_act_series['budget_estime'] <= budget_activities_for_stay_in_city:
                                        activity_dict = row_act_series.to_dict()
                                        activity_dict['nom'] = activity_dict.get('nom', f"Activité {activity_dict.get('type', 'Inconnue')} à {city_name}")
                                        activity_dict['type'] = activity_dict.get('type', 'Inconnu')
                                        activity_dict['description'] = activity_dict.get('description')
                                        activity_dict['latitude'] = activity_dict.get('latitude')
                                        activity_dict['longitude'] = activity_dict.get('longitude')
                                        activity_dict['budget_estime'] = activity_dict.get('budget_estime')
                                        activity_dict['duree_estimee'] = activity_dict.get('duree_estimee')
                                        activity_dict['rating'] = activity_dict.get('rating') 
                                        activity_dict['image_url'] = activity_dict.get('image_url')
                                        if not any(act['nom'] == activity_dict['nom'] for act in all_selected_activities_for_stay):
                                            all_selected_activities_for_stay.append(activity_dict)
                                            total_spent_on_activities += row_act_series['budget_estime']
                except Exception as e_act: logger.error(f"Erreur reco activités: {e_act}", exc_info=True)
    recommendations["budget_activites_depense"] = total_spent_on_activities
    logger.debug(f"Activités sélectionnées pour {city_name} ({len(all_selected_activities_for_stay)}): {[a.get('nom','SansNom') for a in all_selected_activities_for_stay]}")

    city_graph = None
    if OSMNX_AVAILABLE and use_astar_intra_city: city_graph = get_city_drive_graph_django(city_name)

    activity_ptr = 0
    for day_num in range(num_days_in_city):
        activities_for_this_day_raw = []
        for _ in range(num_activity_recs_per_day):
            if activity_ptr < len(all_selected_activities_for_stay): activities_for_this_day_raw.append(all_selected_activities_for_stay[activity_ptr]); activity_ptr += 1
            else: break
        
        daily_ordered_points_data, daily_route_segments_coords = [], []
        hotel_loc_data_for_day = current_hotel_selection.copy() if current_hotel_selection else None
        if hotel_loc_data_for_day and (pd.isna(hotel_loc_data_for_day.get('nom')) or not hotel_loc_data_for_day.get('nom')): hotel_loc_data_for_day['nom'] = f"Hôtel Jour {day_num+1} à {city_name}"
        
        if activities_for_this_day_raw:
            logger.debug(f"Optimisation itinéraire pour {city_name} Jour {day_num+1} avec {len(activities_for_this_day_raw)} activités.")
            ordered_points = []
            if OSMNX_AVAILABLE and city_graph and use_astar_intra_city:
                 ordered_points = optimize_daily_activities_order_on_graph_django(city_graph, hotel_loc_data_for_day, activities_for_this_day_raw)
            else:
                ordered_points = optimize_daily_activities_order_geodesic_django(hotel_loc_data_for_day, activities_for_this_day_raw)
            
            final_points_with_names = []
            if ordered_points:
                for idx_p, p_data_dict in enumerate(ordered_points):
                    current_point_data = p_data_dict.copy()
                    if pd.isna(current_point_data.get('nom')) or not current_point_data.get('nom'):
                        is_h_check = current_point_data.get('type') == 'hotel' or current_point_data.get('price_per_night') is not None
                        current_point_data['nom'] = f"{'Hôtel' if is_h_check else 'Activité'} #{idx_p+1} (AutoNom)"
                    final_points_with_names.append(current_point_data)
            daily_ordered_points_data = final_points_with_names
            
            if OSMNX_AVAILABLE and city_graph and use_astar_intra_city and len(daily_ordered_points_data) > 1:
                for i in range(len(daily_ordered_points_data) - 1):
                    p1, p2 = daily_ordered_points_data[i], daily_ordered_points_data[i+1]
                    if pd.notna(p1.get('latitude')) and pd.notna(p1.get('longitude')) and pd.notna(p2.get('latitude')) and pd.notna(p2.get('longitude')):
                        segment_coords, _, _ = get_astar_route_coords_from_graph_django(city_graph, p1, p2)
                        if segment_coords: daily_route_segments_coords.append(segment_coords)
        elif hotel_loc_data_for_day:
            daily_ordered_points_data = [hotel_loc_data_for_day, hotel_loc_data_for_day]
        
        logger.debug(f"{city_name} Jour {day_num+1}: {len(daily_ordered_points_data)} points dans l'itinéraire.")
        recommendations["activites_par_jour_optimisees"].append(daily_ordered_points_data)
        recommendations["itineraire_segments_par_jour"].append(daily_route_segments_coords)
        
    logger.debug(f"--- Fin recommend_for_city_django pour {city_name} ---")
    return recommendations

# --- Fonction de Planification Principale (Version Complète) ---
def plan_trip_django(target_cities_list, total_budget_str, num_days_str, min_hotel_rating_str,
                     activity_preferences_str, activities_df_global, hotels_df_global,
                     city_coords_map_global, use_astar_for_planning):
    logger.info(f"--- Début plan_trip_django. Villes: {target_cities_list}, Budget: {total_budget_str}, Jours: {num_days_str} ---")
    trip_parameters_for_pdf = {}
    
    if activities_df_global is None or hotels_df_global is None or city_coords_map_global is None:
        logger.error("Données globales manquantes pour plan_trip_django.")
        return None, {"error": "Données de base non initialisées correctement."}, []
        
    try:
        total_budget_val = float(total_budget_str)
        num_days_val = int(num_days_str)
        min_hotel_rating_val = float(min_hotel_rating_str)
        activity_preferences_list = [pref.strip().capitalize() for pref in activity_preferences_str.split(",") if pref.strip()]
    except ValueError as e:
        logger.error(f"Erreur de conversion des paramètres dans plan_trip_django: {e}", exc_info=True)
        return None, {"error": f"Paramètres de voyage invalides: {e}"}, []

    if num_days_val <= 0:
        logger.error("Le nombre de jours doit être positif.")
        return None, {"error": "Le nombre de jours doit être positif."}, []

    trip_parameters_for_pdf = {
        "Villes demandées": ", ".join(target_cities_list),
        "Budget total": f"{total_budget_val:.2f} MAD",
        "Durée du voyage": f"{num_days_val} jours",
        "Rating hôtel minimum": str(min_hotel_rating_val),
        "Préférences d'activités": ", ".join(activity_preferences_list) if activity_preferences_list else "Aucune",
        "Optimisation intra-ville A*": "Activée" if use_astar_for_planning and OSMNX_AVAILABLE else "Désactivée/Non disponible"
    }

    normalized_target_cities = [normalize_city_name(city, CITY_NAME_MAPPING) for city in target_cities_list]
    
    valid_data_cities = set()
    if activities_df_global is not None and not activities_df_global.empty and "ville_normalisee" in activities_df_global.columns:
        valid_data_cities.update(activities_df_global['ville_normalisee'].unique())
    if hotels_df_global is not None and not hotels_df_global.empty and "ville_normalisee" in hotels_df_global.columns:
        valid_data_cities.update(hotels_df_global['ville_normalisee'].unique())
    
    final_target_cities = [city for city in normalized_target_cities if city and city in valid_data_cities]
    ignored_cities_input = [target_cities_list[i] for i, city_norm in enumerate(normalized_target_cities) if not city_norm or city_norm not in valid_data_cities]
    if ignored_cities_input:
        logger.warning(f"Villes ignorées (inconnues/sans données): {', '.join(ignored_cities_input)}")
        trip_parameters_for_pdf["Villes ignorées"] = ', '.join(ignored_cities_input)

    if not final_target_cities:
        logger.error("plan_trip_django: Aucune ville finale valide après filtrage des données.")
        return None, {"error": "Aucune des villes sélectionnées n'a de données disponibles ou n'est reconnue."}, []
    
    logger.debug(f"plan_trip_django: Villes finales ciblées avant optimisation de l'ordre: {final_target_cities}")
    ordered_cities = []
    if len(final_target_cities) == 1:
        ordered_cities = final_target_cities
    else:
        cities_for_path_optimization = [city for city in final_target_cities if city_coords_map_global.get(city) and pd.notna(city_coords_map_global[city].get('latitude')) and pd.notna(city_coords_map_global[city].get('longitude'))]
        cities_without_coords_for_path = [city for city in final_target_cities if city not in cities_for_path_optimization]
        if not cities_for_path_optimization:
            logger.warning("plan_trip_django: Aucune ville avec coordonnées valides pour l'optimisation du chemin. Utilisation de l'ordre d'entrée.")
            ordered_cities = final_target_cities
        elif len(cities_for_path_optimization) == 1:
            ordered_cities = cities_for_path_optimization + cities_without_coords_for_path
        elif len(cities_for_path_optimization) <= 7:
            ordered_cities_segment = find_optimal_path_permutations(cities_for_path_optimization, city_coords_map_global)
            ordered_cities = ordered_cities_segment + cities_without_coords_for_path
        else:
            start_city_path = cities_for_path_optimization[0]; other_cities_path = cities_for_path_optimization[1:]
            ordered_cities_segment = find_optimal_path_greedy(start_city_path, other_cities_path, city_coords_map_global)
            ordered_cities = ordered_cities_segment + cities_without_coords_for_path
    logger.debug(f"plan_trip_django: Ordre des villes après optimisation: {ordered_cities}")
    
    trip_parameters_for_pdf["Ordre de visite suggéré"] = ", ".join(ordered_cities) if ordered_cities else "N/A"
    if not ordered_cities: return None, {"error": "Impossible de déterminer un ordre de visite."}, []

    days_per_city_alloc = [0] * len(ordered_cities)
    if len(ordered_cities) > 0 and num_days_val > 0:
        base_days = num_days_val // len(ordered_cities)
        remainder_days = num_days_val % len(ordered_cities)
        for i in range(len(ordered_cities)): days_per_city_alloc[i] = base_days
        for i in range(remainder_days): 
            if i < len(days_per_city_alloc): days_per_city_alloc[i] += 1
        if num_days_val < len(ordered_cities):
            days_per_city_alloc = [0] * len(ordered_cities)
            for i in range(num_days_val): 
                if i < len(days_per_city_alloc): days_per_city_alloc[i] = 1
        elif any(d == 0 for d in days_per_city_alloc):
             days_per_city_alloc = [max(1,d) for d in days_per_city_alloc]
             current_total_days = sum(days_per_city_alloc)
             idx_reduce = len(days_per_city_alloc) - 1
             while current_total_days > num_days_val and idx_reduce >=0 :
                 if days_per_city_alloc[idx_reduce] > 1:
                     days_per_city_alloc[idx_reduce] -=1
                     current_total_days -=1
                 else: idx_reduce -=1
    logger.debug(f"plan_trip_django: Allocation finale des jours: {days_per_city_alloc}")
    if sum(days_per_city_alloc) == 0 and num_days_val > 0 and ordered_cities:
        logger.error(f"Erreur d'allocation des jours : Aucun jour alloué ! Fallback sur la première ville.")
        if ordered_cities: days_per_city_alloc[0] = num_days_val

    daily_total_budget = total_budget_val / num_days_val if num_days_val > 0 else total_budget_val
    daily_budget_hotel_info = daily_total_budget * 0.40 
    daily_budget_activities_calc = daily_total_budget * 0.30

    trip_plan_final = []
    for i, city_name_plan in enumerate(ordered_cities):
        num_days_in_this_city_plan = days_per_city_alloc[i] if i < len(days_per_city_alloc) else 0
        if num_days_in_this_city_plan == 0: 
            logger.info(f"Saut de {city_name_plan} car 0 jours alloués.")
            continue

        budget_activities_total_for_city_stay = daily_budget_activities_calc * num_days_in_this_city_plan
        
        city_recommendation = recommend_for_city_django(
            city_name_plan, hotels_df_global, activities_df_global,
            daily_budget_hotel_info, budget_activities_total_for_city_stay,
            min_hotel_rating_val, activity_preferences_list,
            num_hotel_recs=1, num_activity_recs_per_day=3,
            num_days_in_city=num_days_in_this_city_plan,
            use_astar_intra_city=use_astar_for_planning
        )
        
        city_recommendation["jours_alloues"] = num_days_in_this_city_plan
        city_recommendation["budget_hotel_quotidien_info"] = daily_budget_hotel_info
        city_recommendation["budget_activites_sejour_total"] = budget_activities_total_for_city_stay
        trip_plan_final.append(city_recommendation)
    
    if not trip_plan_final:
        logger.warning("plan_trip_django: trip_plan_final est vide à la fin. Aucun plan généré.")
        return None, {"error": "Aucun plan n'a pu être généré pour les villes et jours spécifiés."}, ordered_cities

    logger.info(f"Fin de plan_trip_django. {len(trip_plan_final)} éléments dans trip_plan_final.")
    return trip_plan_final, trip_parameters_for_pdf, ordered_cities