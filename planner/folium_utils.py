# planner/folium_utils.py
import folium
from folium import plugins
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

DAILY_ROUTE_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightred',
                      'beige', 'darkblue', 'darkgreen', 'cadetblue', 'lightgray', 'black',
                      'pink', 'brown', 'cyan', 'magenta']

ACTIVITY_TYPE_ICONS_MAP = {
    "Culturelle": "landmark", "Aventure": "mountain-sun", "Gastronomique": "utensils",
    "Historique": "monument", "Loisir": "umbrella-beach", "Bien-être": "spa",
    "Marché": "store", "Nature": "tree", "Artisanat": "palette", "Sport/Loisir": "person-biking",
    "Sport/Aventure": "person-skiing", "Shopping": "shopping-bag", "Religieux": "place-of-worship",
    "Défaut": "map-pin", "hotel": "bed"
}

def generate_trip_map_folium_django(ordered_cities, trip_plan_data, city_coords_map_global, num_days_total_trip, use_astar_on_map=True, osmnx_available_flag=False):
    if not ordered_cities:
        logger.warning("generate_trip_map_folium_django: Pas de villes ordonnées fournies.")
        return None

    latitudes = [city_coords_map_global[city]["latitude"] for city in ordered_cities if city_coords_map_global.get(city) and pd.notna(city_coords_map_global[city].get("latitude"))]
    longitudes = [city_coords_map_global[city]["longitude"] for city in ordered_cities if city_coords_map_global.get(city) and pd.notna(city_coords_map_global[city].get("longitude"))]

    if not latitudes or not longitudes:
        logger.info("generate_trip_map_folium_django: Pas de coordonnées valides pour centrer la carte, utilisation du centre par défaut du Maroc.")
        map_center, zoom_start = [31.7917, -7.0926], 5
    else:
        map_center = [np.mean(latitudes), np.mean(longitudes)]
        if len(latitudes) > 1:
            zoom_lat_range = abs(max(latitudes) - min(latitudes)) if latitudes else 0
            zoom_lon_range = abs(max(longitudes) - min(longitudes)) if longitudes else 0
            if zoom_lat_range > 8 or zoom_lon_range > 8: zoom_start = 5
            elif zoom_lat_range > 4 or zoom_lon_range > 4: zoom_start = 6
            elif zoom_lat_range > 1.5 or zoom_lon_range > 1.5: zoom_start = 7
            elif zoom_lat_range > 0.5 or zoom_lon_range > 0.5: zoom_start = 8
            else: zoom_start = 9
        else: zoom_start = 10

    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")

    city_path_coords_inter = []
    for city_idx, city_name_inter in enumerate(ordered_cities):
        if city_coords_map_global.get(city_name_inter) and \
           pd.notna(city_coords_map_global[city_name_inter].get("latitude")) and \
           pd.notna(city_coords_map_global[city_name_inter].get("longitude")):
            coords_inter = (city_coords_map_global[city_name_inter]["latitude"], city_coords_map_global[city_name_inter]["longitude"])
            city_path_coords_inter.append(coords_inter)
            folium.Marker(
                location=coords_inter,
                popup=f"<b>Étape {city_idx+1}: {city_name_inter}</b>",
                tooltip=f"Ville: {city_name_inter}",
                icon=folium.Icon(color="darkblue", icon_color="white", icon="map-marker", prefix="fa")
            ).add_to(m)
    if len(city_path_coords_inter) > 1:
        folium.PolyLine(city_path_coords_inter, color="#007bff", weight=3, opacity=0.7, dash_array='8, 4', tooltip="Trajet Principal Inter-Villes").add_to(m)

    overall_hotels_fg = folium.FeatureGroup(name="🏨 Hôtels (Séjour)", show=True)
    poi_markers_fg = folium.FeatureGroup(name="📍 Points d'Intérêt (Journalier)", show=True)
    
    show_astar_default = use_astar_on_map and osmnx_available_flag
    astar_routes_fg = folium.FeatureGroup(name="🚗 Itinéraires A* Détaillés", show=show_astar_default)
    geodesic_routes_fg = folium.FeatureGroup(name="🚶 Itinéraires Géodésiques", show=not show_astar_default)

    current_trip_day_map = 0
    placed_hotel_markers_coords = set()
    placed_poi_markers_coords_by_day = {}
    # tooltip_default_text = "Point d'intérêt" # Déjà implicite dans point_data.get

    for city_plan_item in trip_plan_data:
        city_name_on_map = city_plan_item['ville']
        hotel_list = city_plan_item.get("hotel", [])
        hotel_info = hotel_list[0] if isinstance(hotel_list, list) and len(hotel_list) > 0 else None

        if hotel_info and pd.notna(hotel_info.get("latitude")) and pd.notna(hotel_info.get("longitude")):
            hotel_coords = (hotel_info["latitude"], hotel_info["longitude"])
            if hotel_coords not in placed_hotel_markers_coords:
                popup_html_hotel_map_display = f"""
                    <b>{hotel_info.get('name', 'Hôtel Principal')}</b><br>
                    Ville: {city_name_on_map}<br>
                    Rating: {hotel_info.get('rating', 'N/A')} ⭐<br>
                    Prix/nuit: {hotel_info.get('price_per_night', 'N/A')} MAD<br>
                """
                if hotel_info.get('booking_link') and hotel_info['booking_link'] != 'N/A':
                    popup_html_hotel_map_display += f"<a href='{hotel_info['booking_link']}' target='_blank'>Réserver</a>"

                folium.Marker(
                    location=hotel_coords,
                    popup=folium.Popup(popup_html_hotel_map_display, max_width=300),
                    tooltip=f"Hôtel Principal: {hotel_info.get('name', 'N/A')} à {city_name_on_map}",
                    icon=folium.Icon(color="green", icon_color="white", icon="bed", prefix="fa")
                ).add_to(overall_hotels_fg)
                placed_hotel_markers_coords.add(hotel_coords)

        daily_ordered_point_lists = city_plan_item.get('activites_par_jour_optimisees', [])
        daily_astar_segment_lists = city_plan_item.get('itineraire_segments_par_jour', [])

        for day_idx_in_city, ordered_points_this_day in enumerate(daily_ordered_point_lists):
            current_trip_day_map += 1
            if not ordered_points_this_day:
                logger.debug(f"Jour {current_trip_day_map}: Pas de points ordonnés pour ce jour à {city_name_on_map}.")
                continue

            for point_order_in_day, point_data in enumerate(ordered_points_this_day):
                if pd.notna(point_data.get("latitude")) and pd.notna(point_data.get("longitude")):
                    point_coords = (point_data["latitude"], point_data["longitude"])
                    is_hotel_point = point_data.get('type') == 'hotel' or point_data.get('price_per_night') is not None

                    if is_hotel_point and point_coords in placed_hotel_markers_coords:
                        if point_order_in_day == 0 or point_order_in_day == len(ordered_points_this_day) - 1:
                            continue

                    marker_key = (round(point_coords[0], 5), round(point_coords[1], 5), current_trip_day_map)
                    if not is_hotel_point and marker_key in placed_poi_markers_coords_by_day:
                        continue

                    marker_icon_name = ACTIVITY_TYPE_ICONS_MAP.get(point_data.get('type', "Défaut").capitalize(), ACTIVITY_TYPE_ICONS_MAP["Défaut"])
                    marker_color = "purple"
                    if is_hotel_point :
                        marker_color = "darkgreen"
                        marker_icon_name = ACTIVITY_TYPE_ICONS_MAP["hotel"]

                    popup_html_point_display = f"<b>{point_data.get('nom', 'Point d_Intérêt')}</b> (Jour {current_trip_day_map})<br>"
                    popup_html_point_display += f"Type: {point_data.get('type', 'N/A')}<br>"
                    if not is_hotel_point:
                        budget_field_map = 'tarif_recommande' if 'tarif_recommande' in point_data and pd.notna(point_data['tarif_recommande']) else 'budget_estime'
                        budget_val = point_data.get(budget_field_map)
                        budget_display_val_map = f"{budget_val:.0f} MAD" if pd.notna(budget_val) else "N/A"
                        popup_html_point_display += f"Budget est.: {budget_display_val_map}<br>"
                        popup_html_point_display += f"Durée est.: {point_data.get('duree_estimee', 'N/A')}<br>"
                    
                    # CORRECTION ICI : Utiliser des guillemets doubles pour la chaîne par défaut si elle contient une apostrophe
                    tooltip_default_name = "Point d'intérêt" # ou "Point d interet" pour éviter l'apostrophe
                    tooltip_text = point_data.get('nom', tooltip_default_name)

                    folium.Marker(
                        location=point_coords,
                        popup=folium.Popup(popup_html_point_display, max_width=250),
                        tooltip=f"Jour {current_trip_day_map}: {tooltip_text}",
                        icon=folium.Icon(color=marker_color, icon=marker_icon_name, prefix="fa")
                    ).add_to(poi_markers_fg)

                    if not is_hotel_point:
                        placed_poi_markers_coords_by_day[marker_key] = True

            route_color = DAILY_ROUTE_COLORS[(current_trip_day_map - 1) % len(DAILY_ROUTE_COLORS)]

            if use_astar_on_map and osmnx_available_flag and \
               day_idx_in_city < len(daily_astar_segment_lists) and daily_astar_segment_lists[day_idx_in_city]:
                for segment_coords_astar in daily_astar_segment_lists[day_idx_in_city]:
                    if segment_coords_astar and len(segment_coords_astar) > 1:
                        folium.PolyLine(
                            locations=segment_coords_astar, color=route_color, weight=3.5, opacity=0.75,
                            tooltip=f"Itinéraire A* Jour {current_trip_day_map}"
                        ).add_to(astar_routes_fg)
            elif len(ordered_points_this_day) > 1:
                valid_points_for_geodesic = [p for p in ordered_points_this_day if pd.notna(p.get('latitude')) and pd.notna(p.get('longitude'))]
                if len(valid_points_for_geodesic) > 1:
                    for k_geo in range(len(valid_points_for_geodesic) - 1):
                        start_p_geo = valid_points_for_geodesic[k_geo]
                        end_p_geo = valid_points_for_geodesic[k_geo+1]
                        start_p_name_geo = start_p_geo.get('nom', 'Point')
                        end_p_name_geo = end_p_geo.get('nom', 'Point Suivant')
                        folium.PolyLine(
                            locations=[(start_p_geo['latitude'], start_p_geo['longitude']), (end_p_geo['latitude'], end_p_geo['longitude'])],
                            color=route_color, weight=2.5, opacity=0.6, dash_array='5,5',
                            tooltip=f"Géod. Jour {current_trip_day_map}: {start_p_name_geo} → {end_p_name_geo}"
                        ).add_to(geodesic_routes_fg)

    overall_hotels_fg.add_to(m)
    poi_markers_fg.add_to(m)
    if use_astar_on_map and osmnx_available_flag:
        astar_routes_fg.add_to(m)
    geodesic_routes_fg.add_to(m) # Toujours ajouter, son 'show' initial est conditionnel

    folium.LayerControl(collapsed=False).add_to(m)
    plugins.MiniMap(tile_layer="OpenStreetMap", zoom_level_offset=-5, position="bottomright").add_to(m)
    plugins.Fullscreen(position="topright", title="Plein écran", title_cancel="Quitter le plein écran", force_separate_button=True).add_to(m)
    plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    logger.info("Carte Folium générée avec succès pour Django.")
    return m