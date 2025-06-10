# planner/reportlab_utils.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Assurez-vous que les styles de paragraphe sont bien définis
# et que la gestion du fallback de police est en place.

def generate_schedule_content_objects_django(trip_plan_data, num_total_days):
    schedule_story_pdf = []
    schedule_markdown_list = ["### 📅 Emploi du Temps Suggéré\n"] # Pour l'affichage HTML

    styles = getSampleStyleSheet()
    default_font_schedule = 'Helvetica'
    try:
        Paragraph("test", ParagraphStyle('test_bold', fontName='Helvetica-Bold'))
    except Exception:
        default_font_schedule = 'Times-Roman'
        logger.warning("Police Helvetica non trouvée pour l'emploi du temps PDF (schedule), utilisation de Times-Roman.")

    # Définition des styles de paragraphe pour l'emploi du temps
    schedule_normal_style = ParagraphStyle('ScheduleNormal', parent=styles['Normal'], fontName=default_font_schedule, fontSize=9, leading=11)
    schedule_bold_style = ParagraphStyle('ScheduleBold', parent=styles['Normal'], fontName=f'{default_font_schedule}-Bold', fontSize=10, leading=12, spaceBefore=6)
    schedule_h3_style = ParagraphStyle('ScheduleH3', parent=styles['h3'], fontName=f'{default_font_schedule}-Bold', fontSize=11, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#222255"))

    schedule_story_pdf.append(Paragraph("<b>Emploi du Temps Suggéré</b>", schedule_h3_style))
    schedule_story_pdf.append(Spacer(1, 0.1*inch))

    current_day_counter = 1

    for city_plan_item_schedule in trip_plan_data:
        city_name_schedule = city_plan_item_schedule['ville']
        daily_activity_lists_for_schedule = city_plan_item_schedule.get('activites_par_jour_optimisees', [])

        for day_idx, ordered_points_this_day in enumerate(daily_activity_lists_for_schedule):
            if current_day_counter > num_total_days: break

            schedule_markdown_list.append(f"**Jour {current_day_counter} : {city_name_schedule}**\n")
            schedule_story_pdf.append(Paragraph(f"<b>Jour {current_day_counter} : {city_name_schedule}</b>", schedule_bold_style))

            current_day_hotel_name = "Hôtel non spécifié"
            # L'hôtel est le premier point si le trajet part de l'hôtel
            if ordered_points_this_day and \
               (ordered_points_this_day[0].get('type') == 'hotel' or ordered_points_this_day[0].get('price_per_night') is not None):
                current_day_hotel_name = ordered_points_this_day[0].get('name', "Hôtel")
            elif city_plan_item_schedule.get("hotel") and city_plan_item_schedule["hotel"]: # Fallback
                 current_day_hotel_name = city_plan_item_schedule["hotel"][0].get('name', "Hôtel recommandé")

            schedule_markdown_list.append(f"- 🏨 Séjour à : {current_day_hotel_name}\n")
            schedule_story_pdf.append(Paragraph(f"• Séjour à : {current_day_hotel_name}", schedule_normal_style))

            activities_to_list_for_day = []
            if ordered_points_this_day:
                # Exclure le point de départ et d'arrivée s'il s'agit de l'hôtel et qu'il y a d'autres activités
                temp_list = list(ordered_points_this_day) # Copie pour modification
                if len(temp_list) > 1 and (temp_list[0].get('type') == 'hotel' or temp_list[0].get('price_per_night') is not None):
                    temp_list.pop(0)
                if len(temp_list) > 1 and (temp_list[-1].get('type') == 'hotel' or temp_list[-1].get('price_per_night') is not None):
                    temp_list.pop(-1)
                # Filtrer pour ne garder que les vraies activités
                activities_to_list_for_day = [act for act in temp_list if not (act.get('type') == 'hotel' or act.get('price_per_night') is not None)]

            is_schedule_rest_day = False
            if len(ordered_points_this_day) == 2 and \
               (ordered_points_this_day[0].get('type') == 'hotel' or ordered_points_this_day[0].get('price_per_night') is not None) and \
               (ordered_points_this_day[1].get('type') == 'hotel' or ordered_points_this_day[1].get('price_per_night') is not None) and \
                ordered_points_this_day[0].get('name') == ordered_points_this_day[1].get('name'):
                is_schedule_rest_day = True


            if activities_to_list_for_day:
                for act_schedule in activities_to_list_for_day:
                    activity_name_schedule = act_schedule.get('nom', 'Activité')
                    activity_type_schedule = act_schedule.get('type', 'N/A')
                    activity_duration_schedule = act_schedule.get('duree_estimee', 'N/A')
                    schedule_markdown_list.append(f"  - 🎯 {activity_name_schedule} ({activity_type_schedule}, Durée: {activity_duration_schedule})\n")
                    schedule_story_pdf.append(Paragraph(f"  • {activity_name_schedule} ({activity_type_schedule}, Durée: {activity_duration_schedule})", schedule_normal_style))
            elif is_schedule_rest_day or not ordered_points_this_day: # Si c'est un jour de repos ou pas de points
                schedule_markdown_list.append("  - Exploration libre / Détente / Repos à l'hôtel\n")
                schedule_story_pdf.append(Paragraph("  • Exploration libre / Détente / Repos à l'hôtel", schedule_normal_style))
            # else: # Cas où il y a des points mais après filtrage il n'y a plus d'activités (ex: Hotel -> Hotel différent)
                # On pourrait ajouter un message ici si on veut être plus précis

            schedule_markdown_list.append("\n")
            schedule_story_pdf.append(Spacer(1, 0.05*inch))
            current_day_counter += 1

    return "".join(schedule_markdown_list), schedule_story_pdf


def generate_trip_pdf_django(output_buffer, trip_plan_data, trip_params, schedule_pdf_content_objects):
    doc = SimpleDocTemplate(output_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = getSampleStyleSheet()
    story = []
    default_font = 'Helvetica'
    try:
        Paragraph("test", ParagraphStyle('test_bold', fontName='Helvetica-Bold'))
        Paragraph("test", ParagraphStyle('test_oblique', fontName='Helvetica-Oblique')) # Vérifiez si Oblique est utilisé et disponible
    except Exception:
        logger.warning("Polices Helvetica (Bold/Oblique) non trouvées pour le PDF principal, utilisation de Times-Roman.")
        default_font = 'Times-Roman'

    # Définition des styles de paragraphe pour le PDF principal
    link_style = ParagraphStyle('LinkStyle', parent=styles['Normal'], textColor=colors.blue, fontName=default_font)
    normal_style_utf8 = ParagraphStyle('NormalUTF8', parent=styles['Normal'], fontName=default_font, leading=14, fontSize=10)
    bold_normal_style_utf8 = ParagraphStyle('BoldNormalUTF8', parent=normal_style_utf8, fontName=f'{default_font}-Bold')
    title_style_utf8 = ParagraphStyle('TitleUTF8', parent=styles['h1'], fontName=f'{default_font}-Bold', fontSize=18, spaceAfter=12, alignment=1)
    h2_style_utf8 = ParagraphStyle('H2UTF8', parent=styles['h2'], fontName=f'{default_font}-Bold', fontSize=14, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#333366"))
    h3_style_utf8 = ParagraphStyle('H3UTF8', parent=styles['h3'], fontName=f'{default_font}-Bold', fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#444477"))
    italic_style_utf8 = ParagraphStyle('ItalicUTF8', parent=styles['Italic'], fontName=f'{default_font}-Oblique' if default_font == 'Helvetica' else 'Times-Italic', fontSize=9, alignment=2 )

    story.append(Paragraph("Plan de Voyage Recommandé au Maroc", title_style_utf8))
    story.append(Paragraph(f"<i>Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>", italic_style_utf8))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Paramètres du Voyage</b>", h2_style_utf8))
    param_data = []
    for key, value in trip_params.items():
        param_data.append([Paragraph(f"<b>{key}</b>", normal_style_utf8), Paragraph(str(value), normal_style_utf8)])
    param_table = Table(param_data, colWidths=[2.5*inch, 4*inch])
    param_table.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('FONTNAME', (0,0), (0,-1), f'{default_font}-Bold'),
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 0.3*inch))

    if trip_params.get("Ordre de visite suggéré") and len(trip_params["Ordre de visite suggéré"].split(',')) > 1 :
        story.append(Paragraph(f"<b>Ordre de visite suggéré :</b> {trip_params['Ordre de visite suggéré']}", bold_normal_style_utf8))
        story.append(Spacer(1, 0.2*inch))

    current_pdf_day_counter = 1
    total_days_in_trip = int(trip_params.get("Durée du voyage", "0 jours").split(" ")[0])

    for i, city_data in enumerate(trip_plan_data):
        if city_data.get("jours_alloues", 0) == 0: continue
        story.append(Paragraph(f"Plan pour {city_data['ville']} ({city_data['jours_alloues']} jour(s))", h2_style_utf8))
        story.append(Spacer(1, 0.1*inch))

        # Affichage de l'hôtel
        hotel_list_pdf = city_data.get("hotel", [])
        hotel_info = hotel_list_pdf[0] if isinstance(hotel_list_pdf, list) and len(hotel_list_pdf) > 0 else None
        if hotel_info:
            story.append(Paragraph("Hôtel Suggéré :", h3_style_utf8))
            price_val = hotel_info.get('price_per_night')
            price_info = f"Prix: {price_val:.2f} MAD" if pd.notna(price_val) else "Prix: N/A"
            hotel_name_rating_price = f"{hotel_info.get('name', 'N/A')} (Rating: {hotel_info.get('rating', 'N/A')}, {price_info})"
            story.append(Paragraph(f"• {hotel_name_rating_price}", normal_style_utf8))
            if hotel_info.get('booking_link') and hotel_info['booking_link'] != 'N/A':
                story.append(Paragraph(f"<a href='{hotel_info['booking_link']}'>Lien de réservation</a>", link_style))
            story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("<i>Aucun hôtel principal suggéré pour cette ville.</i>", normal_style_utf8))
            story.append(Spacer(1, 0.1*inch))

        # Itinéraire journalier
        daily_activities_for_pdf = city_data.get("activites_par_jour_optimisees", [])
        if any(day_plan for day_plan in daily_activities_for_pdf if day_plan):
            story.append(Paragraph("Itinéraire Journalier Suggéré :", h3_style_utf8))

            for day_idx, points_ordered_this_day in enumerate(daily_activities_for_pdf):
                if not points_ordered_this_day or current_pdf_day_counter > total_days_in_trip : continue

                story.append(Paragraph(f"<b>Jour {current_pdf_day_counter} (Ville: {city_data['ville']}):</b>", bold_normal_style_utf8))

                is_pdf_rest_day = False
                if len(points_ordered_this_day) == 2 and \
                   (points_ordered_this_day[0].get('type') == 'hotel' or points_ordered_this_day[0].get('price_per_night') is not None) and \
                   (points_ordered_this_day[1].get('type') == 'hotel' or points_ordered_this_day[1].get('price_per_night') is not None) and \
                    points_ordered_this_day[0].get('name') == points_ordered_this_day[1].get('name'):
                    is_pdf_rest_day = True

                if is_pdf_rest_day:
                    story.append(Paragraph(f"  • Repos/Exploration libre depuis : {points_ordered_this_day[0].get('name', 'Hôtel')}", normal_style_utf8))
                elif points_ordered_this_day:
                    activities_table_data = [[Paragraph("<b>#</b>", bold_normal_style_utf8), Paragraph("<b>Point d'Intérêt / Étape</b>", bold_normal_style_utf8), Paragraph("<b>Type</b>", bold_normal_style_utf8), Paragraph("<b>Budget (MAD)</b>", bold_normal_style_utf8), Paragraph("<b>Durée</b>", bold_normal_style_utf8)]]

                    for k, point_in_route in enumerate(points_ordered_this_day):
                        is_hotel_point_pdf = point_in_route.get('type') == 'hotel' or point_in_route.get('price_per_night') is not None

                        point_name = point_in_route.get('nom', 'N/A')
                        point_type_display = "Hôtel (Étape)" if is_hotel_point_pdf else point_in_route.get('type', 'N/A')
                        budget_val_str, duree_val_str = "", ""

                        if not is_hotel_point_pdf:
                            budget_field_pdf = 'tarif_recommande' if 'tarif_recommande' in point_in_route and pd.notna(point_in_route['tarif_recommande']) else 'budget_estime'
                            budget_num = point_in_route.get(budget_field_pdf)
                            budget_val_str = f"{budget_num:.0f}" if pd.notna(budget_num) else "N/A"
                            duree_val_str = point_in_route.get('duree_estimee', 'N/A')

                        activities_table_data.append([
                            Paragraph(str(k+1), normal_style_utf8),
                            Paragraph(point_name, normal_style_utf8),
                            Paragraph(point_type_display, normal_style_utf8),
                            Paragraph(budget_val_str, normal_style_utf8),
                            Paragraph(duree_val_str, normal_style_utf8)
                        ])

                    act_table = Table(activities_table_data, colWidths=[0.4*inch, 2.3*inch, 1.0*inch, 1*inch, 1.8*inch])
                    act_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightcyan), ('TEXTCOLOR',(0,0),(-1,0),colors.black),
                        ('ALIGN',(0,0),(-1,-1),'LEFT'), ('VALIGN',(0,0),(-1,-1),'TOP'),
                        ('FONTNAME', (0,0), (-1,0), f'{default_font}-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8), ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                        ('LEFTPADDING', (0,0), (-1,-1), 4), ('RIGHTPADDING', (0,0), (-1,-1), 4),
                    ]))
                    story.append(act_table)
                story.append(Spacer(1, 0.1*inch))
                current_pdf_day_counter +=1

            budget_spent_act = city_data.get('budget_activites_depense', 0)
            budget_total_act_stay = city_data.get('budget_activites_sejour_total', 0)
            story.append(Paragraph(f"<i>Budget total activités estimé pour le séjour à {city_data['ville']}: {budget_spent_act:.0f} / {budget_total_act_stay:.0f} MAD</i>", styles['Italic'])) # styles['Italic'] est standard
            story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("<i>Aucune activité spécifique suggérée pour les jours dans cette ville.</i>", normal_style_utf8))
            story.append(Spacer(1, 0.1*inch))
            num_jours_ville_pdf = city_data.get("jours_alloues", 0)
            days_already_counted_for_city = sum(1 for day_plan in daily_activities_for_pdf if day_plan)
            remaining_days_in_city_to_account_for = num_jours_ville_pdf - days_already_counted_for_city
            current_pdf_day_counter += max(0, remaining_days_in_city_to_account_for)


        if i < len(trip_plan_data) - 1 and current_pdf_day_counter <= total_days_in_trip:
            story.append(PageBreak())

    if schedule_pdf_content_objects: # C'est une liste d'objets ReportLab
        if story and not isinstance(story[-1], PageBreak) and current_pdf_day_counter <= total_days_in_trip :
             story.append(PageBreak())
        story.extend(schedule_pdf_content_objects) # Ajouter directement les objets

    try:
        doc.build(story)
        logger.info(f"Le fichier PDF a été généré avec succès dans le buffer.")
    except Exception as e:
        logger.error(f"Erreur lors de la génération du PDF : {e}", exc_info=True)
        raise # Relaisser l'exception pour que la vue la gère