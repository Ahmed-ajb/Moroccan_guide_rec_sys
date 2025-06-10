# trip_planner_project/trip_planner_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static # Important pour MEDIA

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', include('planner.urls')),
]

# Servir les fichiers statiques et media en mode DEBUG
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT if hasattr(settings, 'STATIC_ROOT') and settings.STATIC_ROOT else None)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) # AJOUTER CETTE LIGNE