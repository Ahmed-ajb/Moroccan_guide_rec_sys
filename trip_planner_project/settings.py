# trip_planner_project/trip_planner_project/settings.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Chemins pour les données et caches (utilisés par planner/utils.py)
DATA_DIR = BASE_DIR / 'planner' / 'data'
MODEL_CACHE_DIR_DJANGO = BASE_DIR / "recommender_models_cache_django"
GRAPHS_CACHE_DIR_DJANGO = BASE_DIR / "city_graphs_cache_django"

SECRET_KEY = 'django-insecure-!!REMPLACEZ-MOI-AVEC-UNE-VRAIE-CLE-SECRETE!!'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'planner',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'trip_planner_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'trip_planner_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

LANGUAGE_CODE = 'fr-fr'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
# STATICFILES_DIRS = [BASE_DIR / "static_global"]

# --- Configuration pour les fichiers MEDIA (uploads utilisateur) ---
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'mediafiles' # Créez ce dossier 'mediafiles' à la racine de votre projet

# Assurer que le dossier MEDIA_ROOT existe (Django ne le crée pas automatiquement)
if not os.path.exists(MEDIA_ROOT):
    try:
        os.makedirs(MEDIA_ROOT)
        print(f"Dossier MEDIA_ROOT créé à: {MEDIA_ROOT}")
    except OSError as e:
        print(f"ERREUR: Impossible de créer le dossier MEDIA_ROOT ({MEDIA_ROOT}): {e}")
# --- Fin Configuration MEDIA ---

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'

LOGGING = {
    'version': 1, 'disable_existing_loggers': False,
    'formatters': { 'simple': {'format': '{levelname} {asctime} {module} {message}', 'style': '{',}, },
    'handlers': { 'console': {'class': 'logging.StreamHandler', 'formatter': 'simple',}, },
    'root': {'handlers': ['console'], 'level': 'INFO',},
    'loggers': {
        'django': {'handlers': ['console'], 'level': 'INFO', 'propagate': False,},
        'planner': {'handlers': ['console'], 'level': 'DEBUG', 'propagate': True,},
    },
}