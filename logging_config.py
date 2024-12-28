"""Configuration for application logging with file and console handlers."""
import logging
import logging.config
import os
from datetime import datetime

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'app_file': {
            'class': 'logging.FileHandler',
            'filename': f'logs/app_{datetime.now().strftime("%Y-%m-%d")}.log',
            'formatter': 'standard'
        },
        'library_file': {
            'class': 'logging.FileHandler',
            'filename': f'logs/libraries_{datetime.now().strftime("%Y-%m-%d")}.log',
            'formatter': 'standard'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console'],
            'level': 'INFO',
        },
        'BiodiversityApp': {
            'handlers': ['app_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'streamlit': {
            'handlers': ['library_file', 'console'],
            'level': 'INFO',
            'propagate': False
        },
        'google': {
            'handlers': ['library_file', 'console'],
            'level': 'INFO',
            'propagate': False
        },
        'vertexai': {
            'handlers': ['library_file', 'console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

def setup_logging():
    """Initialize application logging configuration using predefined settings."""
    logging.config.dictConfig(LOGGING_CONFIG)
