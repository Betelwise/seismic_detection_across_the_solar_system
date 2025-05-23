# src/config.py
settings = {
    'moon': {
        'low_freq_point': 0.2,
        'high_freq_point': 1.0, # Make sure this is a float
        'frequence_window': 0.2,
        'sta_len': 100,
        'lta_len': 1000,
        'thr_on': 3.0, # Make sure this is a float
        'resample': False,
        'df_resample': 6.625, # Add sampling rate if resampling
        'clipping_std_factor': 26, # For clipping outliers
    },
    'mars': {
        'low_freq_point': 0.6,
        'high_freq_point': 4.0, # Make sure this is a float
        'frequence_window': 0.5,
        'sta_len': 20,
        'lta_len': 80,
        'thr_on': 2.1,
        'resample': True,
        'df_resample': 6.625, # Example, adjust as needed
        'clipping_std_factor': 26,
    }
}

def get_event_type_from_path(filepath):
    if 'lunar' in filepath or 'moon' in filepath.lower(): # More robust check
        return 'moon'
    elif 'mars' in filepath:
        return 'mars'
    return 'moon' # Default

def get_settings(event_type):
    return settings.get(event_type, settings['moon'])