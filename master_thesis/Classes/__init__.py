# constant dictionary to rename fields into dwc standard
GROUND_TRUTH_RENAME_DICT = {
    'Barcode': 'catalogNumber',
    'Family': 'family',
    'taxon': 'scientificName',
    'genus': 'genus',
    'spepi': 'specificEpithet',
    'country': 'country',
    'collect_date_begin': 'eventDate',
    'collect_date_end': 'eventDateEnd',
    'collector': 'recordedBy',
    'collect_number - ': 'collector_number_ocr_results',
    'locality': 'locality',
    'geo - position': 'geolocation_ocr_results',
    'geo - radius': 'geolocation_ocr_results',
    'geo - description': 'geolocation_ocr_results'
}