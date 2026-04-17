#DEFAULT
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import datetime
import os
import re
from typing import List, Dict, Optional, Union
#3RD PARTY
from countrycode import countrycode
import difflib
from fuzzywuzzy.fuzz import partial_ratio
import jaro
from Levenshtein import distance as ldist
import numpy as np
import pandas as pd
import pycountry
from pygbif import species as gbif_species

@dataclass
class TranscriptionResult: # Class representing a single transcription result
    tool_name: str
    taxon: str = ""
    scientific_name: str = ""
    genus: str = ""
    specific_epithet: str = ""
    infraspecific_epithet: str = ""
    scientific_name_authorship: str = ""
    family: str = ""
    collector: str = ""
    associated_collectors: str = ""
    collection_date: str = ""
    day: str = ""
    month: str = ""
    year: str = ""
    locality: str = ""
    habitat: str = ""
    minimum_elevation: str = ""
    maximum_elevation: str = ""
    county: str = ""
    state_province: str = ""
    country: str = ""
    specimen_description: str = ""

@dataclass
class FieldQualityScore: # Data class resembling the evaluation result for a specific field 
    field_name: str
    exact_match: bool
    composite_score: float # Final evaluation score for the field that may incorporate multiple aspects such as validity and string similarity
    similarity_score: float # Secondary score that can differ from composite_score, e.g., it is based on pure string similarity
    errors: List[str] # List for errors and other relevant information that represents a log of the evaluation process 

class FieldEvaluator(ABC): # Abstaract base class for field-specific evaluators that evaluates the given transcription against ground truth
    @abstractmethod
    def evaluate(self, transcription: str, ground_truth: str, transcription_record: Optional[Dict], ground_truth_record: Optional[Dict] = None) -> FieldQualityScore:
        pass

    def generic_levenshtein_distance(self, s1: str, s2: str) -> float:
        # Adapted from source: https://openreview.net/pdf?id=kXYl48LfTu
        if not s1 or not s2 or s1.strip() == "nan" or s2.strip() == "nan":
            return 0.0 if s1 == s2 else 1.0
        return 2 * ldist(s1, s2) / ((len(s1) + len(s2)) + ldist(s1, s2))
    
    def jaro_winkler_distance(self, s1: str, s2: str) -> float:
        if not s1 or not s2 or s1.strip() == "nan" or s2.strip() == "nan":
            return 0.0 if s1 == s2 else 1.0
        return jaro.jaro_winkler_metric(s1, s2)
    
    def token_sort_ratio(self, s1: str, s2: str) -> float: # Sorted levenshtein distance
        tokens1 = sorted(s1.split())
        tokens2 = sorted(s2.split())
        sorted_s1 = ' '.join(tokens1)
        sorted_s2 = ' '.join(tokens2)
        return self.generic_levenshtein_distance(sorted_s1, sorted_s2)
    
class TaxonEvaluator(FieldEvaluator): # Evaluator class for taxon-related information
    def evaluate(self, transcription: str, ground_truth: str, transcription_record: Optional[Dict] = None, ground_truth_record: Optional[Dict] = None) -> FieldQualityScore:
        errors = []

        transcription_clean = transcription.strip() # Removal of unnecessary blanks
        ground_truth_clean = ground_truth.strip()

        exact_match = transcription_clean == ground_truth_clean
        exact_match = exact_match or (transcription == "nan" and "indet" in ground_truth.lower()) # undetermined taxa in ground truth and no taxa information in transcription

        transcription_without_abbr = self._remove_abbreviations(transcription_clean.lower())
        ground_truth_without_abbr = self._remove_abbreviations(ground_truth_clean.lower())
        normalized_ls = 1.0 - self.generic_levenshtein_distance(transcription_without_abbr, ground_truth_without_abbr)
        normalized_sort_ratio = 1.0 - self.token_sort_ratio(transcription_without_abbr, ground_truth_without_abbr)
        similarity_score = max(normalized_ls, normalized_sort_ratio) # normalized_sort_ratio seems to be the same as normalized_ls in most cases (same results)

        if transcription_clean == "" or transcription_clean == "nan" or ground_truth_clean == "" or ground_truth_clean == "nan":
            taxonomic_score = 0.0
        else:
            taxonomic_score = self._taxonomic_comparison(transcription_clean, ground_truth_clean)

        if pd.notna(transcription_record['family']) and pd.notna(ground_truth_record['family']):
            family_match = transcription_record['family'].strip().lower() == ground_truth_record['family'].strip().lower()
        else:
            family_match = False

        if exact_match or ground_truth == "":
            composite_score = 1.0
        else:
            composite_score = taxonomic_score if taxonomic_score != 0.0 else similarity_score
            if family_match:
                composite_score = (1.0 - composite_score) * 0.2 + composite_score # bonus of 20% if families match

        if not exact_match:
            if similarity_score < 0.8:
                errors.append("Low string similarity")
            if taxonomic_score < 0.5:
                errors.append("Failed taxonomic validation")
            if 0.5 <= taxonomic_score < 0.85:
                errors.append("Genus match")
            if 0.85 <= taxonomic_score < 1.0:
                errors.append("Species match")
            if taxonomic_score == 1.0:
                errors.append("Exact or synonymous match")
            if family_match:
                errors.append("Family matched")

        return FieldQualityScore(
            field_name="taxon",
            exact_match=exact_match,
            similarity_score=similarity_score,
            composite_score=composite_score,
            errors=errors
        )
    
    def _taxonomic_comparison(self, name1: str, name2: str) -> float:
        # specific epithet starting with an uppercase letter causes issues in GBIF API
        if len(name1.split()) == 2:
            name1 = name1.split()[0] + ' ' + name1.split()[1].lower()
        if len(name2.split()) == 2:
            name2 = name2.split()[0] + ' ' + name2.split()[1].lower()

        try:
            response1 = gbif_species.name_backbone(scientificName=name1)
        except Exception as e:
            print("Error during taxonomic comparison of transcribed name: " + name1 + ": " + str(e))
            return 0.0
        try:
            response2 = gbif_species.name_backbone(scientificName=name2)
        except Exception as e:
            print("Error during taxonomic comparison of ground truth name: " + name2 + ": " + str(e))
            return 0.0
        
        if 'usageKey' in response1 and 'usageKey' in response2:
            if response1['usageKey'] == response2['usageKey']:
                return 1.0
            elif response1.get('status') == 'SYNONYM' or response2.get('status') == 'SYNONYM':
                if (('acceptedUsageKey' in response1 or 'acceptedUsageKey' in response2)
                    and (response1.get('acceptedUsageKey') == response2.get('usageKey')
                    or response2.get('acceptedUsageKey') == response1.get('usageKey') 
                    or response1.get('acceptedUsageKey') == response2.get('acceptedUsageKey'))):
                    return 1.0
                elif ('speciesKey' in response1 and 'speciesKey' in response2 and
                    response1.get('speciesKey') == response2.get('speciesKey')):
                    return 0.85
                if response1.get('genusKey') == response2.get('genusKey'):
                    return 0.5
            elif ('speciesKey' in response1 and 'speciesKey' in response2 and
                response1.get('speciesKey') == response2.get('speciesKey')):
                return 0.85
            if response1.get('genusKey') == response2.get('genusKey'):
                return 0.5
        return 0.0
        
    def _remove_abbreviations(self, name: str) -> str: # Remove common taxonomic abbreviations that can cause issues with backbone comparison
        abbreviations = ['L.', 'subsp.', 'sp.', 'spp.', 'cf.', 'aff.']
        for abbreviation in abbreviations:
            name = name.replace(abbreviation, '')
        return name.strip()
        
class LocalityEvaluator(FieldEvaluator):
    def evaluate(self, transcription: str, ground_truth: str, transcription_record: Optional[Dict] = None, ground_truth_record: Optional[Dict] = None) -> FieldQualityScore:
        # The following combinations of fields make sure that location information distributed across several fields can be compared with Herbonauten ground truth
        field_combs_to_consider = [[], ['habitat'], ['country'], ['stateProvince'], ['minimum_elevation'], ['habitat', 'minimum_elevation'], 
                                   ['habitat', 'state_province'], ['habitat', 'minimum_elevation'], ['state_province', 'country'], ['state_province', 'county'],
                                   ['habitat', 'country'], ['habitat', 'county'], ['habitat', 'minimum_elevation', 'county'], 
                                   ['habitat', 'minimum_elevation', 'state_province'], ['habitat', 'minimum_elevation', 'country'], 
                                   ['habitat', 'minimum_elevation', 'county'], ['habitat', 'minimum_elevation', 'stateProvince', 'county']]
        errors = []

        transcription_clean = transcription.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()

        if ground_truth_clean == "" or ground_truth_clean == "nan":
            if transcription_clean == "" or transcription_clean == "nan":
                exact_match = True
            else:
                exact_match = transcription_clean == ground_truth_clean
        else:
            exact_match = transcription_clean == ground_truth_clean

        transcription_clean = "" if transcription_clean == "nan" else transcription_clean

        max_similarity = 0.0
        if transcription_clean != "":
            for comb in field_combs_to_consider:
                tr_parts = [transcription_clean]
                for field in comb:
                    if field in transcription_record and pd.notna(transcription_record[field]):
                        tr_parts.append(str(transcription_record[field]).strip().lower())
                tr_combined = ' '.join(tr_parts)
                normalized_ls = 1.0 - self.generic_levenshtein_distance(tr_combined, ground_truth_clean)
                jaro_winkler = self.jaro_winkler_distance(tr_combined, ground_truth_clean)
                jaccard_score = self._jaccard_similarity(tr_combined, ground_truth_clean)
                tr_combined = ' '.join(sorted(re.findall(r'\w+', tr_combined)))
                ground_truth_sorted = ' '.join(sorted(re.findall(r'\w+', ground_truth_clean)))
                jaccard_score = self._jaccard_similarity(tr_combined, ground_truth_sorted)
                similarity_score = max(normalized_ls, jaro_winkler, jaccard_score)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    errors = ["Using fields: Locality, " + ', '.join(comb) + " for locality similarity calculation: "+ str(tr_parts)]
            for comb in field_combs_to_consider: # this time without locality information TODO: Update this code to make it avoid repetitions
                tr_parts = []
                for field in comb:
                    if field in transcription_record and pd.notna(transcription_record[field]):
                        tr_parts.append(str(transcription_record[field]).strip().lower())
                tr_combined = ' '.join(tr_parts + [transcription_clean])
                normalized_ls = 1.0 - self.generic_levenshtein_distance(tr_combined, ground_truth_clean)
                jaro_winkler = self.jaro_winkler_distance(tr_combined, ground_truth_clean)
                jaccard_score = self._jaccard_similarity(tr_combined, ground_truth_clean)
                tr_combined = ' '.join(sorted(re.findall(r'\w+', tr_combined)))
                ground_truth_sorted = ' '.join(sorted(re.findall(r'\w+', ground_truth_clean)))
                jaccard_score = self._jaccard_similarity(tr_combined, ground_truth_sorted)
                similarity_score = max(normalized_ls, jaro_winkler, jaccard_score)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    errors = ["Using fields: Locality, " + ', '.join(comb) + " for locality similarity calculation: "+ str(tr_parts)]
        similarity_score = max_similarity

        if exact_match:
            composite_score = 1.0
        else:
            composite_score = similarity_score

        if not exact_match and (similarity_score < 0.7): 
            errors.append("Low locality similarity or incomplete recognition: {:.2f}".format(similarity_score))

        return FieldQualityScore(
            field_name="locality",
            exact_match=exact_match,
            similarity_score=similarity_score,
            composite_score=composite_score,
            errors=errors
        )

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        set1 = set(s1.split())
        set2 = set(s2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 1.0 if not intersection else 0.0
        return len(intersection) / len(union)
    
class CollectionDateEvaluator(FieldEvaluator):

    def evaluate(self, transcription: str, ground_truth: str, transcription_record: Optional[Dict] = None, ground_truth_record: Optional[Dict] = None) -> FieldQualityScore:
        errors = []

        transcription_clean = transcription.strip()
        ground_truth_clean = ground_truth.strip()

        transcription_no_date_flag = transcription_clean == "" or transcription_clean.lower() == "nan" or pd.isna(transcription_clean) or transcription_clean == "<NA>"
        ground_truth_no_date_flag = ground_truth_clean == "" or ground_truth_clean.lower() == "nan" or pd.isna(ground_truth_clean) or ground_truth_clean == "<NA>"

        if 'eventDateEnd' in ground_truth_record and not ground_truth_record['eventDateEnd'] != ground_truth_record['eventDateEnd']:  # NaN check
            if ground_truth_clean != ground_truth_record['eventDateEnd'].strip():
                ground_truth_clean += "/" + ground_truth_record['eventDateEnd'].strip()
        
        exact_match = transcription_clean == ground_truth_clean or (transcription_no_date_flag and ground_truth_no_date_flag)
        transcribed_date = self._parse_date(transcription_clean)
        ground_truth_date = self._parse_date(ground_truth_clean)

        similarity_score = self._calculate_date_similarity(transcribed_date, ground_truth_date)

        if exact_match:
            composite_score = 1.0
        else:
            composite_score = similarity_score

        if not exact_match and similarity_score < 0.8:
            errors.append("Date mismatch or low similarity {:.2f}".format(similarity_score))
        return FieldQualityScore(
            field_name="collection_date",
            exact_match=exact_match,
            similarity_score=similarity_score,
            composite_score=composite_score,
            errors=errors
        )

    def _parse_date(self, date_str: str) -> Optional[Union[Dict, List[Dict]]]:
        if not date_str:
            return None
        
        date_patterns = [ # regular expressions to identify certain collection date formats
            (r'(\d{4})-(\d{1,2})-(\d{1,2})$', 'ymd'),  # YYYY-MM-DD
            (r'(\d{1,2})/(\d{1,2})/(\d{4})$', 'mdy'),  # MM/DD/YYYY
            (r'(\d{1,2})-(\d{1,2})-(\d{4})$', 'mdy'),  # MM-DD-YYYY
            (r'(\d{4})/(\d{1,2})/(\d{1,2})$', 'ymd'),  # YYYY/MM/DD
            (r'(\d{4})-(\d{1,2})$', 'ym'),             # YYYY/MM
            (r'(\d{4})$', 'y'),                        # YYYY
            (r'(\d{4})-(\d{1,2})-(\d{1,2})/(\d{4})-(\d{1,2})-(\d{1,2})$', 'ymd/ymd'),  # YYYY-MM-DD/YYYY-MM-DD
            (r'(\d{4})-(\d{1,2})/(\d{4})-(\d{1,2})$', 'ym/ym'),              # YYYY-MM/YYYY-MM
            (r'(\d{4})/(\d{4})$', 'y/y'),                         # YYYY/YYYY
        ]

        result = list(dict())
        for date in date_str.split('/'):
            for pattern, date_format in date_patterns:
                match = re.match(pattern, date)
                if match:
                    groups = match.groups()
                    try:
                        if date_format == 'ymd':
                            result.append({
                                'year': int(groups[0]),
                                'month': int(groups[1]),
                                'day': int(groups[2])
                            })
                        elif date_format == 'mdy':
                            result.append({
                                'year': int(groups[2]),
                                'month': int(groups[0]),
                                'day': int(groups[1])
                            })
                        elif date_format == 'ym':
                            result.append({
                                'year': int(groups[0]),
                                'month': int(groups[1]),
                                'day': None
                            })
                        elif date_format == 'y':
                            result.append({
                                'year': int(groups[0]),
                                'month': None,
                                'day': None
                            })
                        elif date_format == 'ymd/ymd':
                            result.append({
                                'year': int(groups[0]),
                                'month': int(groups[1]),
                                'day': int(groups[2])
                            })
                            result.append({
                                'year': int(groups[3]),
                                'month': int(groups[4]),
                                'day': int(groups[5])
                            })
                        elif date_format == 'ym/ym':
                            result.append({
                                'year': int(groups[0]),
                                'month': int(groups[1]),
                                'day': None
                            })
                            result.append({
                                'year': int(groups[2]),
                                'month': int(groups[3]),
                                'day': None
                            })
                        elif date_format == 'y/y':
                            result.append({
                                'year': int(groups[0]),
                                'month': None,
                                'day': None
                            })
                            result.append({
                                'year': int(groups[1]),
                                'month': None,
                                'day': None
                            })
                    except ValueError:
                        return None
        if len(result) == 1:
            return result[0]
        if len(result) == 0:
            return None
        return result                    
    
    def _calculate_date_similarity(self, date1: Optional[Union[Dict, List[Dict]]], date2: Optional[Union[Dict, List[Dict]]]) -> float:
        if date1 is None or date2 is None:
            return 0.0

        # Case 1: Both are single dates
        if isinstance(date1, dict) and isinstance(date2, dict):
            year_match = date1.get('year') == date2.get('year')
            month_match = date1.get('month') == date2.get('month')
            day_match = date1.get('day') == date2.get('day')

            if year_match and month_match and day_match:
                return 1.0
            elif year_match and month_match:
                return 0.8
            elif year_match:
                return 0.4
                
            return 0.0
        
        # Case 2: One or both are date ranges
        dates1 = date1 if isinstance(date1, list) else [date1]
        dates2 = date2 if isinstance(date2, list) else [date2]

        d1_year_interval = [dates1[0].get('year'), dates1[-1].get('year')]
        d2_year_interval = [dates2[0].get('year'), dates2[-1].get('year')]

        d1_month_interval = [dates1[0].get('month'), dates1[-1].get('month')]
        d2_month_interval = [dates2[0].get('month'), dates2[-1].get('month')]

        # Assign default 1 and 12 for missing months
        d1_month_interval[0] = 1 if d1_month_interval[0] is None or d1_month_interval[0] == 0 else d1_month_interval[0]
        d1_month_interval[1] = 12 if d1_month_interval[1] is None or d1_month_interval[1] == 0 else d1_month_interval[1]

        d1_day_interval = [dates1[0].get('day'), dates1[-1].get('day')]
        d2_day_interval = [dates2[0].get('day'), dates2[-1].get('day')]

        # Assign month-specific end days 
        if d1_day_interval[0] is None:
            d1_day_interval[0] = 1
        d1_day_interval[0] = 1 if d1_day_interval[0] is None else d1_day_interval[0]
        if d1_day_interval[1] is None:
            if d1_month_interval[1] in [1, 3, 5, 7, 8, 10, 12]:
                d1_day_interval[1] = 31
            elif d1_month_interval[1] in [4, 6, 9, 11]:
                d1_day_interval[1] = 30
            elif d1_month_interval[1] == 2:
                if (d1_year_interval[1] is not None and ((d1_year_interval[1] % 4 == 0 and d1_year_interval[1] % 100 != 0) or (d1_year_interval[1] % 400 == 0))):
                    d1_day_interval[1] = 29
                else:
                    d1_day_interval[1] = 28
            else:
                d1_day_interval[1] = 31

        if d1_year_interval[0] is None or d1_year_interval[1] is None or d2_year_interval[0] is None or d2_year_interval[1] is None:
            return 0.0
        # Construct valid date objects
        d1_start = datetime.date(d1_year_interval[0], d1_month_interval[0], d1_day_interval[0])
        d1_end = datetime.date(d1_year_interval[1], d1_month_interval[1], d1_day_interval[1])
        d2_start = datetime.date(d2_year_interval[0], d2_month_interval[0], d2_day_interval[0])
        d2_end = datetime.date(d2_year_interval[1], d2_month_interval[1], d2_day_interval[1])

        exact_date_overlap = d1_start == d2_start and d1_end == d2_end
        date_overlap = d1_start <= d2_end and d2_start <= d1_end # partial overlap

        if exact_date_overlap:
            return 1.0
        elif date_overlap:
            return 0.6
        
        return 0.0

class CountryEvaluator(FieldEvaluator):
    
    def evaluate(self, transcription: str, ground_truth: str, transcription_record: Optional[Dict] = None, ground_truth_record: Optional[Dict] = None) -> FieldQualityScore:
        errors = []

        transcription_clean = transcription.strip()
        ground_truth_clean = ground_truth.strip()

        exact_match = transcription_clean.lower() == ground_truth_clean.lower()
        similarity_score = self._compare_countries(transcription_clean, ground_truth_clean)

        if exact_match:
            composite_score = 1.0
        elif similarity_score == 1.0:
            composite_score = 1.0
        else:
            composite_score = 0.0

        if not exact_match:
            if similarity_score != 1.0:
                errors.append("Country mismatch")

        return FieldQualityScore(
            field_name="country",
            exact_match=exact_match,
            similarity_score=similarity_score,
            composite_score=composite_score,
            errors=errors
        )
                
    def _compare_countries(self, country1: str, country2: str) -> float:
        if not country1 or not country2:
            if not country1 and not country2:
                return 1.0
            else:
                return 0.0
        try:
            c1 = pycountry.countries.lookup(country1) # lookup name
        except LookupError:
            c1 = countrycode(country1, origin='country.name.en', destination='iso2c') # lookup code
        try:
            c2 = pycountry.countries.lookup(country2)
        except LookupError:
            c2 = countrycode(country2, origin='country.name.de', destination='iso2c') # german 

        if c1 is None and c2 is None:
            return 1.0
        if (c1 is None and c2 is not None) or (c1 is not None and c2 is None): # one of the countries unknown
            return 0.0
        
        if isinstance(c1, str):
            c1_code = c1 # countrycode returns a string
        else:
            c1_code = c1.alpha_2 # pycountry returns an Country object
        if isinstance(c2, str):
            c2_code = c2
        else:
            c2_code = c2.alpha_2
        if c1_code == c2_code:
            return 1.0
        else:
            return 0.0
     
class CollectorEvaluator(FieldEvaluator):

    def evaluate(self, transcription: str, ground_truth: str, transcription_record: Optional[Dict] = None, ground_truth_record: Optional[Dict] = None) -> FieldQualityScore:
        errors = []

        transcription_clean = self._space_after_initials(transcription.strip()).replace(" .", "") # unwanted spaces before dots may be yieled by _space_after_initials
        ground_truth_clean = self._space_after_initials(ground_truth.strip()).replace(" .", "")

        exact_match = transcription_clean.lower() == ground_truth_clean.lower()
        transcription_variants = [transcription_clean] # due to uncertainty about input formats, multiple variants of the names are considered for evaluation
        # Handling of multiple names, different formats and extraction of diverse variants
        if not exact_match:
            # NaN and empty check for associated collectors
            if not (transcription_record['associated_collectors'] != transcription_record['associated_collectors']) and transcription_record.get('associated_collectors', "") != "":
                # create variants associated collectors for formats commonly occurring during transcription and for values with two and more collectors
                transcription_variants.append(transcription_clean + ', ' + self._add_space_after_initials(transcription_record.get('associated_collectors', "")))
                transcription_variants.append(transcription_clean + ' & ' + self._add_space_after_initials(transcription_record.get('associated_collectors', "")))
            multiple_collectors = []
            for single_collector in transcription_clean.split("&"): # split last collector from first collector(s)
                if "," in single_collector: # if first collector is a group of collectors
                    transcription_split = single_collector.split(",")
                    if len(transcription_split) == 2: # most likely "last name, first name" format
                        transcription_variant = transcription_split[1].strip() + ' ' + transcription_split[0].strip()
                        transcription_variants.append(transcription_variant)
                        exact_match = transcription_variant.lower() == ground_truth_clean.lower()
                        multiple_collectors.append(transcription_variant)
            if len(multiple_collectors) > 1:
                transcription_variants.append(' & '.join(multiple_collectors))
        # add dots after single capital letters if missing
        for transcription_variant in transcription_variants.copy():
            # dots and  space are added to single letters without dots
            modified_variant = re.sub(r'\b\w\b(?!\.)', lambda x: x.group(0) + '. ', transcription_variant).replace("  ", " ")
            if modified_variant != transcription_variant:
                transcription_variants.append(modified_variant)
            # remove initials as standalone letters and those with dots
            modified_variant = re.sub(r'\b(\w\.?)\b', '', transcription_variant).replace(".", "").replace("  ", " ")
            if modified_variant != transcription_variant:
                transcription_variants.append(modified_variant)
        
        variants_scores = {}
        for variant in transcription_variants:
            normalized_ls = 1.0 - self.generic_levenshtein_distance(variant.lower(), ground_truth_clean.lower())
            normalized_sort_ratio = 1.0 - self.token_sort_ratio(variant.lower(), ground_truth_clean.lower())
            similarity_score = max(normalized_ls, normalized_sort_ratio)
            variants_scores[variant] = similarity_score
        similarity_score = max(variants_scores.values())

        if exact_match:
            composite_score = 1.0
        else:
            composite_score = similarity_score
        if not exact_match:
            if similarity_score < 0.7:
                errors.append("Low collector name similarity: {:.2f}".format(similarity_score))
        
        return FieldQualityScore(
            field_name="collector",
            exact_match=exact_match,
            similarity_score=similarity_score,
            composite_score=composite_score,
            errors=errors
        )
    
    def _add_space_after_initials(self, name: str) -> str:
        if '.' in name:
            if re.search(r'\.[A-Za-z]', name):
                name = name.replace('.', '. ')
        return name

class Evaluator:
    # This is the main class that manages the entire evaluation process
    def __init__(self):
        self.field_evaluators = {
            "taxon": TaxonEvaluator(),
            "collector": CollectorEvaluator(),
            "collection_date": CollectionDateEvaluator(),
            "locality": LocalityEvaluator(),
            "country": CountryEvaluator()
        }

        self.weights = { # weights for the evaluation formula to compute the total composite score
            "taxon": 0.2,
            "collector": 0.2,
            "collection_date": 0.2,
            "locality": 0.2,
            "country": 0.2
        }

        self.dictionary_fields = { # map to ground truth field names of the Herbonauten dataset
            "taxon": "scientificName",
            "collector": "recordedBy",
            "collection_date": "eventDate",
            "locality": "locality",
            "country": "country"
        }

        self.thresholds = { # quality thresholds
            'high_quality': 0.80, # quality level for automatic transcription
            'medium_quality': 0.65 # manual revision of the automatically transcribed data required. below, specimens require full manual transcription
        }

        self.results_data = []
        self.ground_truth_data = {}
        self.records_skipped = [] # if ground truth is missing for a record then skip it
        self.records_failed_to_evaluate = []
        self.evaluation_results = []
        self.results_for_export = []

    def load_ground_truth(self, ground_truth: Union[str, pd.DataFrame]):
        if isinstance(ground_truth, pd.DataFrame):
            if 'catalogNumber' not in ground_truth.columns:
                raise ValueError("Ground truth DataFrame must contain 'catalogNumber' column.")
            for _, row in ground_truth.iterrows():
                row['catalogNumber'] = row['catalogNumber'].replace(" ", "")
                self.ground_truth_data[row['catalogNumber']] = row
        elif isinstance(ground_truth, str):
            try:
                df = pd.read_csv(ground_truth)
                for _, row in df.iterrows():
                    row['catalogNumber'] = row['catalogNumber'].replace(" ", "")
                    self.ground_truth_data[row['catalogNumber']] = row
            except Exception as e:
                print("An error occrured while loading ground truth data: " + str(e))

        print("Loaded ground truth data for {} specimens.".format(len(self.ground_truth_data)))

    def add_transcription_result(self, catalogNumber: str, result: TranscriptionResult):
        self.results_data.append({
            "catalogNumber": catalogNumber,
            "tool_name": result.tool_name,
            "result": result
        })

    def add_transcription_results(self, path: Union[str, pd.DataFrame]) -> None:
        if isinstance(path, pd.DataFrame):
            for _, row in path.iterrows():
                transcription_result = TranscriptionResult(
                    tool_name=row['tool_name'],
                    taxon=row.get('scientificName', ""),
                    scientific_name=row.get('scientificName', ""),
                    genus=row.get('genus', ""),
                    specific_epithet=row.get('specificEpithet', ""),
                    infraspecific_epithet=row.get('infraspecificEpithet', ""),
                    scientific_name_authorship=row.get('scientificNameAuthorship', ""),
                    family=row.get('family', ""),
                    collector=row.get('recordedBy', ""),
                    associated_collectors=row.get('associatedCollectors', ""),
                    collection_date=row.get('eventDate', ""),
                    year=row.get('year', ""),
                    month=row.get('month', ""),
                    day=row.get('day', ""),
                    locality=row.get('locality', ""),
                    habitat=row.get('habitat', ""),
                    minimum_elevation=row.get('minimumElevation', ""),
                    maximum_elevation=row.get('maximumElevation', ""),
                    county=row.get('county', ""),
                    state_province=row.get('stateProvince', ""),
                    country=row.get('country', ""),
                    specimen_description=row.get('specimenDescription', "")
                )
                self.add_transcription_result(row['catalogNumber'], transcription_result)
            print(f"Loaded {len(self.results_data)} transcription results from DataFrame.")
            return
        else:
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    transcription_result = TranscriptionResult(
                        tool_name=row['tool_name'],
                        taxon=row.get('scientificName', ""),
                        scientific_name=row.get('scientificName', ""),
                        genus=row.get('genus', ""),
                        specific_epithet=row.get('specificEpithet', ""),
                        infraspecific_epithet=row.get('infraspecificEpithet', ""),
                        scientific_name_authorship=row.get('scientificNameAuthorship', ""),
                        collector=row.get('recordedBy', ""),
                        family=row.get('family', ""),
                        collection_date=row.get('eventDate', ""),
                        year=row.get('year', ""),
                        month=row.get('month', ""),
                        day=row.get('day', ""),
                        locality=row.get('locality', ""),
                        country=row.get('country', "")
                    )
                    self.add_transcription_result(row['catalogNumber'], transcription_result)
                print("Loaded " + str(len(self.results_data)) + " transcription results from " + path + ".")
            except Exception as e:
                print("Error loading transcription results from " + path + ": " + str(e))

    def evaluate_single_field(self, field_name: str, transcription: str, ground_truth: str, transcription_record: TranscriptionResult, ground_truth_record: Dict) -> FieldQualityScore:
        if field_name not in self.field_evaluators:
            raise ValueError("No evaluator implemented for field: " + field_name)
        
        return self.field_evaluators[field_name].evaluate(transcription, ground_truth, transcription_record.__dict__, ground_truth_record)

    def evaluate_specimen(self, catalogNumber: str, transcription: TranscriptionResult, ground_truth: Dict) -> Dict:
        # evaluation of a single herbarium sheet
        field_scores = {}

        for field_name in self.weights.keys():
            transcription_value = getattr(transcription, field_name, "")
            if transcription_value is None:
                transcription_value = ""

            ground_truth_value = str(ground_truth.get(self.dictionary_fields[field_name], ""))
            if ground_truth_value is None:
                ground_truth_value = ""

            field_score = self.evaluate_single_field(field_name, transcription_value, ground_truth_value, transcription, ground_truth)
            field_scores[field_name] = field_score

        composite_score = 0.0
        for field in self.weights.keys():
            composite_score += field_scores[field].composite_score * self.weights[field]

        if composite_score >= self.thresholds['high_quality']:
            quality = 'high'
            recommended_pipeline = transcription.tool_name
        elif composite_score >= self.thresholds['medium_quality']:
            quality = 'medium'
            recommended_pipeline = 'manual review'
        else:
            quality = 'low'
            recommended_pipeline = 'herbonauten'

        values = {}
        for field in self.weights.keys():
            values[field] = {
                "transcription": getattr(transcription, field, ""),
                "ground_truth": ground_truth[self.dictionary_fields[field]]
            }

        return {
            "catalogNumber": catalogNumber,
            "tool_name": transcription.tool_name,
            "field_scores": field_scores,
            "composite_score": composite_score,
            "quality": quality,
            "recommended_pipeline": recommended_pipeline, # based on this information, the labels for the training data are assigned
            "single_field_scores": {field: field_scores[field].composite_score for field in field_scores},
            "values": values
        }

    def evaluate_all(self) -> List[Dict]:
        if not self.results_data:
            raise ValueError("No transcription results found for evaluation. Please use add_transcription_result() to add.")
        if not self.ground_truth_data:
            raise ValueError("Ground truth data not found for evaluation.")
        evaluation_results = []

        # Grouping by catalogNumber
        specimen_results = defaultdict(list) # defaultdict to avoid key errors for not existing catalogNumbers
        for specimen_result in self.results_data:
            specimen_results[specimen_result['catalogNumber']].append(specimen_result)
        for catalogNumber, results in specimen_results.items():
            ground_truth_record = self.ground_truth_data.get(catalogNumber)

            if ground_truth_record is None:
                self.records_skipped.append(catalogNumber)
                continue

            for result in results:
                try:
                    evaluation = self.evaluate_specimen(catalogNumber, result['result'], ground_truth_record)
                    evaluation_results.append(evaluation)
                except Exception as e:
                    print("Error evaluating specimen "+ catalogNumber + " with tool " + result['tool_name'] + ": " + str(e))
                    # print(traceback.format_exc()) # comment in for detailed information about the error
                    self.records_failed_to_evaluate.append({"catalogNumber": catalogNumber, "tool_name": result['tool_name'], "error": str(e)})

        evaluation_results = self.harmonize_multilabel_scores(evaluation_results)

        print("Successfully evaluated " + str(len(evaluation_results)) + " specimens. (Skipped " + str(len(self.records_skipped)) + ". Failed for " + str(len(self.records_failed_to_evaluate)) + ".)")
        self.evaluation_results = evaluation_results
        return evaluation_results

    def harmonize_multilabel_scores(self, results: List[Dict]) -> List[Dict]:
        # recalculate composite scores for multilabel specimens
        # this is required for tools like Hespi that produce multiple transcriptions for one catalogNumber
        # group by catalogNumber and tool_name
        grouped_results = defaultdict(list)
        for result in results:
            catalog_number = result['catalogNumber']
            tool = result['tool_name']
            if catalog_number not in grouped_results.keys():
                grouped_results[catalog_number] = dict()
            if tool not in grouped_results[catalog_number]:
                grouped_results[catalog_number][tool] = []
            grouped_results[catalog_number][tool].append(result['field_scores'])
        for catalog_number, result in grouped_results.items():
            for tool, field_scores_list in result.items():
                if len(field_scores_list) > 1:
                    max_field_scores = dict() # find the highest field-specific scores for multilabel specimens
                    for field in self.weights.keys():
                        for field_scores in field_scores_list:
                            if field not in max_field_scores:
                                max_field_scores[field] = field_scores[field]
                            else:
                                if field_scores[field].composite_score > max_field_scores[field].composite_score:
                                    max_field_scores[field] = field_scores[field]
                    composite_score = 0.0
                    for field in self.weights.keys(): # recalculate the CS for multilabel specimens
                        composite_score += max_field_scores[field].composite_score * self.weights[field]
                    for result_record in results: # update the result attributes
                        if result_record['catalogNumber'] == catalog_number and result_record['tool_name'] == tool:
                            result_record['field_scores'] = max_field_scores
                            result_record['composite_score'] = composite_score
                            if composite_score >= self.thresholds['high_quality']:
                                result_record['quality'] = 'high'
                                result_record['recommended_pipeline'] = tool
                            elif composite_score >= self.thresholds['medium_quality']:
                                result_record['quality'] = 'medium'
                                result_record['recommended_pipeline'] = 'manual review'
                            else:
                                result_record['quality'] = 'low'
                                result_record['recommended_pipeline'] = 'herbonauten'
        return results
    
    def export_results(self, results: List[Dict], filename: str = 'transcription_evaluation_results.csv'):
        if not results:
            print("No results available for export.")
            return
        
        export_data = []

        for result in results:
            record = {
                'catalogNumber': result['catalogNumber'],
                'tool_name': result['tool_name'],
                'composite_score': result['composite_score'],
                'quality': result['quality'],
                'recommended_pipeline': result['recommended_pipeline']
            }
            
            for field, _ in result['single_field_scores'].items():
                record_transcription = result['values'][field]['transcription']
                record_ground_truth = result['values'][field]['ground_truth']
                record[f'{field}_transcription'] = record_transcription
                record[f'{field}_ground_truth'] = record_ground_truth
                record[f'{field}_exact_match'] = result['field_scores'][field].exact_match
                record[f'{field}_composite_score'] = result['field_scores'][field].composite_score
                record[f'{field}_errors'] = '; '.join(result['field_scores'][field].errors)
            
            export_data.append(record)
        
        self.results_for_export = export_data

        folder = os.path.dirname(filename)
        print("Exporting results to '" + filename + "'...")
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        export_df = pd.DataFrame(export_data)
        export_df.to_csv(filename, index=False)

        print("Results successfully exported.")