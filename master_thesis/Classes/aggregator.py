#DEFAULT
from abc import ABC, abstractmethod
from dateutil.parser import parse
import re
from typing import List
#3RD PARTY
import pandas as pd 

class Aggregator(): # This class aggregates the heterogeneous transcriptions from different tools.
    def __init__(self, tool_names: List[str]) -> None:
        self.tool_names = tool_names
        self.transcriptions = dict()
        for tool in tool_names:
            self.transcriptions[tool] = pd.DataFrame()
        self.output = pd.DataFrame()

    def add_transcriptions(self, tool_name: str, transcriptions: str) -> None:
        if tool_name not in self.tool_names:
            raise ValueError("Tool " + tool_name + " not found.")
        
        if transcriptions.endswith('csv'):
            transcriptions_df = pd.read_csv(transcriptions, sep=',').fillna('')
        elif transcriptions.endswith('xlsx'):
            transcriptions_df = pd.read_excel(transcriptions).fillna('')

        if tool_name == 'hespi':
            transcriptions_df = self._edit_hespi_transcriptions(transcriptions_df)
        elif tool_name == 'vouchervision':
            transcriptions_df = self._edit_vouchervision_transcriptions(transcriptions_df)
        
        transcriptions_df['tool_name'] = tool_name
        self.transcriptions[tool_name] = transcriptions_df

    def aggregate(self) -> None:
        aggregated_data = pd.DataFrame()

        for tool_name, df in self.transcriptions.items():
            print("Aggregating data from tool: " + tool_name + " with " + str(len(df)) + " records.")
            aggregated_data = pd.concat([aggregated_data, df], ignore_index=True)

        self.output = aggregated_data.drop_duplicates().reset_index(drop=True)

    def get_aggregated_data(self) -> pd.DataFrame:
        return self.output
    
    def save_aggregated_data(self, filepath: str):
        self.output.to_csv(filepath, index=False)

    def _edit_hespi_transcriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        hespi_transformer = HespiDataTransformer(df)
        return hespi_transformer.transform()

    def _edit_vouchervision_transcriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        vouchervision_transformer = VouchervisionDataTransformer(df)
        return vouchervision_transformer.transform()
    
class DataTransformer(ABC): # Abstract class for tool-specific transformers that converts tool-specific transcription data for aggregation
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.fields_mapping = {} # mapping of Herbonauten column names to DwC-compliant column names

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        pass

class HespiDataTransformer(DataTransformer):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)
        self.fields_mapping = {
            'species': 'specificEpithet',
            'infrasp_taxon': 'infraspecificEpithet',
            'genus': 'genus',
            'authority': 'scientificNameAuthorship',
            'family': 'family',
            'collector': 'recordedBy',
            'day': 'day',
            'month': 'month',
            'year': 'year',
            'locality': 'locality',
            'country': 'country'
        }

    def transform(self) -> pd.DataFrame:
        transformed_data = pd.DataFrame()
        event_date = ""

        for source_field, target_field in self.fields_mapping.items():
            if source_field in self.data.columns:
                transformed_data[target_field] = self.data[source_field]
        if 'id' in self.data.columns:
            transformed_data['catalogNumber'] = self.data['id']
        if 'year' in transformed_data.columns:
            def extract_year(date: str) -> str: # helper function for applying year extraction to the year column
                if pd.isna(date) or date == '':
                    return date
                date_str = str(date)
                date_str = filter(str.isdigit, date_str) # only numbers
                date_str = ''.join(date_str)
                return date_str
            transformed_data['year'] = transformed_data['year'].apply(extract_year)
        if 'month' in transformed_data.columns:
            def extract_month(date: str) -> str:
                roman_numerals = {
                    'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 'VII': '7', 
                    'VIII': '8', 'IX': '9', 'X': '10', 'XI': '11', 'XII': '12'
                }
                german_months = {
                    'JANUAR': '1', 'FEBRUAR': '2', 'MÄRZ': '3', 'MAERZ': '3', 'APRIL': '4', 
                    'MAI': '5', 'JUNI': '6', 'JULI': '7', 'AUGUST': '8', 'SEPTEMBER': '9', 
                    'OKTOBER': '10', 'NOVEMBER': '11', 'DEZEMBER': '12'
                }
                if pd.isna(date) or date == '':
                    return date
                date_str = str(date)
                date_str = ''.join(filter(str.isalnum, date_str)) # only alphanumeric chars
                date_str = german_months.get(date_str.upper(), date_str)
                date_str = roman_numerals.get(date_str.upper(), date_str)
                return date_str
            transformed_data['month'] = transformed_data['month'].apply(extract_month)
        if 'day' in transformed_data.columns:
            def extract_day(date: str) -> str:
                if pd.isna(date) or date == '':
                    return date
                date_str = str(date)
                date_str = ''.join(filter(str.isdigit, date_str)) # only numbers
                return date_str
            transformed_data['day'] = transformed_data['day'].apply(extract_day)
        for index, row in transformed_data.iterrows():
            date_parts = []
            if pd.notna(row.get('year')) and row.get('year') != '':
                date_parts.append(row['year'])
            if pd.notna(row.get('month')) and row.get('month') != '':
                date_parts.append(row['month'])
                if pd.notna(row.get('day')) and row.get('day') != '':
                    date_parts.append(row['day'])
            event_date = '-'.join(date_parts) # merge year, month and day to yyyy-mm-dd
            if event_date != "":
                try: # necessary to prevent transformation from stopping if a date cannot be parsed
                    parsed_date = parse(event_date, fuzzy=True)
                    event_date = parsed_date.strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    print("Could not parse hespi date from: " + event_date + " for: " + transformed_data.at[index, 'catalogNumber'])
                    event_date = pd.NA
                transformed_data.at[index, 'eventDate'] = event_date
        if 'genus' in transformed_data.columns:
            if 'specificEpithet' in transformed_data.columns:
                for index, row in transformed_data.iterrows():
                    if pd.notna(row['specificEpithet']) and row['specificEpithet'] != '':
                        transformed_data.at[index, 'scientificName'] = row['genus'].strip() + ' ' + row['specificEpithet'].strip() 
            else:
                for index, row in transformed_data.columns:
                    transformed_data.at[index, 'scientificName'] = row['genus'].strip() # sometimes Hespi adds spaces to the values
            if 'infraspecificEpithet' in transformed_data.columns:
                for index, row in transformed_data.iterrows():
                    if pd.notna(row['infraspecificEpithet']):
                        if pd.notna(row['scientificName']) and row['scientificName'] != '':
                            transformed_data.at[index, 'scientificName'] += ' ' + row['infraspecificEpithet'].strip()
        elif 'specificEpithet' in transformed_data.columns:
                for index, row in transformed_data.iterrows():
                    capitalized_spec_epithet = row['specificEpithet'][0].isupper() # sometimes Genus ends up in specificEpithet
                    if pd.notna(row['specificEpithet']) and capitalized_spec_epithet and len(row['specificEpithet'].strip().split(' ')) > 1:
                        transformed_data.at[index, 'scientificName'] = row['specificEpithet'].strip()
        return transformed_data

class VouchervisionDataTransformer(DataTransformer):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)
        self.fields_mapping = {
            'scientificName': 'scientificName',
            'genus': 'genus',
            'family': 'family',
            'specificEpithet': 'specificEpithet',
            'scientificNameAuthorship': 'scientificNameAuthorship',
            'collector': 'recordedBy',
            'associatedCollectors': 'associatedCollectors',
            'collectionDate': 'eventDate',
            'collectionDateEnd': 'eventDateEnd',
            'locality': 'locality',
            'habitat': 'habitat',
            'minimumElevationInMeters': 'minimumElevationInMeters',
            'maximumElevationInMeters': 'maximumElevationInMeters',
            'county': 'county',
            'stateProvince': 'stateProvince',
            'country': 'country',
            'specimenDescription': 'specimenDescription'
        }

    def transform(self) -> pd.DataFrame:
        transformed_data = pd.DataFrame()
        for source_field, target_field in self.fields_mapping.items():
            if source_field in self.data.columns:
                transformed_data[target_field] = self.data[source_field]
        if 'Filename' in self.data.columns:
            filename_column = 'Filename'
        elif 'filename' in self.data.columns:
            filename_column = 'filename' # most recent version of VV uses "filename"
        if filename_column in self.data.columns:
            transformed_data['catalogNumber'] = self.data[filename_column].str.replace(r'\.[^.]+$', '', regex=True) # remove file extension from filename
        for date_column in ['eventDate', 'eventDateEnd']: # ranges are parsed by VV in two columns
            for index, row in transformed_data.iterrows():
                if pd.isna(row[date_column]):
                    transformed_data.at[index, date_column] = pd.NA
                else:
                    transformed_data.at[index, date_column] = re.sub(r'-00$', '', row[date_column]) # replace -00 at the end (unknown day or month) 
            for index, row in transformed_data.iterrows():
                date_str = row[date_column]
                if pd.notna(date_str) and date_str != '' and not date_str.startswith('0000'):
                    try:
                        date_str = date_str.replace('-00', '') if len(date_str) == 7 else date_str
                        parsed_date = parse(date_str, fuzzy=True, default=pd.Timestamp(1000, 1, 1)) # Flag date to identify unparseable dates
                        if parsed_date.year == 1000: # Flag date => NA is assigned
                            transformed_data.at[index, date_column] = pd.NA
                        else:
                            if date_column == 'eventDate':
                                transformed_data.at[index, 'year'] = str(parsed_date.year)
                                transformed_data.at[index, 'month'] = str(parsed_date.month) if parsed_date.month != 1 or '-01' in date_str else pd.NA # distinguish from flag date
                                transformed_data.at[index, 'day'] = str(parsed_date.day) if date_str.count('-') == 2 else pd.NA # two dashes indicator for presence of day
                            else: # for eventDateEnd, add slash followed by year, month or day
                                if transformed_data.at[index, 'eventDate'] is not pd.NA:
                                    if transformed_data.at[index, 'year'] is not pd.NA and transformed_data.at[index, 'year'] != str(parsed_date.year):
                                        transformed_data.at[index, 'year'] = transformed_data.at[index, 'year'] + '/' + str(parsed_date.year)
                                    if transformed_data.at[index, 'month'] is not pd.NA and transformed_data.at[index, 'month'] != str(parsed_date.month):
                                        transformed_data.at[index, 'month'] = transformed_data.at[index, 'month'] + '/' + str(parsed_date.month)
                                    if transformed_data.at[index, 'day'] is not pd.NA and transformed_data.at[index, 'day'] != str(parsed_date.day):
                                        transformed_data.at[index, 'day'] = transformed_data.at[index, 'day'] + '/' + str(parsed_date.day)
                                try:
                                    transformed_data.at[index, 'eventDate'] = transformed_data.at[index, 'eventDate'] + '/' + parsed_date.strftime('%Y-%m-%d')
                                except Exception as e:
                                    print(f"Error updating eventDate at index "+ str(index) + ": "  + str(e))
                                    transformed_data.at[index, 'eventDate'] = pd.NA
                    except (ValueError, TypeError):
                        transformed_data.at[index, date_column] = pd.NA
                        print("Could not parse VoucherVision date from: " + date_str + " for: " + transformed_data.at[index, 'catalogNumber'])
                else:
                    transformed_data.at[index, date_column] = pd.NA # unknown year => set to NA to prevent parsing errors
        
        return transformed_data 