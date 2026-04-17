"""
This file contains methods to randomly select catalog numbers from an input Herbonauten metadata file
"""

#DEFAULT
import datetime
import random
#3RD PARTY
import pandas as pd

def pick_random_catalog_numbers_stratified_families(file, num_samples, column_name):
    """ 
    This function picks x random catalogNumbers stratified by families from a given CSV file and stores these numbers back in a new CSV file.
    The distribution of the families in the sampled catalogNumbers aligns with the respective distribution in the input file, with each 
    family having at least one occurrence in the result.

    Parameters
    ----------
    file: str or pd.Dataframe
        Input file containing catalog numbers for random picking. Is a path or a Dataframe object
    num_samples: int
        Number of records to be picked
    column_name: str
        Name of the key column containing catalog numbers

    Returns
    -------
    str
        Path of the CSV file containing the randomly sampled catalog numbers
    """
    if type(file) == str:
        df = pd.read_csv(file, sep=';')
    elif type(file) == pd.DataFrame:
        df = file
    if column_name not in df.columns:
        raise ValueError("The input file does not contain the specified key column.")
        
    family_counts = df['Family'].value_counts(normalize=True) # Get the relative family distribution of the input file
    sampled_catalog_numbers = []

    for family, proportion in family_counts.items():
        family_df = df[df['Family'] == family]
        num_family_samples = max(1, int(proportion * num_samples))  # Ensure at least one sample per family
        unique_family_catalog_numbers = family_df[column_name].dropna().unique() # Ensure unique catalog numbers and NA exclusion
        if len(unique_family_catalog_numbers) < num_family_samples: # Check if required number of catalog number per family is reached
            num_family_samples = len(unique_family_catalog_numbers)
        family_samples = random.sample(list(unique_family_catalog_numbers), num_family_samples) # Sample from each family
        sampled_catalog_numbers += family_samples

    print("Length: " + str(len(sampled_catalog_numbers)))

    # If less samples than requested due to rounding, sample more from the entire set
    while len(sampled_catalog_numbers) < num_samples:
        remaining_needed = num_samples - len(sampled_catalog_numbers)
        unique_catalog_numbers = df[column_name].dropna().unique()
        additional_samples = random.sample(list(unique_catalog_numbers), remaining_needed)
        sampled_catalog_numbers += additional_samples

    sampled_df = pd.DataFrame(sampled_catalog_numbers, columns=[column_name])
    sampled_df = sampled_df.drop_duplicates()
    output_file_path = f"data/stratified_sample_catalog_numbers_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sampled_df.to_csv(output_file_path, index=False)
    print("Stratified sampled catalog numbers saved to " + output_file_path)
    return output_file_path