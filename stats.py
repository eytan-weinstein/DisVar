import json
import os

from protein import Protein


##########################
###   Core functions   ###
##########################


def compute_exome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and actual mutational frequencies for the human exome.
    These are stored at stored at './data/mutational_frequencies/exome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    exome = ''
    all_actual, pathogenic_actual, benign_actual = _initialize_mutation_frequencies()

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        exome += protein.coding_sequence
        all_variants = protein.missense_variants['disordered'] | protein.missense_variants['folded']
        all_actual, pathogenic_actual, benign_actual = _update_variant_counts(all_variants, all_actual, pathogenic_actual, benign_actual)
        
    expected = Protein().compute_null_expectation_mutational_frequencies(CDS = exome)
    print("The length of the exome is {} bp.".format(len(exome)))
    _save_frequencies('exome', expected, all_actual, pathogenic_actual, benign_actual)

def compute_disordered_exome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and actual mutational frequencies for the disordered human exome.
    These are stored at stored at './data/mutational_frequencies/disordered_exome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    disordered_exome = ''
    all_actual, pathogenic_actual, benign_actual = _initialize_mutation_frequencies()

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        disordered_nt = [(start * 3, end * 3) for start, end in protein.disordered_regions]
        for start, end in disordered_nt:
            disordered_exome += protein.coding_sequence[start : end]
        all_variants = protein.missense_variants['disordered']
        all_actual, pathogenic_actual, benign_actual = _update_variant_counts(all_variants, all_actual, pathogenic_actual, benign_actual)
        
    expected = Protein().compute_null_expectation_mutational_frequencies(CDS = disordered_exome)
    print("The length of the disordered exome is {} bp.".format(len(disordered_exome)))
    _save_frequencies('disordered_exome', expected, all_actual, pathogenic_actual, benign_actual)

def compute_folded_exome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and actual mutational frequencies for the folded human exome.
    These are stored at stored at './data/mutational_frequencies/folded_exome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    folded_exome = ''
    all_actual, pathogenic_actual, benign_actual = _initialize_mutation_frequencies()

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        disordered_nt = [(start * 3, end * 3) for start, end in protein.disordered_regions]
        prev_end = 0
        for start, end in disordered_nt:
            if start > prev_end:
                folded_exome += protein.coding_sequence[prev_end : start]
            prev_end = end
        if prev_end < len(protein.coding_sequence):
            folded_exome += protein.coding_sequence[prev_end : ]
        all_variants = protein.missense_variants['folded']
        all_actual, pathogenic_actual, benign_actual = _update_variant_counts(all_variants, all_actual, pathogenic_actual, benign_actual)
        
    expected = Protein().compute_null_expectation_mutational_frequencies(CDS = folded_exome)
    print("The length of the folded exome is {} bp.".format(len(folded_exome)))
    _save_frequencies('folded_exome', expected, all_actual, pathogenic_actual, benign_actual)


############################
###   Helper functions   ###
############################


def _initialize_mutation_frequencies():
    """
    Initializes three mutational frequency dictionaries for all, pathogenic, and benign variants.

    Returns
    -------
    tuple[dict]
        A tuple containing three initialized mutational frequency dictionaries
    """
    return (Protein()._initialize_mutational_frequencies() for _ in range(3))

def _update_variant_counts(all_variants, all_actual, pathogenic_actual, benign_actual):
    """
    Updates mutational frequency dictionaries based on observed amino acid changes in variants.

    Parameters
    ----------
    all_variants : dict[str:str]
        Dictionary mapping variants to their clinical classifications
    all_actual : dict[str:float]
        Dictionary storing counts for all variants
    pathogenic_actual : dict[str:float]
        Dictionary storing counts for pathogenic and likely pathogenic variants
    benign_actual : dict[str:float]
        Dictionary storing counts for benign and likely benign variants
    
    Returns
    -------
    tuple[dict[str:float]]
        all_actual, pathogenic_actual, benign_actual
    """
    for variant in all_variants:
        _, aa_change = Protein()._parse_aa_change(variant, whole_change = True)
        if aa_change not in all_actual:
            continue
        all_actual[aa_change] += 1
        if all_variants[variant] in ['pathogenic', 'likely pathogenic']:
            pathogenic_actual[aa_change] += 1
        elif all_variants[variant] in ['benign', 'likely benign']:
            benign_actual[aa_change] += 1
    
    return all_actual, pathogenic_actual, benign_actual

def _save_frequencies(basepath, expected, all_actual, pathogenic_actual, benign_actual):
    """
    Saves expected and actual mutational frequencies as JSON files under './data/mutational_frequencies/{basepath}'.

    Parameters
    ----------
    basepath : str
        Subdirectory name under './data/mutational_frequencies' where results will be stored
    expected : dict[str:float]
        Expected mutational frequency dictionary
    all_actual : dict[str:float]
        Counts for all variants
    pathogenic_actual : dict[str:float]
        Counts for pathogenic and likely pathogenic variants
    benign_actual : dict[str:float]
        Counts for benign and likely benign variants
    """
    expected_basepath = f'./data/mutational_frequencies/{basepath}/expected'
    os.makedirs(expected_basepath, exist_ok = True)
    with open(os.path.join(expected_basepath, 'expected.json'), 'w') as f:
        json.dump(expected, f, indent = 4)
    actual_basepath = f'./data/mutational_frequencies/{basepath}/actual'
    os.makedirs(actual_basepath, exist_ok = True)
    with open(os.path.join(actual_basepath, 'all.json'), 'w') as f:
        json.dump(all_actual, f, indent = 4)
    with open(os.path.join(actual_basepath, 'pathogenic.json'), 'w') as f:
        json.dump(pathogenic_actual, f, indent = 4)
    with open(os.path.join(actual_basepath, 'benign.json'), 'w') as f:
        json.dump(benign_actual, f, indent = 4)