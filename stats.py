from collections import Counter
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from protein import Protein


BASEPATH = os.path.dirname(os.path.abspath(__file__))


##########################
###   Core functions   ###
##########################


def find_pathogenic_residues(group = 'disordered_exome', aa_change = False, plot_title = 'Pathogenic Missense Variants by Amino Acid, All Disordered Regions'):
    """
    Finds residues enriched for pathogenicity by log₂FC pathogenic/expected and log₂FC pathogenic/benign. 
    The results are returned as a table of enrichment scores, and a figure of multiple plots is generated.

    Parameters
    ----------
    group : str
        The group of proteins of interest. This must correspond to a folder name in data/mutational_frequencies 
    aa_change : bool
        If True, enrichments will be computed over amino acid changes (e.g. T/M) instead of the amino acids themselves
    plot_title : str
        The title to give to the figure of multiple plots
    
    Returns
    -------
    enrichment_scores : pd.DataFrame
        A data frame of enrichment scores for pathogenic/expected and pathogenic/benign
    fig : Figure
        Plots showing enrichment and relative and absolute frequencies of amino acids/amino acid changes within group
    """
    # Retrieve mutational frequencies
    mutational_frequencies = os.path.join(BASEPATH, 'data/mutational_frequencies')
    with open(os.path.join(mutational_frequencies, group, 'expected/expected.json'), 'r') as file:
        expected = json.load(file)
    with open(os.path.join(mutational_frequencies, group, 'observed/pathogenic.json'), 'r') as file:
        pathogenic = json.load(file)
    with open(os.path.join(mutational_frequencies, group, 'observed/benign.json'), 'r') as file:
        benign = json.load(file)
    
    # Combine by source amino acid
    if not aa_change:

        expected_combined = {}
        for k, v in expected.items():
            src = k.split('/')[0]
            expected_combined[src] = expected_combined.get(src, 0) + v
        expected = expected_combined

        pathogenic_combined = {}
        for k, v in pathogenic.items():
            src = k.split('/')[0]
            pathogenic_combined[src] = pathogenic_combined.get(src, 0) + v
        pathogenic = pathogenic_combined

        benign_combined = {}
        for k, v in benign.items():
            src = k.split('/')[0]
            benign_combined[src] = benign_combined.get(src, 0) + v
        benign = benign_combined
    
    # Calculate enrichment of pathogenic variants relative to expected mutability 
    pathogenic_to_expected_enrichment_scores = {}
    pathogenic_normalized = Protein()._normalize_mutational_frequencies(pathogenic)
    for k, v in pathogenic_normalized.items():
        eps = 1e-9
        pathogenic_to_expected_enrichment_scores[k] = np.log2((v + eps) / (expected[k] + eps))

    # Calculate enrichment of pathogenic variants relative to benign variants
    pathogenic_to_benign_enrichment_scores = {}
    total_pathogenic_count = sum(pathogenic.values())
    total_benign_count = sum(benign.values())
    for k, v in pathogenic.items():
        eps = 1e-9
        pathogenic_to_benign_enrichment_scores[k] = np.log2(((v + eps) / (benign[k] + eps)) / (((total_pathogenic_count - v) + eps) / ((total_benign_count - benign[k]) + eps)))
    
    # Calculate combined enrichment score
    combined_enrichment_score = {}
    for k, v in pathogenic.items():
        combined_enrichment_score[k] = pathogenic_to_expected_enrichment_scores[k] + pathogenic_to_benign_enrichment_scores[k]
    
    # Combine enrichment scores into final table
    enrichment_scores = pd.DataFrame([pathogenic_to_expected_enrichment_scores, pathogenic_to_benign_enrichment_scores, combined_enrichment_score]).T
    enrichment_scores.columns = ['log₂FC pathogenic/expected', 'log₂FC pathogenic/benign', 'log₂FC pathogenic/expected + log₂FC pathogenic/benign']
    enrichment_scores = enrichment_scores.sort_values(by = 'log₂FC pathogenic/expected + log₂FC pathogenic/benign', ascending = False)

    # Subplots
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex = True, figsize = (8, 11), gridspec_kw = {'height_ratios': [1, 2, 1]})
    fig.suptitle(plot_title, fontsize = 14, y = 0.98)
    fig.subplots_adjust(top = 0.92)  
    amino_acids = sorted(expected.keys())
    point_size = 40
    if aa_change:
        point_size = 20

    # Absolute frequency plot
    ax0.bar(amino_acids, [pathogenic[amino_acid] for amino_acid in amino_acids], color = 'lime', alpha = 0.7)
    ax0.set_ylabel('absolute frequency', fontsize = 10)
    ax0.tick_params(axis = 'x', labelbottom = True)
    if aa_change:
        ax0.set_xticks(range(len(amino_acids)))
        ax0.set_xticklabels(amino_acids, rotation = 90, fontsize = 5)
    
    # Relative frequency plot
    expected_y = [expected[a] for a in amino_acids]
    benign_normalized = Protein()._normalize_mutational_frequencies(benign)
    benign_normalized_y = [benign_normalized[a] for a in amino_acids]
    pathogenic_normalized_y = [pathogenic_normalized[a] for a in amino_acids]
    ax1.scatter(amino_acids, expected_y, s = point_size, color = 'blue', label = 'expected')
    ax1.scatter(amino_acids, pathogenic_normalized_y, s = point_size, color = 'red', label = 'pathogenic')
    ax1.scatter(amino_acids, benign_normalized_y, s = point_size, color = 'green', label = 'benign')
    ax1.tick_params(axis = 'x', labelbottom = True)
    if aa_change:
        ax1.set_xticks(range(len(amino_acids)))
        ax1.set_xticklabels(amino_acids, rotation = 90, fontsize = 5)
    ax1.set_ylabel('relative frequency', fontsize = 10)
    ax1.set_ylim(0, max(expected_y + benign_normalized_y + pathogenic_normalized_y) + 0.02)
    ax1.legend()

    # Enrichment plot
    ax2.scatter(amino_acids, [pathogenic_to_expected_enrichment_scores[a] for a in amino_acids], s = point_size, color = 'blue', label = 'log₂FC pathogenic/expected')
    ax2.scatter(amino_acids, [pathogenic_to_benign_enrichment_scores[a] for a in amino_acids], s = point_size, color = 'green', label = 'log₂FC pathogenic/benign')
    ax2.axhline(0.0, color = 'black', linestyle = ':')
    ax2.tick_params(axis = 'x', labelbottom = True)
    if aa_change:
        ax2.set_xticks(range(len(amino_acids)))
        ax2.set_xticklabels(amino_acids, rotation = 90, fontsize = 5)
    ax2.set_ylabel('enrichment', fontsize = 10)
    ax2.set_xlabel('amino acid', fontsize = 10)
    ax2.legend()

    plt.close(fig)

    return enrichment_scores, fig

def compute_exome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for the human exome.
    These are stored at stored at './data/mutational_frequencies/exome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    expected, all_observed, pathogenic_observed, benign_observed = {}, {}, {}, {}

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        expected = dict(Counter(expected) + Counter(protein.compute_null_expectation_mutational_frequencies()))
        all_variants = protein.missense_variants['disordered'] | protein.missense_variants['folded']
        all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
                
    _save_frequencies('exome', expected, all_observed, pathogenic_observed, benign_observed)

def compute_disordered_exome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for the disordered human exome.
    These are stored at stored at './data/mutational_frequencies/disordered_exome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    disordered_exome = ''
    all_observed, pathogenic_observed, benign_observed = _initialize_mutation_frequencies()

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        disordered_nt = [(start * 3, end * 3) for start, end in protein.disordered_regions]
        for start, end in disordered_nt:
            disordered_exome += protein.coding_sequence[start : end]
        all_variants = protein.missense_variants['disordered']
        all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
        
    expected = Protein().compute_null_expectation_mutational_frequencies(CDS = disordered_exome)
    print("The length of the disordered exome is {} bp.".format(len(disordered_exome)))
    _save_frequencies('disordered_exome', expected, all_observed, pathogenic_observed, benign_observed)

def compute_folded_exome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for the folded human exome.
    These are stored at stored at './data/mutational_frequencies/folded_exome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    folded_exome = ''
    all_observed, pathogenic_observed, benign_observed = _initialize_mutation_frequencies()

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
        all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
        
    expected = Protein().compute_null_expectation_mutational_frequencies(CDS = folded_exome)
    print("The length of the folded exome is {} bp.".format(len(folded_exome)))
    _save_frequencies('folded_exome', expected, all_observed, pathogenic_observed, benign_observed)


############################
###   Helper functions   ###
############################


def _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed):
    """
    Updates mutational frequency dictionaries based on observed amino acid changes in variants.

    Parameters
    ----------
    all_variants : dict[str:str]
        Dictionary mapping variants to their clinical classifications
    all_observed : dict[str:float]
        Dictionary storing counts for all variants
    pathogenic_observed : dict[str:float]
        Dictionary storing counts for pathogenic and likely pathogenic variants
    benign_observed : dict[str:float]
        Dictionary storing counts for benign and likely benign variants
    
    Returns
    -------
    tuple[dict[str:float]]
        all_observed, pathogenic_observed, benign_observed
    """
    for variant in all_variants:
        _, aa_change = Protein()._parse_aa_change(variant, whole_change = True)
        all_observed[aa_change] = all_observed.get(aa_change, 0) + 1
        if all_variants[variant] in ['pathogenic', 'likely pathogenic']:
            pathogenic_observed[aa_change] = pathogenic_observed.get(aa_change, 0) + 1
        elif all_variants[variant] in ['benign', 'likely benign']:
            benign_observed[aa_change] = benign_observed.get(aa_change, 0) + 1
    
    return all_observed, pathogenic_observed, benign_observed

def _save_frequencies(subdirectory, expected, all_observed, pathogenic_observed, benign_observed):
    """
    Saves expected and observed mutational frequencies as JSON files under './data/mutational_frequencies/{subdirectory}'.

    Parameters
    ----------
    basepath : str
        Subdirectory name under './data/mutational_frequencies' where results will be stored
    expected : dict[str:float]
        Expected mutational frequency dictionary
    all_observed : dict[str:float]
        Counts for all variants
    pathogenic_observed : dict[str:float]
        Counts for pathogenic and likely pathogenic variants
    benign_observed : dict[str:float]
        Counts for benign and likely benign variants
    """
    expected_basepath = os.path.join(BASEPATH, f'data/mutational_frequencies/{subdirectory}/expected')
    os.makedirs(expected_basepath, exist_ok = True)
    with open(os.path.join(expected_basepath, 'expected.json'), 'w') as f:
        json.dump(expected, f, indent = 4)
    observed_basepath = os.path.join(BASEPATH, f'data/mutational_frequencies/{subdirectory}/observed')
    os.makedirs(observed_basepath, exist_ok = True)
    with open(os.path.join(observed_basepath, 'all.json'), 'w') as f:
        json.dump(all_observed, f, indent = 4)
    with open(os.path.join(observed_basepath, 'pathogenic.json'), 'w') as f:
        json.dump(pathogenic_observed, f, indent = 4)
    with open(os.path.join(observed_basepath, 'benign.json'), 'w') as f:
        json.dump(benign_observed, f, indent = 4)