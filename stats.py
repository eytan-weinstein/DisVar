from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from protein import Protein


BASEPATH = os.path.dirname(os.path.abspath(__file__))


POSSIBLE_SNV_AA_CONSEQUENCES = {
'F/I': 0, 'F/L': 0, 'F/V': 0, 'F/Y': 0, 'F/S': 0, 'F/C': 0,
'L/I': 0, 'L/V': 0, 'L/S': 0, 'L/F': 0, 'L/M': 0, 'L/W': 0, 'L/H': 0, 'L/P': 0, 'L/R': 0, 'L/Q': 0,
'I/L': 0, 'I/V': 0, 'I/F': 0, 'I/N': 0, 'I/T': 0, 'I/S': 0, 'I/M': 0, 'I/K': 0, 'I/R': 0, 
'M/L': 0, 'M/V': 0, 'M/K': 0, 'M/T': 0, 'M/R': 0, 'M/I': 0,
'V/I': 0, 'V/L': 0, 'V/F': 0, 'V/D': 0, 'V/A': 0, 'V/G': 0, 'V/E': 0, 'V/M': 0,
'S/T': 0, 'S/P': 0, 'S/A': 0, 'S/Y': 0, 'S/C': 0, 'S/F': 0, 'S/L': 0, 'S/W': 0, 'S/R': 0, 'S/G': 0, 'S/N': 0, 'S/I': 0,
'P/T': 0, 'P/A': 0, 'P/S': 0,'P/H': 0, 'P/R': 0, 'P/L': 0, 'P/Q': 0,
'T/P': 0, 'T/A': 0, 'T/S': 0, 'T/N': 0, 'T/I': 0, 'T/K': 0, 'T/R': 0, 'T/M': 0,
'A/T': 0, 'A/P': 0, 'A/S': 0, 'A/D': 0, 'A/G': 0, 'A/V': 0, 'A/E': 0,
'Y/N': 0, 'Y/H': 0, 'Y/D': 0, 'Y/S': 0, 'Y/C': 0, 'Y/F': 0,
'H/N': 0, 'H/D': 0, 'H/Y': 0, 'H/P': 0, 'H/R': 0, 'H/L': 0, 'H/Q': 0,
'Q/K': 0, 'Q/E': 0, 'Q/P': 0, 'Q/R': 0, 'Q/L': 0, 'Q/H': 0,
'N/H': 0, 'N/D': 0, 'N/Y': 0, 'N/T': 0, 'N/S': 0, 'N/I': 0, 'N/K': 0,
'K/Q': 0, 'K/E': 0, 'K/T': 0, 'K/R': 0, 'K/I': 0, 'K/N': 0, 'K/M': 0,
'D/N': 0, 'D/H': 0, 'D/Y': 0, 'D/A': 0, 'D/G': 0, 'D/V': 0, 'D/E': 0,
'E/K': 0, 'E/Q': 0, 'E/A': 0, 'E/G': 0, 'E/V': 0, 'E/D': 0,
'C/S': 0, 'C/R': 0, 'C/G': 0, 'C/Y': 0, 'C/F': 0, 'C/W': 0, 
'W/R': 0, 'W/G': 0, 'W/S': 0, 'W/L': 0, 'W/C': 0,
'R/S': 0, 'R/G': 0, 'R/C': 0, 'R/H': 0, 'R/P': 0, 'R/L': 0, 'R/Q': 0, 'R/W': 0, 'R/K': 0, 'R/T': 0, 'R/I': 0, 'R/M': 0,
'G/S': 0, 'G/R': 0, 'G/C': 0, 'G/D': 0, 'G/A': 0, 'G/V': 0, 'G/E': 0, 'G/W': 0}


##########################
###   Core functions   ###
##########################


def find_pathogenic_residues(group = 'disordered_proteome', aa_change = False, plot_title = 'Pathogenic Missense Variants by Amino Acid, All Disordered Regions'):
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

def compute_proteome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for the human proteome.
    These are stored at stored at './data/mutational_frequencies/proteome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    expected, all_observed, pathogenic_observed, benign_observed = (deepcopy(POSSIBLE_SNV_AA_CONSEQUENCES) for _ in range(4))

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        null_expectation_mutational_frequencies = protein.compute_null_expectation_mutational_frequencies()
        expected = {k: expected[k] + null_expectation_mutational_frequencies.get(k, 0) for k in expected}
        all_variants = protein.missense_variants['disordered'] | protein.missense_variants['folded']
        all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
                
    _save_frequencies('proteome', Protein()._normalize_mutational_frequencies(expected), all_observed, pathogenic_observed, benign_observed)

def compute_disordered_proteome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for the disordered human proteome.
    These are stored at stored at './data/mutational_frequencies/disordered_proteome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    expected, all_observed, pathogenic_observed, benign_observed = (deepcopy(POSSIBLE_SNV_AA_CONSEQUENCES) for _ in range(4))

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        disordered_nt = [(start * 3, end * 3) for start, end in protein.disordered_regions]
        for start, end in disordered_nt:
            if start == 0:
                start = 3
            try:
                flanking_after = protein.coding_sequence[end + 1]
            except:
                flanking_after = 'T'
            disordered_sequence = protein.coding_sequence[start - 1 : end] + flanking_after
            null_expectation_mutational_frequencies = protein.compute_null_expectation_mutational_frequencies(CDS = disordered_sequence)
            expected = {k: expected[k] + null_expectation_mutational_frequencies.get(k, 0) for k in expected}
        all_variants = protein.missense_variants['disordered']
        all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
  
    _save_frequencies('disordered_proteome', Protein()._normalize_mutational_frequencies(expected), all_observed, pathogenic_observed, benign_observed)

def compute_folded_proteome_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for the folded human proteome.
    These are stored at stored at './data/mutational_frequencies/folded_proteome'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    expected, all_observed, pathogenic_observed, benign_observed = (deepcopy(POSSIBLE_SNV_AA_CONSEQUENCES) for _ in range(4))

    for UniProt_ID in os.listdir(proteome_dir):
        protein = Protein(file_path = os.path.join(proteome_dir, UniProt_ID))
        disordered_nt = [(start * 3, end * 3) for start, end in protein.disordered_regions]
        folded_sequences = []
        prev_end = 0
        for start, end in disordered_nt:
            if start > prev_end:
                if prev_end == 0:
                    prev_end = 3
                try:
                    flanking_after = protein.coding_sequence[start + 1]
                except:
                    flanking_after = 'T'
                folded_sequence = protein.coding_sequence[prev_end - 1 : start] + flanking_after
                folded_sequences.append(folded_sequence)
            prev_end = end
        if prev_end < len(protein.coding_sequence):
            if prev_end == 0:
                prev_end = 3
            folded_sequence = protein.coding_sequence[prev_end - 1 : ] + 'T'
            folded_sequences.append(folded_sequence)
        for folded_sequence in folded_sequences:
            null_expectation_mutational_frequencies = protein.compute_null_expectation_mutational_frequencies(CDS = folded_sequence)
            expected = {k: expected[k] + null_expectation_mutational_frequencies.get(k, 0) for k in expected}
        all_variants = protein.missense_variants['folded']
        all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
  
    _save_frequencies('folded_proteome', Protein()._normalize_mutational_frequencies(expected), all_observed, pathogenic_observed, benign_observed)

def compute_RGG_IDR_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for all disordered RGG-containing regions.
    These are stored at stored at './data/mutational_frequencies/RGG_IDRs'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    RGG_IDRs = {}
    with open(os.path.join(BASEPATH, 'data/RGG_IDRs.tsv')) as f:
        for line in f:
            RGG_IDR = line.strip().split('_')
            RGG_IDRs[RGG_IDR[0]] = [int(x) for x in RGG_IDR[2:]]
    
    expected, all_observed, pathogenic_observed, benign_observed = (deepcopy(POSSIBLE_SNV_AA_CONSEQUENCES) for _ in range(4))

    for UniProt_ID in RGG_IDRs:
        try:
            protein = Protein(file_path = os.path.join(proteome_dir, f'{UniProt_ID}.json'))
        except:
            continue
        disordered_nt = [((start - 1) * 3, (end - 1) * 3) for start, end in [RGG_IDRs[UniProt_ID]]]
        for start, end in disordered_nt:
            if start == 0:
                start = 3
            try:
                flanking_after = protein.coding_sequence[end + 1]
            except:
                flanking_after = 'T'
            disordered_sequence = protein.coding_sequence[start - 1 : end] + flanking_after
            null_expectation_mutational_frequencies = protein.compute_null_expectation_mutational_frequencies(CDS = disordered_sequence)
            expected = {k: expected[k] + null_expectation_mutational_frequencies.get(k, 0) for k in expected}
        all_variants = protein.missense_variants['disordered'] | protein.missense_variants['folded']
        all_variants = {k: v for k, v in all_variants.items() if (RGG_IDRs[UniProt_ID][0] <= int(k.split(';')[0]) <= RGG_IDRs[UniProt_ID][1])}
        all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
  
    _save_frequencies('RGG_IDRs', Protein()._normalize_mutational_frequencies(expected), all_observed, pathogenic_observed, benign_observed)

def compute_NARDINI_IDR_cluster_mutational_frequencies(proteome_dir):
    """
    Computes and then saves expected and observed mutational frequencies for all clusters of IDRs annotated by NARDINI (https://www.cell.com/cell/fulltext/S0092-8674(25)01191-2).
    These are stored at stored at './data/mutational_frequencies/NARDINI_IDRs_Cluster_{cluster}'

    Parameters
    ----------
    proteome_dir : str
        Path to the directory where Protein files for the human proteome have been downloaded by the main method of protein.py
    """
    NARDINI_IDR_clusters = pd.read_csv(os.path.join(BASEPATH, 'data/NARDINI_IDR_clusters.csv'))

    for cluster in range(0, 30):

        IDRs = NARDINI_IDR_clusters[NARDINI_IDR_clusters['Cluster Number'] == cluster]

        expected, all_observed, pathogenic_observed, benign_observed = (deepcopy(POSSIBLE_SNV_AA_CONSEQUENCES) for _ in range(4))

        for UniProt_ID, start_position, end_position in zip(IDRs['Uniprot'], IDRs['Start Pos'], IDRs['End Pos']):
            try: 
                protein = Protein(file_path = os.path.join(proteome_dir, f'{UniProt_ID}.json'))
                disordered_nt = [(start * 3, end * 3) for start, end in [[start_position, end_position]]]
                for start, end in disordered_nt:
                    if start == 0:
                        start = 3
                    try:
                        flanking_after = protein.coding_sequence[end + 1]
                    except:
                        flanking_after = 'T'
                    disordered_sequence = protein.coding_sequence[start - 1 : end] + flanking_after
                    null_expectation_mutational_frequencies = protein.compute_null_expectation_mutational_frequencies(CDS = disordered_sequence)
                    expected = {k: expected[k] + null_expectation_mutational_frequencies.get(k, 0) for k in expected}
                all_variants = protein.missense_variants['disordered'] | protein.missense_variants['folded']
                all_variants = {k: v for k, v in all_variants.items() if ((start_position + 1) <= int(k.split(';')[0]) <= (end_position + 1))}
                all_observed, pathogenic_observed, benign_observed = _update_variant_counts(all_variants, all_observed, pathogenic_observed, benign_observed)
            except:
                continue
        
        _save_frequencies(f'NARDINI_IDRs_Cluster_{cluster}', Protein()._normalize_mutational_frequencies(expected), all_observed, pathogenic_observed, benign_observed)

def compute_NARDINI_IDR_cluster_size(proteome_dir):
    NARDINI_IDR_clusters = pd.read_csv(os.path.join(BASEPATH, 'data/NARDINI_IDR_clusters.csv'))

    cluster_sizes = {}

    for cluster in range(0, 30):

        IDRs = NARDINI_IDR_clusters[NARDINI_IDR_clusters['Cluster Number'] == cluster]

        cluster_size = 0

        for UniProt_ID, start_position, end_position in zip(IDRs['Uniprot'], IDRs['Start Pos'], IDRs['End Pos']):
            try: 
                protein = Protein(file_path = os.path.join(proteome_dir, f'{UniProt_ID}.json'))
                disordered_nt = [(start * 3, end * 3) for start, end in [[start_position, end_position]]]
                for start, end in disordered_nt:
                    if start == 0:
                        start = 3
                    try:
                        flanking_after = protein.coding_sequence[end + 1]
                    except:
                        flanking_after = 'T'
                    disordered_sequence = protein.coding_sequence[start - 1 : end] + flanking_after
                    cluster_size += len(disordered_sequence) - 2
            except:
                continue
        
        cluster_sizes[cluster] = cluster_size
        

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
        if aa_change in all_observed:
            all_observed[aa_change] = all_observed[aa_change] + 1
        else:
            continue
        if all_variants[variant] in ['pathogenic', 'likely pathogenic']:
            pathogenic_observed[aa_change] = pathogenic_observed[aa_change] + 1
        elif all_variants[variant] in ['benign', 'likely benign']:
            benign_observed[aa_change] = benign_observed[aa_change] + 1
    
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