import aiohttp
import argparse
import asyncio
from functools import lru_cache
import gzip
from io import BytesIO
import json
import os
import pandas as pd
import re
import requests

import metapredict


AMINO_ACIDS = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

CODONS = {
    # Phenylalanine
    'TTT': 'F', 'TTC': 'F',
    # Leucine
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    # Isoleucine
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    # Methionine (Start)
    'ATG': 'M',
    # Valine
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    # Serine
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    # Proline
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Threonine
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Alanine
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Tyrosine
    'TAT': 'Y', 'TAC': 'Y',
    # Histidine
    'CAT': 'H', 'CAC': 'H',
    # Glutamine
    'CAA': 'Q', 'CAG': 'Q',
    # Asparagine
    'AAT': 'N', 'AAC': 'N',
    # Lysine
    'AAA': 'K', 'AAG': 'K',
    # Aspartic Acid
    'GAT': 'D', 'GAC': 'D',
    # Glutamic Acid
    'GAA': 'E', 'GAG': 'E',
    # Cysteine
    'TGT': 'C', 'TGC': 'C',
    # Tryptophan
    'TGG': 'W',
    # Arginine
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Glycine
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    # Stop codons
    'TAA': '*', 'TAG': '*', 'TGA': '*'
}


class Protein:

    ATTRIBUTES = ('save_dir',
        'UniProt_ID',
        'Ensembl_gene_ID',
        'protein_name',
        'aa_sequence',
        'coding_sequence',
        'disordered_regions',
        'missense_variants')

    def __init__(self, file_path = None, UniProt_ID = None):
        """
        Constructor for the Protein class.

        Parameters
        ----------
        ** The following is required for initializing an instance of the Protein class from cached local files

        file_path : str OR None
            Path to a file with the data from a preexisting Protein class

        ** The following is required for initializing an instance of the Protein class from online records

        UniProt_ID : str OR None
            UniProt ID of the protein

        ** Neither is necessary to instantiate an empty Protein object
        """
        if file_path is None and UniProt_ID is None:
            for name in self.ATTRIBUTES:
                setattr(self, name, None)

        elif file_path is not None:
            self._load(file_path)

        else:

            # Save directory is empty when initializing from online records
            self.save_dir = None

            # Point back to protein and gene IDs
            self.UniProt_ID = UniProt_ID
            self.Ensembl_gene_ID = self._map_UniProt_to_Ensembl(UniProt_ID)
            self.protein_name = self._fetch_protein_name()

            # Fetch amino acid sequence and coding sequence
            self.aa_sequence = self._fetch_aa_sequence_from_UniProt(UniProt_ID)
            self.coding_sequence = self._fetch_coding_sequence_from_Ensembl()

            # Annotate disordered regions
            self.disordered_regions = metapredict.predict_disorder_domains(self.aa_sequence).disordered_domain_boundaries

            # Annotate known missense variants from dbSNP and ClinVar
            self.missense_variants = self.annotate_missense_variants(self._fetch_dbSNP_missense_variants() | self._fetch_ClinVar_missense_variants())

            print(f"Protein {self.protein_name} ({self.UniProt_ID}) initialized with {len(self.missense_variants['disordered'])} disordered and {len(self.missense_variants['folded'])} folded missense variants.")

    ##########################
    ###   Public methods   ###
    ##########################

    def save(self, save_dir = None, custom_name = None):
        """
        Saves the Protein object to a file.

        Parameters
        ----------
        save_dir : str OR None
            Directory to save the file
        custom_name : str OR None
            Custom name for the file (if None, uses UniProt ID)
        """
        if save_dir is None and self.save_dir is None:
            raise ValueError("save_dir must be specified to save the Protein object.")

        # Add metadata
        final_json = self.missense_variants
        final_json['UniProt_ID'] = self.UniProt_ID
        final_json['Ensembl_gene_ID'] = self.Ensembl_gene_ID
        final_json['protein_name'] = self.protein_name
        final_json['aa_sequence'] = self.aa_sequence
        final_json['coding_sequence'] = self.coding_sequence
        final_json['disordered_regions'] = self.disordered_regions

        # Save by UniProt ID
        os.makedirs(save_dir, exist_ok = True)
        if custom_name is None:
            custom_name = self.UniProt_ID
        file_path = os.path.join(save_dir, f"{custom_name}.json")
        with open(file_path, 'w') as f:
            json.dump(final_json, f, indent = 4)

    ###########################
    ###   Private methods   ###
    ###########################

    def _load(self, file_path):
        """
        Loads a Protein object from a file.

        Parameters
        ----------
        file_path : str
            Path to the file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.UniProt_ID = data['UniProt_ID']
        self.Ensembl_gene_ID = data['Ensembl_gene_ID']
        self.protein_name = data['protein_name']
        self.aa_sequence = data['aa_sequence']
        self.coding_sequence = data['coding_sequence']
        self.missense_variants = {k: v for k, v in data.items() if k in ['disordered', 'folded']}
        self.disordered_regions = data['disordered_regions']
        self.save_dir = os.path.dirname(file_path)

    def _map_UniProt_to_Ensembl(self, UniProt_ID):
        """
        Maps a UniProt ID to an Ensembl gene ID.

        Parameters
        ----------
        UniProt_ID : str
            UniProt ID of the protein

        Returns
        -------
        Ensembl_gene_ID : str
            Corresponding Ensembl gene ID
        """
        URL = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{UniProt_ID}?content-type=application/json"
        response = requests.get(URL, headers = {"Content-Type": "application/json"})
        if not response.ok:
            raise ValueError(f"Failed to map UniProt ID {UniProt_ID} to Ensembl (HTTP {response.status_code})")
        data = response.json()
        if not data:
            raise ValueError(f"No Ensembl mapping found for UniProt ID {UniProt_ID}")
        Ensembl_gene_ID = next((entry.get('id') for entry in data if entry.get('type') == 'Gene'), data[0]['id'])
        return Ensembl_gene_ID

    def _fetch_protein_name(self):
        """
        Fetches the corresponding protein name for an Ensembl gene ID.

        Returns
        -------
        protein_name : str
            Corresponding protein name
        """
        URL = f"https://rest.ensembl.org/lookup/id/{self.Ensembl_gene_ID}"
        response = requests.get(URL, headers = {"Content-Type": "application/json"})
        if not response.ok:
            raise ValueError(f"No data found for Ensembl ID {self.Ensembl_gene_ID}")
        data = response.json()
        protein_name = data.get('display_name')
        if not protein_name:
            raise ValueError(f"No display_name found for Ensembl ID {self.Ensembl_gene_ID}")
        return protein_name

    def _fetch_aa_sequence_from_UniProt(self, UniProt_ID):
        """
        Fetches the amino acid sequence from UniProt given a UniProt ID.

        Parameters
        ----------
        UniProt_ID : str
            UniProt ID of the protein

        Returns
        -------
        aa_sequence : str
            Amino acid sequence of the protein
        """
        URL = f"https://rest.uniprot.org/uniprotkb/{UniProt_ID}.fasta"
        response = requests.get(URL)
        if not response.ok:
            raise ValueError(f"Failed to fetch FASTA for {UniProt_ID} (HTTP {response.status_code})")
        FASTA_lines = response.text.splitlines()
        aa_sequence = ''.join(line.strip() for line in FASTA_lines if not line.startswith('>'))
        return aa_sequence

    def _fetch_coding_sequence_from_Ensembl(self):
        """
        Fetches the coding DNA sequence from Ensembl given a UniProt ID

        Returns
        -------
        coding_sequence : str
            Coding sequence for the protein
        """
        # Get all transcripts for the gene
        URL = f"https://rest.ensembl.org/lookup/id/{self.Ensembl_gene_ID}?expand=1;content-type=application/json"
        response = requests.get(URL, headers = {"Content-Type": "application/json"})
        if not response.ok:
            raise ValueError(f"Failed to fetch transcripts for gene {self.Ensembl_gene_ID} (HTTP {response.status_code})")
        data = response.json()
        transcripts = data.get('Transcript', [])
        if not transcripts:
            raise ValueError(f"No transcripts found for gene {self.Ensembl_gene_ID}")

        # Choose canonical transcript if available, else first
        canonical_transcript = next((t for t in transcripts if t.get('is_canonical')), transcripts[0])
        transcript_ID = canonical_transcript['id']

        # Fetch CDS sequence from Ensembl
        URL = f"https://rest.ensembl.org/sequence/id/{transcript_ID}?type=cds;content-type=application/json"
        response = requests.get(URL, headers = {"Content-Type": "application/json"})
        if not response.ok:
            raise ValueError(f"Failed to fetch CDS for transcript {transcript_ID} (HTTP {response.status_code})")
        data = response.json()
        coding_sequence = data.get('seq')
        if not coding_sequence:
            raise ValueError(f"No CDS found for transcript {transcript_ID}")

        # No need for stop codon sequence
        return coding_sequence[:-3]

    def _fetch_dbSNP_missense_variants(self):
        """
        Fetches known missense variants from dbSNP for the protein.

        Returns
        -------
        VEP_data : dict[str: str]
            A dictionary mapping amino acid changes (e.g., "45;A/G") to clinical significance annotations
        """
        async def async_fetch():
            BASE_URL = "https://rest.ensembl.org"
            BASE_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
            sem = asyncio.Semaphore(5)

            async def safe_request(session, method, URL, **kwargs):
                # Perform a request with retries and exponential backoff
                for attempt in range(5):
                    try:
                        async with session.request(method, URL, **kwargs) as response:
                            # Retry if Ensembl says "Too Many Requests"
                            if response.status == 429:
                                wait = 2 ** attempt
                                print(f"Rate limited (429) → sleeping {wait}s and retrying...")
                                await asyncio.sleep(wait)
                                continue
                            response.raise_for_status()
                            return await response.json()
                    except Exception as e:
                        wait = 2 ** attempt
                        print(f"Request failed ({e}) → retrying in {wait}s...")
                        await asyncio.sleep(wait)
                raise RuntimeError(f"Failed after retries: {URL}")

            async with aiohttp.ClientSession(headers = BASE_HEADERS) as session:

                # Retrieve all missense variants
                URL = f"{BASE_URL}/overlap/id/{self.Ensembl_gene_ID}?feature=variation"
                variants = await safe_request(session, "GET", URL)
                missense_variants = [variant for variant in variants if 'missense_variant' in variant.get('consequence_type', [])]
                missense_variants = {variant['id']: variant['clinical_significance'] for variant in missense_variants}
                rsids = list(missense_variants.keys())

                # Helper function: batch POST for VEP data
                async def fetch_VEP_batch(rsids_batch):
                    async with sem:
                        URL = f"{BASE_URL}/vep/human/id"
                        payload = {"ids": rsids_batch}
                        await asyncio.sleep(0.1) # Brief pause to stay under 15 requests/second
                        try:
                            return await safe_request(session, "POST", URL, data = json.dumps(payload))
                        except Exception as e:
                            print(f"Batch failed ({len(rsids_batch)} rsIDs): {e}")
                            return [None] * len(rsids_batch)

                # Fetch VEP data in batches
                batch_size = 200
                VEP_batches = [rsids[i: i + batch_size] for i in range(0, len(rsids), batch_size)]
                VEP_results_batches = await asyncio.gather(*[fetch_VEP_batch(b) for b in VEP_batches])
                def extract_VEP(item):
                    transcript_consequences = item['transcript_consequences'][0]
                    aa_position = transcript_consequences.get('protein_start') or transcript_consequences.get('aa_start')
                    aa_change = transcript_consequences.get('amino_acids')
                    if aa_position is not None and aa_change is not None:
                        return str(aa_position) + ';' + aa_change
                    return None
                VEP_data = {}
                for batch in VEP_results_batches:
                    for item in batch:
                        if item is not None:
                            vep_info = extract_VEP(item)
                        if vep_info:
                            VEP_data[item['id']] = vep_info

                # Map to clinical significance annotations
                def map_clinical_significance(annotation):
                    if (len(annotation) != 1) or (annotation[0] not in ['pathogenic', 'likely pathogenic', 'benign', 'likely benign']):
                        annotation = 'uncertain significance'
                    else:
                        annotation = annotation[0]
                    return annotation
                VEP_data = {VEP_data[id]: map_clinical_significance(missense_variants[id]) for id in missense_variants if id in VEP_data}

                return VEP_data

        return asyncio.run(async_fetch())

    def _fetch_ClinVar_missense_variants(self):
        """
        Fetches known missense variants from ClinVar for the protein.
        These are added to the dbSNP missense variants, with the annotations from ClinVar taking priority in case of conflict.

        Returns
        -------
        VEP_data : dict[str: str]
            A dictionary mapping amino acid changes (e.g., "45;A/G") to clinical significance annotations
        """
        # Helper function to extract valid amino changes
        def extract_valid_amino_acid_change(s):
            match = re.search(r'\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)$', str(s).strip())
            if match:
                from_aa, pos, to_aa = match.groups()
                if from_aa != to_aa and from_aa in AMINO_ACIDS and to_aa in AMINO_ACIDS:
                    return f"{pos};{AMINO_ACIDS[from_aa]}/{AMINO_ACIDS[to_aa]}"
            return None

        # Fetch all variants from variant_summary.txt.gz (which must be in your working directory as ./data/variant_summary.txt)
        # Download periodically from https://www.ncbi.nlm.nih.gov/clinvar/docs/ftp_primer/ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz/
        df = self._fetch_ClinVar_variant_summary()
        df = df[(df['GeneSymbol'] == self.protein_name) & (df['Type'] == 'single nucleotide variant')].reset_index(drop = True)
        df['AminoAcidChange'] = df['Name'].apply(extract_valid_amino_acid_change)

        # Normalize clinical significance annotations
        CLIN_SIG_MAP = {
            'pathogenic': 'pathogenic',
            'likely pathogenic': 'likely pathogenic',
            'benign': 'benign',
            'likely benign': 'likely benign',
            'uncertain significance': 'uncertain significance',
            'pathogenic/likely pathogenic': 'pathogenic',
            'benign/likely benign': 'benign',
            'conflicting classifications of pathogenicity': 'uncertain significance'
        }
        df['ClinicalSignificance'] = df['ClinicalSignificance'].str.strip().str.lower().map(CLIN_SIG_MAP)
        df = df[df['ClinicalSignificance'].isin(['pathogenic', 'likely pathogenic', 'benign', 'likely benign', 'uncertain significance'])]
        VEP_data = {change: significance for change, significance in zip(df['AminoAcidChange'], df['ClinicalSignificance']) if change is not None}

        return VEP_data

    @staticmethod
    @lru_cache(maxsize = 1)
    def _fetch_ClinVar_variant_summary():
        """
        Fetches the latest ClinVar variant summary file from online or from the local cache.

        Returns
        -------
        variant_summary : pd.DataFrame
            The relevant protein-specific ClinVar variant data
        """
        variant_summary_path = './data/variant_summary.txt'
        os.makedirs(os.path.dirname(variant_summary_path), exist_ok = True)

        # This can be downloaded periodically as it's updated by NCBI
        try:
            df = pd.read_csv(variant_summary_path, sep = '\t', dtype = str)
        except:
            URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
            response = requests.get(URL)
            if not response.ok:
                raise ValueError(f"Failed to download ClinVar variant summary (HTTP {response.status_code})")
            with gzip.GzipFile(fileobj = BytesIO(response.content)) as f:
                df = pd.read_csv(f, sep = '\t', dtype = str)
                df.to_csv(variant_summary_path, sep = '\t', index = False)
        return df

    def annotate_missense_variants(self, missense_variants):
        """
        Annotates missense variants for disorder.

        Returns
        -------
        VEP_data : dict[str: str]
            A dictionary mapping amino acid changes (e.g., "45;A/G") to clinical significance annotations
        """
        missense_variants_by_disorder = {'disordered': {}, 'folded': {}}
        for aa_change, clinical_significance in missense_variants.items():

            # Verify that amino acid position is valid (off by 1 indexing inconsistencies are common)
            try:
                aa_position = int(aa_change.split(';')[0])
                original_aa = aa_change.split(';')[1].split('/')[0]
                final_aa = aa_change.split(';')[1].split('/')[1]
            except:
                continue
            try:
                true_position = self.aa_sequence[aa_position]
            except:
                true_position = None
            try:
                off_by_1 = self.aa_sequence[aa_position - 1]
            except:
                off_by_1 = None
            if true_position == original_aa:
                aa_position += 1
            elif off_by_1 == original_aa:
                pass
            else:
                continue

            # Discount substitutions of the start codon
            if aa_position == 1:
                continue

            # Annotate disorder
            domain = 'folded'
            for region in self.disordered_regions:
                if ((aa_position - 1) >= region[0]) and ((aa_position - 1) < region[1]):
                    domain = 'disordered'
                    break
            amended_aa_change = str(aa_position) + ';' + original_aa + '/' + final_aa
            missense_variants_by_disorder[domain][amended_aa_change] = clinical_significance
        
        return missense_variants_by_disorder 
    
    def _build_codon_substitution_table(self):
        """
        Utility function to build a null expectation codon substitution table.
        """
        # Null expectation codon substitution frequences from the Wellcome Sanger 1000 Genomes Project
        codon_substitutions = pd.read_csv("https://raw.githubusercontent.com/pjshort/dddMAPS/master/data/forSanger_1KG_mutation_rate_table.txt", sep = r'\s+')

        # Discount substitutions from start and stop codons
        codon_substitutions = codon_substitutions[~codon_substitutions['from'].isin(['ATG', 'TAA', 'TAG', 'TGA'])] 

        # Discount substitutions to stop codons
        codon_substitutions = codon_substitutions[~codon_substitutions['to'].isin(['TAA', 'TAG', 'TGA'])]

        # Map codons to amino acid substitutions
        codon_substitutions['aa_change'] = codon_substitutions.apply(lambda row: f"{CODONS[row['from']]}/{CODONS[row['to']]}", axis = 1)

        # Normalize codon substituiton frequencies
        codon_substitutions['mu_snp'] = codon_substitutions['mu_snp'] / codon_substitutions['mu_snp'].sum()

        # Transform to a dictionary mapping codons to their possible amino acid substitutions and their frequencies
        codon_substitutions = {codon: dict(zip(sub_df['aa_change'], sub_df['mu_snp'])) for codon, sub_df in codon_substitutions.groupby('from')}

        # Save to /data directory
        with open('./data/codon_substitutions.json', 'w') as f:
            json.dump(codon_substitutions, f, indent = 4)
    
    def _compute_null_expectation_mutational_frequencies(self, CDS):
        """
        Given a coding sequence, computes a normalized null expectation of the germline mutational frequencies.

        Parameters
        ----------
        CDS : str
            A coding sequence 

        Returns
        -------
        null_expectation_mutational_frequencies : dict[str:float]
            A dictionary mapping amino acid changes (e.g., "A/G") to their expected proportions among mutations annotated in that sequence 
        """
        # Ensure coding sequence can be separated by codons
        if len(CDS) < 3:
            raise Exception("CDS must have at least one codon.")
        CDS = CDS[:len(CDS) - len(CDS) % 3]

        # Compute a null expectation of the mutational frequencies by sliding along the sequence by codon
        with open('./data/codon_substitutions.json', 'r') as f:
            codon_substitutions = json.load(f)
        null_expectation_mutational_frequencies = {substitution: 0.0 for substitution in (k for v in codon_substitutions.values() for k in v.keys())}
        it = iter(CDS)
        for codon in zip(it, it, it):
            codon_str = ''.join(codon)
            if codon_str in codon_substitutions:
                for aa_change, mu in codon_substitutions[codon_str].items():
                    null_expectation_mutational_frequencies[aa_change] += mu
        
        # Normalize mutational frequencies
        total = sum(null_expectation_mutational_frequencies.values())
        null_expectation_mutational_frequencies = {k: v / total for k, v in null_expectation_mutational_frequencies.items()} if total > 0 else null_expectation_mutational_frequencies

        return null_expectation_mutational_frequencies


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description = "Fetch and save Protein objects from UniProt IDs.")
    parser.add_argument('--UniProt', required = True, nargs = '+', help = "One or more UniProt IDs separated by space.")    
    parser.add_argument('--save_dir', required = True, help = "Directory where Protein JSON files will be saved.")
    args = parser.parse_args()

    # Handle single ID or list of IDs
    uni_ids = [uid.strip() for uid in args.UniProt]

    # Save each Protein object
    for uid in uni_ids:
        try:
            print(f"Processing {uid}...")
            protein = Protein(UniProt_ID = uid)
            protein.save(save_dir = args.save_dir)
            print(f"Saved {uid} to {args.save_dir}")
        except:
            print(f"Could not build a Protein object for {uid}.")
            continue