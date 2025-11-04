import aiohttp
import asyncio
import json
import requests

from Bio import Entrez
import metapredict


class Protein:

    def __init__(self, file_path = None, UniProt_ID = None, NCBI_email = None, NCBI_API_key = None):
        """
        Constructor for the Protein class. 

        Parameters
        ----------
        ** The following are required for initializing an instance of the Protein class from cached local files

        file_path : str OR None
            Path to a file with the data from a preexisting Protein class

        ** The following are required for initializing an instance of the Protein class from online records

        UniProt_ID : str OR None
            UniProt ID of the protein
        NCBI_email : str OR None
            Email for NCBI API access 
        NCBI_API_key : str OR None
            API key for NCBI API access
        """
        if file_path is not None:
            pass
            
        else:

            # Initialize Entrez parameters
            # Eytan's NCBI email and API key:
                # NCBI_email = 'eytan.weinstein@mail.utoronto.ca'
                # NCBI_api_key = '643e574e016033ae76f392a14a748cb6f808'
            Entrez.email = NCBI_email
            Entrez.api_key = NCBI_API_key

            # Point back to protein and gene IDs
            self.UniProt_ID = UniProt_ID
            self.Ensembl_gene_ID = self._map_UniProt_to_Ensembl(UniProt_ID)
            self.protein_name = self._fetch_protein_name()

            # Fetch amino acid sequence and coding sequence
            self.aa_sequence = self._fetch_aa_sequence_from_UniProt(UniProt_ID)
            self.coding_sequence = self._fetch_coding_sequence_from_Ensembl()

            # Annotate disordered regions
            self.disordered_regions = metapredict.predict_disorder_domains(self.aa_sequence).disordered_domain_boundaries

            # Annotate known missense variants
            self.missense_variants = self._fetch_dbSNP_missense_variants()

    ###########################
    ###   Private methods   ###
    ###########################

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
        URL_xref = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{UniProt_ID}?content-type=application/json"
        response_xref = requests.get(URL_xref, headers = {"Content-Type": "application/json"})
        if response_xref.status_code != 200:
            raise ValueError(f"Failed to map UniProt ID {UniProt_ID} to Ensembl (HTTP {response_xref.status_code})")
        data_xref = response_xref.json()
        if not data_xref:
            raise ValueError(f"No Ensembl mapping found for UniProt ID {UniProt_ID}")
        Ensembl_gene_ID = next((entry.get('id') for entry in data_xref if entry.get('type') == 'Gene'), data_xref[0]['id'])
        return Ensembl_gene_ID
    
    def _fetch_protein_name(self):
        """
        Fetches the correspoind protein name for an Ensembl gene ID.

        Returns
        -------
        protein_name : str
            Corresponding protein name
        """
        # Retrieve NCBI gene ID
        try:
            handle = Entrez.esearch(db = 'gene', term = f'{self.Ensembl_gene_ID}[All Fields]', retmax = 1)
            record = Entrez.read(handle)
            handle.close()
            gene_ID = record['IdList'][0]
        except:
            raise ValueError(f"No NCBI ID found for Ensembl ID {self.Ensembl_gene_ID}")
        
        # Retrieve protein name
        try:
            handle = Entrez.efetch(db = 'gene', id = gene_ID, retmode = 'xml')
            record = Entrez.read(handle)
            handle.close()
            protein_name = record[0]['Entrezgene_gene']['Gene-ref']['Gene-ref_locus']
        except:
            raise ValueError(f"No NCBI protein name found for Ensembl ID {self.Ensembl_gene_ID}")
        
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
        # Access UniProt REST API to fetch FASTA
        URL = f"https://rest.uniprot.org/uniprotkb/{UniProt_ID}.fasta"
        response = requests.get(URL)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch FASTA for {UniProt_ID} (HTTP {response.status_code})")

        # Parse FASTA to get sequence
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
        URL_lookup = f"https://rest.ensembl.org/lookup/id/{self.Ensembl_gene_ID}?expand=1;content-type=application/json"
        response_lookup = requests.get(URL_lookup, headers = {"Content-Type": "application/json"})
        if response_lookup.status_code != 200:
            raise ValueError(f"Failed to fetch transcripts for gene {self.Ensembl_gene_ID} (HTTP {response_lookup.status_code})")
        gene_data = response_lookup.json()
        transcripts = gene_data.get('Transcript', [])
        if not transcripts:
            raise ValueError(f"No transcripts found for gene {self.Ensembl_gene_ID}")

        # Choose canonical transcript if available, else first
        canonical_transcript = next((t for t in transcripts if t.get('is_canonical')), transcripts[0])
        transcript_ID = canonical_transcript['id']

        # Fetch CDS sequence from Ensembl
        URL_CDS = f"https://rest.ensembl.org/sequence/id/{transcript_ID}?type=cds;content-type=application/json"
        response_CDS = requests.get(URL_CDS, headers = {"Content-Type": "application/json"})
        if response_CDS.status_code != 200:
            raise ValueError(f"Failed to fetch CDS for transcript {transcript_ID} (HTTP {response_CDS.status_code})")
        data_cds = response_CDS.json()
        coding_sequence = data_cds.get('seq')
        if not coding_sequence:
            raise ValueError(f"No CDS found for transcript {transcript_ID}")

        # No need for stop codon sequence
        return coding_sequence[:-3]
    
    def _fetch_dbSNP_missense_variants(self):
        """
        Fetches known missense variants from dbSNP for the protein.

        Returns
        -------
        protein_missense_variants : dict
            Nested dictionary of missense variants by type
        """
        async def async_fetch():
            BASE_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
            BASE_URL = "https://rest.ensembl.org"

            async with aiohttp.ClientSession(headers = BASE_HEADERS) as session:

                # Initialize dictionary of missense variants by type
                protein_missense_variants = {'disordered': {}, 'folded': {}}

                # Retrieve all missense variants 
                URL_variants = f"{BASE_URL}/overlap/id/{self.Ensembl_gene_ID}?feature=variation"
                try:
                    async with session.get(URL_variants) as resp:
                        if resp.status != 200:
                            raise ValueError(f"Failed to access dbSNP data for {self.Ensembl_gene_ID} (HTTP {resp.status})")
                        variants = await resp.json()
                except Exception as e:
                    raise RuntimeError(f"Failed fetching variant list for {self.Ensembl_gene_ID}: {e}")
                missense_variants = [variant for variant in variants if 'missense_variant' in variant.get('consequence_type', [])]
                rsids = [variant['id'] for variant in missense_variants]

                # Helper function: batch POST for VEP data
                async def fetch_VEP_batch(rsids_batch):
                    url = f"{BASE_URL}/vep/human/id"
                    payload = {"ids": rsids_batch}
                    try:
                        async with session.post(url, data = json.dumps(payload)) as resp:
                            if resp.status != 200:
                                return [None] * len(rsids_batch)
                            return await resp.json()
                    except:
                        return [None] * len(rsids_batch)

                # Fetch VEP data in batches
                batch_size = 200
                VEP_batches = [rsids[i: i + batch_size] for i in range(0, len(rsids), batch_size)]
                VEP_tasks = [fetch_VEP_batch(batch) for batch in VEP_batches]
                VEP_results_batches = await asyncio.gather(*VEP_tasks)
                def extract_VEP(item):
                    transcript_consequences = item['transcript_consequences'][0]
                    aa_position = transcript_consequences.get('protein_start') or transcript_consequences.get('aa_start')
                    aa_change = transcript_consequences.get('amino_acids') 
                    if aa_position is not None and aa_change is not None:
                        return {'aa_position': aa_position, 'aa_change': aa_change}
                    return None
                VEP_results = [extract_VEP(item) for batch in VEP_results_batches for item in batch]

                # Clean VEP data (off by 1 indexing inconsistencies are common)
                for i in range(len(VEP_results)):
                    if VEP_results[i] is None:
                        continue
                    aa_position = VEP_results[i]['aa_position']
                    aa_change = VEP_results[i]['aa_change']
                    if self.aa_sequence[aa_position] != aa_change[0]:
                        if self.aa_sequence[aa_position - 1] == aa_change[0]:
                            aa_position -= 1
                        else:
                            continue

                    # Retrieve clinical significance annotations
                    clinical_significance = missense_variants[i]['clinical_significance']
                    if (len(clinical_significance) != 1) or (clinical_significance[0] not in ['pathogenic', 'likely pathogenic', 'benign', 'likely benign']):
                        clinical_significance = 'uncertain significance'
                    else:
                        clinical_significance = clinical_significance[0]
                    
                    # Determine if variant is in disordered or folded region
                    domain = 'folded'
                    for region in self.disordered_regions:
                        if (aa_position - 1 >= region[0]) and (aa_position < region[1]):
                            domain = 'disordered'
                            break

                    # Add variant to record
                    aa_change_str = str(aa_position + 1) + ';' + aa_change
                    protein_missense_variants[domain][aa_change_str] = clinical_significance
 
                return protein_missense_variants
            
        return asyncio.run(async_fetch())


if __name__ == "__main__":
    # Example usage
    protein = Protein(UniProt_ID = "O00571", NCBI_email = 'eytan.weinstein@mail.utoronto.ca', NCBI_API_key = '643e574e016033ae76f392a14a748cb6f808') # DDX3X