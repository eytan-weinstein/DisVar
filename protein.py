class Protein:

    def __init__(self, sequence = None, uniprot_id = None):
        """
        Constructor for Protein class. Requires either an amino acid sequence
        or the UniProt ID of the protein for which the that amino acid 
        sequence can be retrieved.

        Parameters
        ----------
        sequence :  str OR None
            Amino acid sequence
        uniprot_id : str OR None
            UniProt ID of the protein
        """
        if sequence is None and uniprot_id is None:
            raise ValueError("Either sequence or uniprot_id must be provided.")
        if sequence is not None:
            self.sequence = sequence
        else:
            self.sequence = self._fetch_sequence_from_uniprot(uniprot_id)
    
    ###########################
    ###   Private methods   ###
    ###########################

    def _fetch_sequence_from_uniprot(self, uniprot_id):
        """
        Fetches the amino acid sequence from UniProt given a UniProt ID.

        Parameters
        ----------
        uniprot_id : str
            UniProt ID of the protein

        Returns
        -------
        sequence : str
            Amino acid sequence of the protein
        """
        # Placeholder for actual implementation to fetch sequence from UniProt
        # In a real implementation, this would involve making an HTTP request
        # to the UniProt API and parsing the response.
        raise NotImplementedError("Fetching sequence from UniProt is not implemented.")