import os
import json
from multiprocessing import get_context, cpu_count
import argparse

# ========== LOAD HEAVY FILES ONCE IN PARENT ========= #

print("Loading gnomAD allele numbers...")
with open('/neuhaus/eytan/gnomAD_allele_numbers.json', 'r') as f:
    ALL_GNOMAD_ALLELE_NUMBERS = json.load(f)

print("Loading GTF...")
with open('/neuhaus/eytan/gencode.v44.annotation.gtf', 'r') as f:
    GTF_LINES = f.readlines()


# ========== WORKER FUNCTION (RUNS IN FORKED PROCESS) ========= #

def process_uid(uid, save_dir):
    try:
        from protein import Protein    # import inside worker to avoid pickling issues

        print(f"Processing {uid}...")
        file_path = os.path.join(save_dir, f"{uid}.json")

        protein = Protein(file_path=file_path)

        # Simplify missense variants
        try:
            protein.gnomAD_missense_variants = {
                'disordered': list(protein.gnomAD_missense_variants['disordered'].keys()),
                'folded': list(protein.gnomAD_missense_variants['folded'].keys())
            }
        except:
            pass

        protein.gnomAD_allele_numbers = protein._fetch_gnomAD_allele_numbers(
            path_to_gtf=GTF_LINES,
            all_gnomAD_allele_numbers=ALL_GNOMAD_ALLELE_NUMBERS
        )

        protein.save(save_dir=save_dir)
        print(f"Saved {uid}")

    except Exception as e:
        print(f"Error with {uid}: {e}")


# ========== MAIN ========== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot_file", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--workers", default=30, type=int)
    args = parser.parse_args()

    # Read UniProt IDs
    with open(args.uniprot_file) as f:
        ids = [line.strip().split("\t")[0] for line in f if line.strip()]

    print(f"Found {len(ids)} UniProt IDs.")

    # Create forked worker pool
    ctx = get_context("fork")
    pool = ctx.Pool(args.workers)

    # Dispatch jobs
    tasks = [(uid, args.save_dir) for uid in ids]
    pool.starmap(process_uid, tasks)

    pool.close()
    pool.join()

    print("All done.")