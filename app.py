import os
import uuid
import pandas as pd
import requests
from flask import Flask, render_template, request, send_file, send_from_directory
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
import zipfile

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Get compound data from PubChem ===
def get_compound_data(name):
    try:
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
        res = requests.get(search_url)
        res.raise_for_status()
        cid = res.json()["IdentifierList"]["CID"][0]

        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
        smiles_res = requests.get(smiles_url)
        smiles_res.raise_for_status()
        smiles = smiles_res.json()["PropertyTable"]["Properties"][0]["IsomericSMILES"]

        return cid, smiles, "Found"
    except Exception as e:
        print(f"❌ {name}: Not Found ({e})")
        return None, None, "Not Found"

# === Save SDF and convert to PDB ===
def save_sdf_and_pdb(cid, name, sdf_dir, pdb_dir):
    sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
    sdf_path = os.path.join(sdf_dir, f"{name}_{cid}.sdf")
    pdb_path = os.path.join(pdb_dir, f"{name}_{cid}.pdb")
    try:
        res = requests.get(sdf_url)
        res.raise_for_status()
        with open(sdf_path, 'w') as f:
            f.write(res.text)
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if mol:
            AllChem.EmbedMolecule(mol)
            Chem.MolToPDBFile(mol, pdb_path)
            return sdf_path, pdb_path
        return sdf_path, "Error"
    except Exception as e:
        print(f"⚠️ {name} SDF/PDB error: {e}")
        return "Error", "Error"

# === Calculate Lipinski Rule ===
def calculate_lipinski(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return (
            round(Descriptors.MolWt(mol), 2),
            round(Crippen.MolLogP(mol), 2),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            sum([
                Descriptors.MolWt(mol) > 500,
                Crippen.MolLogP(mol) > 5,
                Lipinski.NumHDonors(mol) > 5,
                Lipinski.NumHAcceptors(mol) > 10
            ])
        )
    return [None] * 6

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['phytochem_file']
        if not file:
            return "⚠️ No file uploaded"

        # Create a unique folder for output
        session_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join("results", session_id)
        sdf_dir = os.path.join(output_dir, "SDF_files")
        pdb_dir = os.path.join(output_dir, "PDB_files")
        os.makedirs(sdf_dir)
        os.makedirs(pdb_dir)

        # Load Excel
        df = pd.read_excel(file)
        compounds = df.iloc[:, 0].dropna().tolist()

        results = []
        for name in compounds:
            cid, smiles, status = get_compound_data(name)
            sdf_path = pdb_path = ''
            mw = logp = hbd = hba = rot = violations = None

            if status == "Found":
                sdf_path, pdb_path = save_sdf_and_pdb(cid, name, sdf_dir, pdb_dir)
                mw, logp, hbd, hba, rot, violations = calculate_lipinski(smiles)

            results.append({
                "Compound Name": name,
                "PubChem CID": cid or '',
                "SMILES": smiles or '',
                "SDF Path": sdf_path,
                "PDB Path": pdb_path,
                "Status": status,
                "Molecular Weight": mw,
                "LogP": logp,
                "H-Bond Donors": hbd,
                "H-Bond Acceptors": hba,
                "Rotatable Bonds": rot,
                "Lipinski Violations": violations
            })

        final_df = pd.DataFrame(results)
        excel_out = os.path.join(output_dir, "phytochemicals_results.xlsx")
        final_df.to_excel(excel_out, index=False)

        # Zip all content
        zip_path = os.path.join("results", f"{session_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for folder, _, files in os.walk(output_dir):
                for file in files:
                    filepath = os.path.join(folder, file)
                    zipf.write(filepath, os.path.relpath(filepath, output_dir))

        return send_file(zip_path, as_attachment=True)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)