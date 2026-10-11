#!/usr/bin/env python
"""Черновик Croissant (core + RAI) для датасета из data/<папка>/*.molecules.csv.
URL размещения (HF/Dataverse/Kaggle/OpenML) ещё не существует: --base-url по умолчанию заглушка, файл НЕ готов к подаче.
Проверка: python scripts/make_croissant.py && python -c "import mlcroissant as m; m.Dataset('croissant/noise_ceiling_mol.json')" """
import argparse, glob, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/full_clean")
ap.add_argument("--base-url", default="https://TODO-hosting-not-created.invalid/noise-ceiling-mol")
ap.add_argument("--out", default="croissant/noise_ceiling_mol.json")
a = ap.parse_args()

ctx = {"@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
       "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/", "data": {"@id": "cr:data", "@type": "@json"},
       "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "dct": "http://purl.org/dc/terms/", "examples": {"@id": "cr:examples", "@type": "@json"},
       "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject", "fileSet": "cr:fileSet",
       "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath", "key": "cr:key",
       "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path", "recordSet": "cr:recordSet", "references": "cr:references",
       "equivalentProperty": "cr:equivalentProperty", "regex": "cr:regex", "samplingRate": "cr:samplingRate", "repeated": "cr:repeated", "replace": "cr:replace", "sc": "https://schema.org/", "separator": "cr:separator",
       "source": "cr:source", "subField": "cr:subField", "transform": "cr:transform"}
dist, rsets = [], []
for f in sorted(glob.glob(f"{a.data}/*.molecules.csv")):
    task = Path(f).name.replace(".molecules.csv", "")
    md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
    fid = f"{task}-csv"
    dist.append({"@type": "cr:FileObject", "@id": fid, "name": f"{task}.molecules.csv", "description": f"Molecules and document-averaged pChEMBL for task {task}",
                 "contentUrl": f"{a.base_url}/{task}.molecules.csv", "encodingFormat": "text/csv", "md5": md5})
    def fld(n, dt, d):
        return {"@type": "cr:Field", "@id": f"{task}/{n}", "name": n, "description": d, "dataType": dt,
                "source": {"fileObject": {"@id": fid}, "extract": {"column": n}}}
    rsets.append({"@type": "cr:RecordSet", "@id": task, "name": task, "description": f"Endpoint {task}: one row per standardized molecule",
                  "field": [fld("mol_id", "sc:Text", "InChIKey connectivity block (first 14 characters) of the standardized molecule"),
                            fld("smiles", "sc:Text", "Standardized canonical SMILES (largest fragment, neutralized)"),
                            fld("y", "sc:Float", "pChEMBL value, mean over documents"),
                            fld("n_docs", "sc:Integer", "Number of distinct ChEMBL documents reporting the molecule"),
                            fld("first_year", "sc:Integer", "Earliest publication year among those documents")]})
meta = {"@context": ctx, "@type": "sc:Dataset", "conformsTo": "http://mlcommons.org/croissant/1.0", "name": "noise-ceiling-mol",
        "description": "Replicate potency measurements (Ki, IC50) for the same molecules from different ChEMBL documents, standardized and "
                       "decontaminated against MoleculeNet/TDC, for estimating a noise ceiling in molecular property benchmarks.",
        "license": "https://creativecommons.org/licenses/by-sa/3.0/", "url": a.base_url, "version": "0.1.0",
        "citeAs": "TODO: not published; authors and DOI not assigned (anonymous for double-blind review)",
        "datePublished": "2026-10-11", "creator": {"@type": "sc:Person", "name": "TODO (anonymous for review)"},
        "isLiveDataset": False, "distribution": dist, "recordSet": rsets,
        "rai:dataCollection": "Extracted via the ChEMBL REST API (ChEMBL_37, released 2026-05-01): activities with standard_relation '=', pChEMBL present, empty data_validity_comment, biochemical assays (type B) with confidence_score >= 8, standard_type Ki or IC50, single-protein human targets. Structures standardized with RDKit (largest fragment, neutralization); values averaged within a document.",
        "rai:dataCollectionType": "Secondary data analysis of curated literature data (ChEMBL)",
        "rai:dataPreprocessingProtocol": "Decontamination: exact InChIKey matches (full and 14-character) and ECFP4 Tanimoto > 0.8 analogs of MoleculeNet and TDC molecules removed; Polaris not included (requires login). No document-year filter was applied; model pretraining corpora (PubChem/ZINC) were not checked (not verified).",
        "rai:dataBiases": "Literature bias toward well-studied targets and optimized chemical series; molecules with replicates in several documents are not representative of the whole dataset; inter-document differences include systematic assay offsets, not only random error.",
        "rai:dataLimitations": "Only 17 target-measurement-type tasks pass the replicate threshold; replicate molecules are a small fraction (about 1-6%) of each task; conditions of assays are not compared; stereoisomers are merged by InChIKey connectivity block; tautomers are not normalized.",
        "rai:personalSensitiveInformation": "None: molecular structures and bioactivity values only; no personal data.",
        "rai:dataUseCases": "Estimating noise ceilings and reporting model performance as a fraction of the ceiling; comparing fingerprint, graph and foundation-model baselines. Not intended for clinical decisions or compound prioritization without validation.",
        "rai:dataSocialImpact": "Low direct risk; could support more cautious interpretation of benchmark rankings. Misuse as a ground-truth potency reference should be avoided."}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(meta, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"{a.out}: {len(dist)} файлов, {len(rsets)} recordSet; URL размещения — заглушка")
