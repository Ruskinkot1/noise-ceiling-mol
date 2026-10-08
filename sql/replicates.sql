-- Измерения для одной мишени (параметр :target = CHEMBL id), только "чистые" точные значения.
-- Повторы между документами отбираются позже в Python (scripts/02), чтобы фильтры были параметризуемыми.
SELECT a.activity_id,
       td.chembl_id  AS target_chembl_id,
       td.pref_name  AS target_name,
       md.chembl_id  AS molecule_chembl_id,
       cs.canonical_smiles,
       a.standard_type,
       a.pchembl_value,
       ass.chembl_id AS assay_chembl_id,
       d.chembl_id   AS doc_chembl_id,
       d.year        AS doc_year
FROM activities a
JOIN assays ass            ON a.assay_id  = ass.assay_id
JOIN target_dictionary td  ON ass.tid     = td.tid
JOIN docs d                ON a.doc_id    = d.doc_id
JOIN molecule_dictionary md ON a.molregno = md.molregno
JOIN compound_structures cs ON a.molregno = cs.molregno
WHERE td.chembl_id = :target
  AND a.standard_type IN ('IC50', 'Ki')
  AND a.standard_relation = '='
  AND a.standard_units = 'nM'
  AND a.pchembl_value IS NOT NULL
  AND a.data_validity_comment IS NULL
  AND COALESCE(a.potential_duplicate, 0) = 0
  AND ass.assay_type = 'B'
  AND ass.confidence_score >= 8
  AND d.year IS NOT NULL;
