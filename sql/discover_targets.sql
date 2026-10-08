-- Ранжирование мишеней по числу молекул, измеренных минимум в 2 разных документах (один standard_type).
SELECT target_chembl_id, target_name, standard_type,
       COUNT(*) AS n_mols_2plus_docs, SUM(n_docs) AS n_doc_measurements
FROM (
  SELECT td.chembl_id AS target_chembl_id, td.pref_name AS target_name,
         a.standard_type, a.molregno, COUNT(DISTINCT a.doc_id) AS n_docs
  FROM activities a
  JOIN assays ass           ON a.assay_id = ass.assay_id
  JOIN target_dictionary td ON ass.tid = td.tid
  WHERE a.standard_type IN ('IC50','Ki') AND a.standard_relation = '='
    AND a.standard_units = 'nM' AND a.pchembl_value IS NOT NULL
    AND a.data_validity_comment IS NULL AND ass.assay_type = 'B'
    AND ass.confidence_score >= 8 AND td.target_type = 'SINGLE PROTEIN'
  GROUP BY td.chembl_id, a.standard_type, a.molregno
  HAVING COUNT(DISTINCT a.doc_id) >= 2
)
GROUP BY target_chembl_id, standard_type
HAVING COUNT(*) >= :min_mols
ORDER BY n_mols_2plus_docs DESC
LIMIT :top;
