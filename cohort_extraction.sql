{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww38200\viewh21120\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # After published the paper\
\
# removed patients who discharged withing first 24 hourse from first heparin dose. We need to remove that patients from our cohort, as we don't know what happened to those patients after dischrging)\
# removed ; danaproid and checked for all possible event_txt s.\
# checked again for drug type - ufh / lmwh\
# checked all possibilities were checked for event_txt.\
   \
 --#1 PATIENTS WHO WERE ADMINISTRATED BY ANY HEPARING GROUP MEDICATION\
  \
  WITH \
  FULL_COHORT AS\
\
   (SELECT pres.subject_id as emar_subject_id, pres.hadm_id as emar_hadm_id, EMAR.emar_id, EMAR.charttime AS emar_charttime, EMAR.scheduletime   \
   AS  emar_scheduletime,EMAR.storetime AS emar_storetime, EMAR.medication, EMAR.event_txt,\
      \
   PRES.subject_id as pres_subject_id, PRES.hadm_id as pres_hadm_id, pres.pharmacy_id,  PRES.starttime AS pres_starttime, PRES.stoptime AS \
   pres_stoptime, PRES.drug_type, PRES.drug, PRES.prod_strength, PRES.dose_val_rx, PRES.dose_unit_rx, PRES.form_val_disp, PRES.form_unit_disp,  \
   PRES.doses_per_24_hrs, PRES.route\
\
   FROM `physionet-data.mimic_hosp.emar` EMAR\
   LEFT JOIN `physionet-data.mimic_hosp.prescriptions` PRES -- as this 'left join', even rows with null 'hadm_id' s in emar, will be still included in the output.\
   ON EMAR.pharmacy_id = PRES.pharmacy_id\
\
   WHERE \
  -- selecting patients who were prescribed in Heparin group medications\
  -- why considered prescriptions? (could have used EMRA?) - difficult to caputure successful administration records in EMAR as there are lot of possible event_txt\
  -- but this ended up with removing all records in emar, where pharmacy_id is null. Therefore , had to consider 'medication' in 'emar', instead of 'drug' in 'prescriptions'. Successfull 'event_txt' were consider in latter steps.\
  -- But then realized, we considered the data from 'presriptions' table inorder to calssifiy : 'T'/ 'P'/'NA'/'TBD', and only 'T'/ 'P' records will be retained.\
  -- Therefore no use of rows with blank 'prescriptions' data. If we used 'medication' to fileter heparingroup medicine, then we will be ende up with lot of blank precrption data, ad there are blank 'pharmacy_id' for lot of rows in 'emar' table.\
  -- As we used 'prescription' table data in order to classify : 'T'/ 'P'/'NA'/'TBD', we coniderd 'drug' in prescrptions table to filter the heparin group medications.\
\
  (LOWER(PRES.DRUG) LIKE '%heparin%' OR\
  LOWER(PRES.DRUG) LIKE '%bemiparin%' OR\
  LOWER(PRES.DRUG) LIKE '%dalteparin%' OR\
  LOWER(PRES.DRUG) LIKE '%danaparoid%' OR -- remove this. this is non-heparin anticogulant use to treat HIT\
  LOWER(PRES.DRUG) LIKE '%enoxaparin%' OR\
  LOWER(PRES.DRUG) LIKE '%nadroparin%' OR\
  LOWER(PRES.DRUG) LIKE '%parnaparin%' OR\
  LOWER(PRES.DRUG) LIKE '%reviparin%' OR\
  LOWER(PRES.DRUG) LIKE '%tinzaparin%')\
\
  )\
\
  --#2 EXTRATCING DITINCT ADMINISTRATIONS (MAINLY CONSIDERING MEDICATION, DOSE AND ROUTE) AND CORRESPONDING ADMINISTRATIONS COUNT, TO DECIDE THERAPEAUTIC OR PROPHYLACTIC (THIS IS ANALYSED BY EXPORTING AS AN EXCEL)\
\
  ,T_OR_P_ANALYSIS AS --(T - THERAPUEATIC, P - PROPHYLACTIC)\
  (SELECT medication, event_txt, drug_type, drug, prod_strength, dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route, COUNT((medication, event_txt, drug_type, drug, prod_strength, dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route)) AS ROW_COUNT FROM FULL_COHORT\
GROUP BY medication, event_txt, drug_type, drug, prod_strength, dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route\
ORDER BY ROW_COUNT DESC\
  )\
\
--#3 AFTER ANALYSING IN EXCEL, CLASSIFY PATIENTS\
  -- MISSED --> 'O' ( -- CHECK --> 'O')\
  -- N/A CASES (HPEARIN FLUSH OR NO ENOUGH DATA) -> '1'\
  -- Prophylactic --> '2'\
  -- THERAPEUTIC --> '3'\
 \
  ,T_OR_P_CLASSIFY AS \
        (SELECT T_OR_P_ANALYSIS.*,\
         CASE\
             WHEN \
                 (UPPER(T_OR_P_ANALYSIS.medication) LIKE '%FLUSH%' OR UPPER(T_OR_P_ANALYSIS.drug) LIKE '%FLUSH%') \
                 THEN 'N/A'\
\
             WHEN \
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'DIALYS' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'DWELL' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'IMPELLA' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'FEMORAL VEIN' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'PO' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'NG' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'PO/NG' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'NG/PO' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'TP' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'IP' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'LOCK' OR\
                  UPPER(T_OR_P_ANALYSIS.ROUTE) LIKE 'DLPICC'\
                  THEN 'N/A'\
            \
             WHEN \
                   (T_OR_P_ANALYSIS.dose_val_rx = '0' OR T_OR_P_ANALYSIS.dose_val_rx IS NULL) AND\
                   (T_OR_P_ANALYSIS.form_val_disp = '0' OR T_OR_P_ANALYSIS.form_val_disp IS NULL) -- INCORRECT RECORDS WHERE DOSE = 0\
                   THEN 'N/A'\
\
             WHEN \
                 ((UPPER(T_OR_P_ANALYSIS.dose_unit_rx) LIKE '%UNIT%') OR (UPPER(T_OR_P_ANALYSIS.dose_unit_rx) = 'IU')) AND\
                 (T_OR_P_ANALYSIS.route = 'IV' OR T_OR_P_ANALYSIS.route = 'IV DRIP' OR T_OR_P_ANALYSIS.route = 'IV BOLUS')\
                 THEN 'T'\
\
             WHEN \
                 ((UPPER(T_OR_P_ANALYSIS.dose_unit_rx) LIKE '%UNIT%') OR (UPPER(T_OR_P_ANALYSIS.dose_unit_rx) = 'IU')) AND\
                 (T_OR_P_ANALYSIS.route = 'SC' OR T_OR_P_ANALYSIS.route = 'SUBCUT')\
                 THEN 'P'\
\
\
             WHEN \
                  UPPER(T_OR_P_ANALYSIS.dose_unit_rx) LIKE '%MG%' AND\
                  SAFE_CAST(T_OR_P_ANALYSIS.dose_val_rx AS NUMERIC) <= 40 AND\
                  T_OR_P_ANALYSIS.doses_per_24_hrs = 1\
                  THEN 'P'    \
                           \
             WHEN \
                  UPPER(T_OR_P_ANALYSIS.dose_unit_rx) LIKE '%MG%' AND\
                  SAFE_CAST(T_OR_P_ANALYSIS.dose_val_rx AS NUMERIC) <= 40 AND\
                  T_OR_P_ANALYSIS.doses_per_24_hrs = 2\
                  THEN 'T'  \
              \
             WHEN \
                  UPPER(T_OR_P_ANALYSIS.dose_unit_rx) LIKE '%MG%' AND\
                  SAFE_CAST(T_OR_P_ANALYSIS.dose_val_rx AS NUMERIC) <= 40 AND\
                  ((T_OR_P_ANALYSIS.doses_per_24_hrs IS NULL) OR (T_OR_P_ANALYSIS.doses_per_24_hrs = 0))\
                  THEN 'N/A' \
\
             WHEN \
                  UPPER(T_OR_P_ANALYSIS.dose_unit_rx) LIKE '%MG%' AND\
                  SAFE_CAST(T_OR_P_ANALYSIS.dose_val_rx AS NUMERIC) > 40 \
                  THEN 'T' \
\
\
             ELSE \
                  'TBD' --RECORDS WHICH WERE NOT FALLEN INTO ANY OF THE CATEGORIES\
\
         END AS CLASSIFICATION\
         FROM T_OR_P_ANALYSIS\
        )\
\
--#4 Join of #1 and #3. This classifies each heaprin adminitration as 'T' / 'P' / 'N/A' and extracts reords with  'T' / 'P'.Also this add whether the each heparin administered is LMWH or UFH. \
\
  ,FULL_COHORT_HEPARIN_ADMINISTERED_WITHOUT_CONSIDERING_PLATELETS AS  \
        (SELECT FULL_COHORT.emar_subject_id, FULL_COHORT.emar_hadm_id, FULL_COHORT.PHARMACY_ID, FULL_COHORT.emar_charttime, FULL_COHORT.emar_scheduletime,   \
        FULL_COHORT.emar_storetime, FULL_COHORT.medication, FULL_COHORT.event_txt,\
        FULL_COHORT.pres_starttime, FULL_COHORT.pres_stoptime, \
        T_OR_P_CLASSIFY.*except(medication, event_txt) \
\
        ,CASE  -- UNFRACTIONED OR LMW HEPARIN  (TRY WITH 'IN', INSTEAD OF DIFFERENT 'OR')\
\
        WHEN (lower(T_OR_P_CLASSIFY.DRUG) like 'heparin%') --T_OR_P_CLASSIFY.DRUG in ( 'Heparin', 'Heparin Sodium', 'heparin (porcine)', 'Heparin CRRT', 'Heparin (IABP)', "Heparin (Impella)",  "Heparin (Hemodialysis)")) \
        THEN\
           'UFH'  --UNFRACTIONED HEPARIN\
\
        WHEN ((lower(T_OR_P_CLASSIFY.DRUG) like '%enoxaparin%') or (lower(T_OR_P_CLASSIFY.DRUG) like '%dalteparin%')) --T_OR_P_CLASSIFY.DRUG in ( 'Heparin', 'Heparin Sodium', 'heparin (porcine)', 'Heparin CRRT', 'Heparin (IABP)', "Heparin (Impella)",  "Heparin (Hemodialysis)")) \
        THEN\
           'LMWH'  --UNFRACTIONED HEPARIN\
\
        ELSE\
          'NA'  -- LOW MOLECULAR WEIGHT HEPARIN # Enoxaparin, Dalteparin\
\
        END AS HEPARIN_TYPE\
\
      , row_number() over(\
        partition by emar_hadm_id\
        order by emar_charttime asc, classification desc) as rn\
        # earliest event appraes first. In next step, only rn=1 records will be ratained. Whey we do this? because some hadm_ids have multiple first_hep admin records in emar. For eaxmple, for emar_hadm_id - 25731558 - had two first hep admin records, where classification / treatment_type of one is 'P' and other one is 'T' - When we partition as in above line, we take the first record (if have multiple first records) with 'T' (as we filtered 'T' or 'P' by descending order)\
\
        FROM FULL_COHORT\
        LEFT JOIN T_OR_P_CLASSIFY --This join links hadm_id with the classification category(T, P or N/A)\
        ON\
(\
\
  ((FULL_COHORT.medication = T_OR_P_CLASSIFY.medication) OR (FULL_COHORT.medication is null and T_OR_P_CLASSIFY.medication is null)) AND\
  ((FULL_COHORT.event_txt = T_OR_P_CLASSIFY.event_txt) OR (FULL_COHORT.event_txt is null and T_OR_P_CLASSIFY.event_txt is null)) AND\
  ((FULL_COHORT.drug_type = T_OR_P_CLASSIFY.drug_type) OR (FULL_COHORT.drug_type is null and T_OR_P_CLASSIFY.drug_type is null)) AND\
  ((FULL_COHORT.drug = T_OR_P_CLASSIFY.drug) OR (FULL_COHORT.drug is null and T_OR_P_CLASSIFY.drug is null)) AND\
  ((FULL_COHORT.prod_strength = T_OR_P_CLASSIFY.prod_strength) OR (FULL_COHORT.prod_strength is null and T_OR_P_CLASSIFY.prod_strength is null)) AND\
  ((FULL_COHORT.dose_val_rx = T_OR_P_CLASSIFY.dose_val_rx) OR (FULL_COHORT.dose_val_rx is null and T_OR_P_CLASSIFY.dose_val_rx is null)) AND\
  ((FULL_COHORT.dose_unit_rx = T_OR_P_CLASSIFY.dose_unit_rx) OR (FULL_COHORT.dose_unit_rx is null and T_OR_P_CLASSIFY.dose_unit_rx is null)) AND\
  ((FULL_COHORT.form_val_disp = T_OR_P_CLASSIFY.form_val_disp) OR (FULL_COHORT.form_val_disp is null and T_OR_P_CLASSIFY.form_val_disp is null)) AND\
  ((FULL_COHORT.form_unit_disp = T_OR_P_CLASSIFY.form_unit_disp) OR (FULL_COHORT.form_unit_disp is null and T_OR_P_CLASSIFY.form_unit_disp is null)) AND\
  ((FULL_COHORT.doses_per_24_hrs = T_OR_P_CLASSIFY.doses_per_24_hrs) OR (FULL_COHORT.doses_per_24_hrs is null and T_OR_P_CLASSIFY.doses_per_24_hrs is null)) AND\
  ((FULL_COHORT.route = T_OR_P_CLASSIFY.route) OR (FULL_COHORT.route is null and T_OR_P_CLASSIFY.route is null)) \
\
)\
\
          WHERE (T_OR_P_CLASSIFY.CLASSIFICATION = 'T' OR T_OR_P_CLASSIFY.CLASSIFICATION = 'P') -- There are two 'TBD' records for emar_hadm_id - 21582209, but no need to bother about it, as there were rows with 'T' for that emar_hadm_id, before 'TBD' records (later we filtered only the first record of each emar_hadm_id)\
        AND\
        # condition 1 - coz for some hadm_ids, all hep related records from emar table, had event_txt for unsuccessful administrations.\
        #               This issue had hadm_id count = 149 (ex: hadm_id -24975936, 29137563)\
\
         \
        #To check these 149 hadm_ids,\
/*\
SELECT distinct emar_hadm_id from FULL_COHORT_HEPARIN_ADMINISTERED_WITHOUT_CONSIDERING_PLATELETS\
where event_txt IN ("Delayed Administered" , "Stopped", "Not Given", "Stopped - Unscheduled", "Hold Dose", "Documented in O.R. Holding") and \
\
emar_hadm_id not in\
(SELECT distinct emar_hadm_id from FULL_COHORT_HEPARIN_ADMINISTERED_WITHOUT_CONSIDERING_PLATELETS\
where \
event_txt IN ("Administered" , "Confirmed", "Started", "Restarted", "Administered in Other Location", " in Other Location"))\
*/\
\
# All events - Administered , Delayed Administered , Stopped , Confirmed , Not Given , Restarted , Stopped - Unscheduled , Hold Dose, Started ,  in Other Location , Administered in Other Location , Documented in O.R. Holding\
\
# sucessful - Administered ,Confirmed , Restarted , Started ,  in Other Location , Administered in Other Location\
\
# unsucessful - Delayed Administered , Stopped , Not Given ,Stopped - Unscheduled , Hold Dose , Documented in O.R. Holding\
\
# all - distinct hadm_id = 23211\
# row count - 346585\
\
# successful - distinct hadm_id = 23062\
# row count - 343145\
\
# unsuccessful - distinct hadm_id = 2344\
# row count - 3439\
\
-- event_txt that reflects sucessful administrations - may be some others as well.\
\
       -- FULL_COHORT.event_txt IN ("Administered" , "Confirmed", "Started", "Restarted", "Administered in Other Location", " in Other Location", "Delayed Administered", "Delayed Started")\
\
\
(\
trim(FULL_COHORT.event_txt) not in ('Hold Dose', 'Not Confirmed', 'Removed', 'Not Started', 'Not Applied', 'Delayed', 'Not Assessed','Documented in O.R. Holding', 'Flushed in Other Location', 'Delayed Flushed', 'Delayed Not Confirmed', 'Flushed','Not Given', 'Delayed Not Started', 'Stopped - Unscheduled', 'Stopped in Other Location','Stopped - Unscheduled in Other Location', 'Stopped As Directed', 'Delayed Stopped', 'Stopped', 'Delayed Stopped As Directed', 'Infusion Reconciliation', 'Infusion Reconciliation Not Done')\
\
or FULL_COHORT.event_txt is null\
\
)\
        AND\
\
        # condition 2 - coz in returned cohort, in emar, medications are sometimes non hep. This happened coz we just did a left join from presciptions to emar on pharmacy_id, and we only considered hep drugs in prescriptions (step 1 - full cohort). So in this case, hep related pharmacy_id in prescriptions table may have either in hep related medication or non hep medication (rarely) for the same pharmacy_id in emar.\
# As we are checking for hep admins from emar table (coz charttimes are not mentioned in prescriptions) , administrations other than heparin should be excluded.      \
\
# hadm_id count - 69 (but some of these hadm_id may have hep medications too in emar), rows - 92 (when exclude condition 1)\
# ex - pharmacy_id - 66443739\
# to check rows\
/*select * from FULL_COHORT_HEPARIN_ADMINISTERED_WITHOUT_CONSIDERING_PLATELETS\
where \
  (LOWER(medication) NOT LIKE '%heparin%' AND\
  LOWER(medication) NOT LIKE '%bemiparin%' AND\
  LOWER(medication) NOT LIKE '%dalteparin%' AND\
  --LOWER(medication) NOT LIKE '%danaparoid%' AND\
  LOWER(medication) NOT LIKE '%enoxaparin%' AND\
  LOWER(medication) NOT LIKE '%nadroparin%' AND\
  LOWER(medication) NOT LIKE '%parnaparin%' AND\
  LOWER(medication) NOT LIKE '%reviparin%' AND\
  LOWER(medication) NOT LIKE '%tinzaparin%')\
*/\
\
# 36 hadm_ids didn't have any hep medication in emar, though the drugs prescibed in presciptions table\
# To check for those hadm_ids,\
/*\
select distinct emar_hadm_id from FULL_COHORT\
where emar_hadm_id not in\
(select distinct emar_hadm_id from FULL_COHORT\
where\
 (LOWER(medication)  LIKE '%heparin%' OR\
  LOWER(medication)  LIKE '%bemiparin%' OR\
  LOWER(medication)  LIKE '%dalteparin%' OR\
  LOWER(medication)  LIKE '%danaparoid%' OR\
  LOWER(medication)  LIKE '%enoxaparin%' OR\
  LOWER(medication)  LIKE '%nadroparin%' OR\
  LOWER(medication)  LIKE '%parnaparin%' OR\
  LOWER(medication)  LIKE '%reviparin%' OR\
  LOWER(medication)  LIKE '%tinzaparin%'))\
  */\
\
      ( LOWER(FULL_COHORT.medication) LIKE '%heparin%' OR\
        LOWER(FULL_COHORT.medication) LIKE '%bemiparin%' OR\
        LOWER(FULL_COHORT.medication) LIKE '%dalteparin%' OR\
        --LOWER(FULL_COHORT.medication) LIKE '%danaparoid%' OR -- remove this. this is non-heparin anticogulant use to treat HIT\
        LOWER(FULL_COHORT.medication) LIKE '%enoxaparin%' OR\
        LOWER(FULL_COHORT.medication) LIKE '%nadroparin%' OR\
        LOWER(FULL_COHORT.medication) LIKE '%parnaparin%' OR\
        LOWER(FULL_COHORT.medication) LIKE '%reviparin%' OR\
        LOWER(FULL_COHORT.medication) LIKE '%tinzaparin%')\
\
)\
\
#--------------------------------------------------------------------------------------------------------------------------------------------------\
\
#  We need to Take first heaprin administration for each hospital admission. No null 'emar_hadm_id' s in 'FULL_COHORT_HEPARIN_ADMINISTERED_WITHOUT_CONSIDERING_PLATELETS', as considered 'hadm_id' fomr 'prescriptions' table. in 'full_cohort' at very begining (no null 'hadm_id'/'subject_id'/'pharmacy_id' in 'prescriptions' table).\
\
, first_hep AS  -- take first heaprin admin for each hadm_id (this admin can happen before admitting to icu, or within the icu - may be in the first icu stay or during a subsequent icu stay)\
(\
select emar_hadm_id, emar_charttime as hep_start, classification as treatment_types, heparin_type as hep_types, event_txt, drug\
from FULL_COHORT_HEPARIN_ADMINISTERED_WITHOUT_CONSIDERING_PLATELETS\
where rn = 1\
)\
\
#-------------------------\
\
, first_icu_before_first_hep as  --here we get icu stay Intime of the corresponding icustay, where the first heaparin dose was given, We cannot get stay_ID of the particular icu stay, because of the way we grouped. We do it in the next step.\
\
(\
  select hadm_id, min(icu.intime) as icu_in_time_first_hep from first_hep \
  \
  --to get the first icu stay details (when the admission has multiple icustays) where the patient received first heparin dose DURING that icustay \
\
  left join `physionet-data.mimic_icu.icustays` icu\
  on first_hep.emar_hadm_id = icu.hadm_id\
\
  where hep_start BETWEEN intime AND outtime --heparin dose reecived after the patient was admitted to the ICU, and before dischargning from ICU\
\
  group by hadm_id\
)\
\
#-------------------------\
\
, first_hep_with_hep_type_and_treatment_type_dermographics as  --# one row per one hadm_id, , row count - 13416, distcint hadm_id count - 13416\
(\
  select \
  \
  ADM.subject_id AS subject_id\
\
  -- icu stay-related data of first heparin dose\
, first_icu_before_first_hep.hadm_id, stay_id, hep_start, icu_in_time_first_hep, ICU.outtime as icu_out_time_first_hep, ICU.first_careunit , ICU.last_careunit, adm.admittime, adm.dischtime\
\
  -- first heparin administration related details - treatmemt type(theraputic/ prophylactic) and heaprin type (UFH/ LMWH) of 1st dose of heparin admin for each hadm_id\
, treatment_types, hep_types, event_txt, drug\
\
-- admission details\
, ADM.admittime, ADM.dischtime, ADM.admission_type, ADM.admission_location, ADM.hospital_expire_flag --IF '1', PATIENT DIED DURING THAT ADMISSION\
\
-- Join dermographics\
, PAT.gender, PAT.anchor_age\
\
  from first_icu_before_first_hep -- to get icu intime of first_hep admin of each hadm_id\
  left join first_hep -- to get 'first_hep_admin_time' of each hadm_id\
  on first_icu_before_first_hep.hadm_id = first_hep.emar_hadm_id \
  \
  left join `physionet-data.mimic_icu.icustays` ICU\
  on first_icu_before_first_hep.hadm_id = icu.hadm_id\
\
  left join `physionet-data.mimic_core.admissions` ADM\
  on first_icu_before_first_hep.hadm_id = adm.hadm_id\
\
  left join `physionet-data.mimic_core.patients` PAT\
  on ADM.subject_id = PAT.subject_id\
 \
  where\
  (first_icu_before_first_hep.icu_in_time_first_hep = icu.intime)\
  and\
  TIMESTAMP_DIFF(adm.dischtime, first_hep.hep_start, HOUR) > 24 -- removed patients who were discharged within 24 hours since first heparin dose.\
 \
)\
\
#-------------------------------------------------------------------------------------------------------------------------------------------\
\
#5 - This joins #4 with platelet count records.\
# This joins each hep admin with platelet count records for that hadm_id\
# multiple rows per one hadm_id\
\
\
, JOIN_HEAPRIN_PLATELET_LEFT AS  --ALL hadm_id s had at least one platelet count record. In case, if a patient wouldn't have any platelet count record, that patient was assumed to be HIT Negative.\
(\
\
SELECT HEP_ADMIN_COHORT.hadm_id,\
PLATLETS.charttime as p_charttime, PLATLETS.platelet\
\
FROM first_hep_with_hep_type_and_treatment_type_dermographics HEP_ADMIN_COHORT\
\
left join `physionet-data.mimic_core.admissions` adm -- 'hadm_id' was null most of the time in `mimic_derived.complete_blood_count` but not ;'subject_id'. Therefore, 'subject_id' ahd to be used when do join of `mimic_derived.complete_blood_count`. To get the 'subject_id', we had to use 'admissions' table.\
\
on HEP_ADMIN_COHORT.hadm_id = adm.hadm_id\
\
LEFT JOIN `physionet-data.mimic_derived.complete_blood_count` as PLATLETS   \
ON adm.subject_id = PLATLETS.subject_id \
\
where platelet is not null --some records have null platelet count readings. remove them\
and PLATLETS.charttime BETWEEN adm.admittime AND adm.dischtime -- to remove other hadm_ids of particular subject_ids\
)\
\
#------------------------------------------------------------------------------------------------------------------------------------------------\
\
# check for vital signs - labs (hosp), bg (hosp), bg_art (hosp), vitalsigns (icu), GCS (icu)\
\
# 28682905 - had multiple icu stays, and first hep was given later - not in first icu stay\
# 20084622 - first hep was given before first icustay\
# 21362779 - had multiple icu stays, first hep was given during first icustay\
\
, item_id_to_name_without_specimen AS  -- convert from itemid into a meaningful column full blood count related parameters in labevents\
(\
  SELECT\
    MAX(subject_id) as subject_id\
  , MAX(hadm_id) as hadm_id\
  , specimen_id\
  , MAX(charttime) as charttime\
  -- convert from itemid into a meaningful column\
  , MAX(CASE WHEN itemid = 51221 THEN valuenum ELSE NULL END) AS hematocrit\
  , MAX(CASE WHEN itemid = 51221 THEN ref_range_lower ELSE NULL END) AS hematocrit_ref_range_lower\
  , MAX(CASE WHEN itemid = 51221 THEN ref_range_upper ELSE NULL END) AS hematocrit_ref_range_upper\
\
  , MAX(CASE WHEN itemid = 51222 THEN valuenum ELSE NULL END) AS hemoglobin\
  , MAX(CASE WHEN itemid = 51222 THEN ref_range_lower ELSE NULL END) AS hemoglobin_ref_range_lower\
  , MAX(CASE WHEN itemid = 51222 THEN ref_range_upper ELSE NULL END) AS hemoglobin_ref_range_upper\
\
  , MAX(CASE WHEN itemid = 51265 THEN valuenum ELSE NULL END) AS platelets\
  , MAX(CASE WHEN itemid = 51265 THEN ref_range_lower ELSE NULL END) AS platelets_ref_range_lower\
  , MAX(CASE WHEN itemid = 51265 THEN ref_range_upper ELSE NULL END) AS platelets_ref_range_upper\
\
  , MAX(CASE WHEN itemid = 51301 THEN valuenum ELSE NULL END) AS wbc\
  , MAX(CASE WHEN itemid = 51301 THEN ref_range_lower ELSE NULL END) AS wbc_ref_range_lower\
  , MAX(CASE WHEN itemid = 51301 THEN ref_range_upper ELSE NULL END) AS wbc_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50862  THEN valuenum ELSE NULL END) AS  albumin\
 , MAX(CASE WHEN itemid = 50862 THEN ref_range_lower ELSE NULL END) AS  albumin_ref_range_lower\
 , MAX(CASE WHEN itemid = 50862 THEN ref_range_upper ELSE NULL END) AS  albumin_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50930  THEN valuenum ELSE NULL END) AS  globulin\
 , MAX(CASE WHEN itemid = 50930 THEN ref_range_lower ELSE NULL END) AS  globulin_ref_range_lower\
 , MAX(CASE WHEN itemid = 50930 THEN ref_range_upper ELSE NULL END) AS  globulin_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50976  THEN valuenum ELSE NULL END) AS  total_protein\
 , MAX(CASE WHEN itemid = 50976 THEN ref_range_lower ELSE NULL END) AS  total_protein_ref_range_lower\
 , MAX(CASE WHEN itemid = 50976 THEN ref_range_upper ELSE NULL END) AS  total_protein_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50868  THEN valuenum ELSE NULL END) AS  aniongap\
 , MAX(CASE WHEN itemid = 50868 THEN ref_range_lower ELSE NULL END) AS  aniongap_ref_range_lower\
 , MAX(CASE WHEN itemid = 50868 THEN ref_range_upper ELSE NULL END) AS  aniongap_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50882  THEN valuenum ELSE NULL END) AS  bicarbonate\
 , MAX(CASE WHEN itemid = 50882 THEN ref_range_lower ELSE NULL END) AS  bicarbonate_ref_range_lower\
 , MAX(CASE WHEN itemid = 50882 THEN ref_range_upper ELSE NULL END) AS  bicarbonate_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51006  THEN valuenum ELSE NULL END) AS  bun\
 , MAX(CASE WHEN itemid = 51006 THEN ref_range_lower ELSE NULL END) AS  bun_ref_range_lower\
 , MAX(CASE WHEN itemid = 51006 THEN ref_range_upper ELSE NULL END) AS  bun_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50893  THEN valuenum ELSE NULL END) AS  calcium\
 , MAX(CASE WHEN itemid = 50893 THEN ref_range_lower ELSE NULL END) AS  calcium_ref_range_lower\
 , MAX(CASE WHEN itemid = 50893 THEN ref_range_upper ELSE NULL END) AS  calcium_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50902  THEN valuenum ELSE NULL END) AS  chloride\
 , MAX(CASE WHEN itemid = 50902 THEN ref_range_lower ELSE NULL END) AS  chloride_ref_range_lower\
 , MAX(CASE WHEN itemid = 50902 THEN ref_range_upper ELSE NULL END) AS  chloride_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50912  THEN valuenum ELSE NULL END) AS  creatinine\
 , MAX(CASE WHEN itemid = 50912 THEN ref_range_lower ELSE NULL END) AS  creatinine_ref_range_lower\
 , MAX(CASE WHEN itemid = 50912 THEN ref_range_upper ELSE NULL END) AS  creatinine_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50931  THEN valuenum ELSE NULL END) AS  glucose\
 , MAX(CASE WHEN itemid = 50931 THEN ref_range_lower ELSE NULL END) AS  glucose_ref_range_lower\
 , MAX(CASE WHEN itemid = 50931 THEN ref_range_upper ELSE NULL END) AS  glucose_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50983  THEN valuenum ELSE NULL END) AS  sodium\
 , MAX(CASE WHEN itemid = 50983 THEN ref_range_lower ELSE NULL END) AS  sodium_ref_range_lower\
 , MAX(CASE WHEN itemid = 50983 THEN ref_range_upper ELSE NULL END) AS  sodium_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50971  THEN valuenum ELSE NULL END) AS  potassium\
 , MAX(CASE WHEN itemid = 50971 THEN ref_range_lower ELSE NULL END) AS  potassium_ref_range_lower\
 , MAX(CASE WHEN itemid = 50971 THEN ref_range_upper ELSE NULL END) AS  potassium_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 52069  THEN valuenum ELSE NULL END) AS  abs_basophils\
 , MAX(CASE WHEN itemid = 52069 THEN ref_range_lower ELSE NULL END) AS  abs_basophils_ref_range_lower\
 , MAX(CASE WHEN itemid = 52069 THEN ref_range_upper ELSE NULL END) AS  abs_basophils_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 52073  THEN valuenum ELSE NULL END) AS  abs_eosinophils\
 , MAX(CASE WHEN itemid = 52073 THEN ref_range_lower ELSE NULL END) AS  abs_eosinophils_ref_range_lower\
 , MAX(CASE WHEN itemid = 52073 THEN ref_range_upper ELSE NULL END) AS  abs_eosinophils_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51133  THEN valuenum ELSE NULL END) AS  abs_lymphocytes\
 , MAX(CASE WHEN itemid = 51133 THEN ref_range_lower ELSE NULL END) AS  abs_lymphocytes_ref_range_lower\
 , MAX(CASE WHEN itemid = 51133 THEN ref_range_upper ELSE NULL END) AS  abs_lymphocytes_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 52074  THEN valuenum ELSE NULL END) AS  abs_monocytes\
 , MAX(CASE WHEN itemid = 52074 THEN ref_range_lower ELSE NULL END) AS  abs_monocytes_ref_range_lower\
 , MAX(CASE WHEN itemid = 52074 THEN ref_range_upper ELSE NULL END) AS  abs_monocytes_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 52075  THEN valuenum ELSE NULL END) AS  abs_neutrophils\
 , MAX(CASE WHEN itemid = 52075 THEN ref_range_lower ELSE NULL END) AS  abs_neutrophils_ref_range_lower\
 , MAX(CASE WHEN itemid = 52075 THEN ref_range_upper ELSE NULL END) AS  abs_neutrophils_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51143  THEN valuenum ELSE NULL END) AS  atyps\
 , MAX(CASE WHEN itemid = 51143 THEN ref_range_lower ELSE NULL END) AS  atyps_ref_range_lower\
 , MAX(CASE WHEN itemid = 51143 THEN ref_range_upper ELSE NULL END) AS  atyps_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51144  THEN valuenum ELSE NULL END) AS  bands\
 , MAX(CASE WHEN itemid = 51144 THEN ref_range_lower ELSE NULL END) AS  bands_ref_range_lower\
 , MAX(CASE WHEN itemid = 51144 THEN ref_range_upper ELSE NULL END) AS  bands_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 52135  THEN valuenum ELSE NULL END) AS  imm_granulocytes\
 , MAX(CASE WHEN itemid = 52135 THEN ref_range_lower ELSE NULL END) AS  imm_granulocytes_ref_range_lower\
 , MAX(CASE WHEN itemid = 52135 THEN ref_range_upper ELSE NULL END) AS  imm_granulocytes_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51251  THEN valuenum ELSE NULL END) AS  metas\
 , MAX(CASE WHEN itemid = 51251 THEN ref_range_lower ELSE NULL END) AS  metas_ref_range_lower\
 , MAX(CASE WHEN itemid = 51251 THEN ref_range_upper ELSE NULL END) AS  metas_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51257  THEN valuenum ELSE NULL END) AS  nrbc\
 , MAX(CASE WHEN itemid = 51257 THEN ref_range_lower ELSE NULL END) AS  nrbc_ref_range_lower\
 , MAX(CASE WHEN itemid = 51257 THEN ref_range_upper ELSE NULL END) AS  nrbc_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51196  THEN valuenum ELSE NULL END) AS  d_dimer\
 , MAX(CASE WHEN itemid = 51196 THEN ref_range_lower ELSE NULL END) AS  d_dimer_ref_range_lower\
 , MAX(CASE WHEN itemid = 51196 THEN ref_range_upper ELSE NULL END) AS  d_dimer_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51214  THEN valuenum ELSE NULL END) AS  fibrinogen\
 , MAX(CASE WHEN itemid = 51214 THEN ref_range_lower ELSE NULL END) AS  fibrinogen_ref_range_lower\
 , MAX(CASE WHEN itemid = 51214 THEN ref_range_upper ELSE NULL END) AS  fibrinogen_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51297  THEN valuenum ELSE NULL END) AS  thrombin\
 , MAX(CASE WHEN itemid = 51297 THEN ref_range_lower ELSE NULL END) AS  thrombin_ref_range_lower\
 , MAX(CASE WHEN itemid = 51297 THEN ref_range_upper ELSE NULL END) AS  thrombin_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51237  THEN valuenum ELSE NULL END) AS  inr\
 , MAX(CASE WHEN itemid = 51237 THEN ref_range_lower ELSE NULL END) AS  inr_ref_range_lower\
 , MAX(CASE WHEN itemid = 51237 THEN ref_range_upper ELSE NULL END) AS  inr_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51274  THEN valuenum ELSE NULL END) AS  pt\
 , MAX(CASE WHEN itemid = 51274 THEN ref_range_lower ELSE NULL END) AS  pt_ref_range_lower\
 , MAX(CASE WHEN itemid = 51274 THEN ref_range_upper ELSE NULL END) AS  pt_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 51275  THEN valuenum ELSE NULL END) AS  ptt\
 , MAX(CASE WHEN itemid = 51275 THEN ref_range_lower ELSE NULL END) AS  ptt_ref_range_lower\
 , MAX(CASE WHEN itemid = 51275 THEN ref_range_upper ELSE NULL END) AS  ptt_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50861  THEN valuenum ELSE NULL END) AS  alt\
 , MAX(CASE WHEN itemid = 50861 THEN ref_range_lower ELSE NULL END) AS  alt_ref_range_lower\
 , MAX(CASE WHEN itemid = 50861 THEN ref_range_upper ELSE NULL END) AS  alt_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50863  THEN valuenum ELSE NULL END) AS  alp\
 , MAX(CASE WHEN itemid = 50863 THEN ref_range_lower ELSE NULL END) AS  alp_ref_range_lower\
 , MAX(CASE WHEN itemid = 50863 THEN ref_range_upper ELSE NULL END) AS  alp_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50878  THEN valuenum ELSE NULL END) AS  ast\
 , MAX(CASE WHEN itemid = 50878 THEN ref_range_lower ELSE NULL END) AS  ast_ref_range_lower\
 , MAX(CASE WHEN itemid = 50878 THEN ref_range_upper ELSE NULL END) AS  ast_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50867  THEN valuenum ELSE NULL END) AS  amylase\
 , MAX(CASE WHEN itemid = 50867 THEN ref_range_lower ELSE NULL END) AS  amylase_ref_range_lower\
 , MAX(CASE WHEN itemid = 50867 THEN ref_range_upper ELSE NULL END) AS  amylase_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50885  THEN valuenum ELSE NULL END) AS  bilirubin_total\
 , MAX(CASE WHEN itemid = 50885 THEN ref_range_lower ELSE NULL END) AS  bilirubin_total_ref_range_lower\
 , MAX(CASE WHEN itemid = 50885 THEN ref_range_upper ELSE NULL END) AS  bilirubin_total_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50883  THEN valuenum ELSE NULL END) AS  bilirubin_direct\
 , MAX(CASE WHEN itemid = 50883 THEN ref_range_lower ELSE NULL END) AS  bilirubin_direct_ref_range_lower\
 , MAX(CASE WHEN itemid = 50883 THEN ref_range_upper ELSE NULL END) AS  bilirubin_direct_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50884  THEN valuenum ELSE NULL END) AS  bilirubin_indirect\
 , MAX(CASE WHEN itemid = 50884 THEN ref_range_lower ELSE NULL END) AS  bilirubin_indirect_ref_range_lower\
 , MAX(CASE WHEN itemid = 50884 THEN ref_range_upper ELSE NULL END) AS  bilirubin_indirect_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50910  THEN valuenum ELSE NULL END) AS  ck_cpk\
 , MAX(CASE WHEN itemid = 50910 THEN ref_range_lower ELSE NULL END) AS  ck_cpk_ref_range_lower\
 , MAX(CASE WHEN itemid = 50910 THEN ref_range_upper ELSE NULL END) AS  ck_cpk_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50911  THEN valuenum ELSE NULL END) AS  ck_mb\
 , MAX(CASE WHEN itemid = 50911 THEN ref_range_lower ELSE NULL END) AS  ck_mb_ref_range_lower\
 , MAX(CASE WHEN itemid = 50911 THEN ref_range_upper ELSE NULL END) AS  ck_mb_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50927  THEN valuenum ELSE NULL END) AS  ggt\
 , MAX(CASE WHEN itemid = 50927 THEN ref_range_lower ELSE NULL END) AS  ggt_ref_range_lower\
 , MAX(CASE WHEN itemid = 50927 THEN ref_range_upper ELSE NULL END) AS  ggt_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50954  THEN valuenum ELSE NULL END) AS  ld_ldh\
 , MAX(CASE WHEN itemid = 50954 THEN ref_range_lower ELSE NULL END) AS  ld_ldh_ref_range_lower\
 , MAX(CASE WHEN itemid = 50954 THEN ref_range_upper ELSE NULL END) AS  ld_ldh_ref_range_upper\
\
-- blood gases (vital signs in mimic derived_first24h_bg)\
\
 --, MAX(CASE WHEN itemid = 52033 THEN value ELSE NULL END) AS specimen  -- when this value = 'ART', can use for bg_art\
\
 , MAX(CASE WHEN itemid = 50813 THEN valuenum ELSE NULL END) AS lactate_bg, MAX(CASE WHEN itemid = 50813 THEN ref_range_lower ELSE NULL END) AS lactate_bg_ref_range_lower, MAX(CASE WHEN itemid = 50813 THEN ref_range_upper ELSE NULL END) AS lactate_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50820 THEN valuenum ELSE NULL END) AS ph_bg, MAX(CASE WHEN itemid = 50820 THEN ref_range_lower ELSE NULL END) AS ph_bg_ref_range_lower, MAX(CASE WHEN itemid = 50820 THEN ref_range_upper ELSE NULL END) AS ph_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50817 THEN valuenum ELSE NULL END) AS so2_bg, MAX(CASE WHEN itemid = 50817 THEN ref_range_lower ELSE NULL END) AS so2_bg_ref_range_lower, MAX(CASE WHEN itemid = 50817 THEN ref_range_upper ELSE NULL END) AS so2_bg_ref_range_upper \
\
 , MAX(CASE WHEN itemid = 50821 THEN valuenum ELSE NULL END) AS po2_bg, MAX(CASE WHEN itemid = 50821 THEN ref_range_lower ELSE NULL END) AS po2_bg_ref_range_lower, MAX(CASE WHEN itemid = 50821 THEN ref_range_upper ELSE NULL END) AS po2_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50818 THEN valuenum ELSE NULL END) AS pco2_bg, MAX(CASE WHEN itemid = 50818 THEN ref_range_lower ELSE NULL END) AS pco2_bg_ref_range_lower, MAX(CASE WHEN itemid = 50818 THEN ref_range_upper ELSE NULL END) AS pco2_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50801 THEN valuenum ELSE NULL END) AS aado2_bg, MAX(CASE WHEN itemid = 50801 THEN ref_range_lower ELSE NULL END) AS aado2_bg_ref_range_lower, MAX(CASE WHEN itemid = 50801 THEN ref_range_upper ELSE NULL END) AS aado2_bg_ref_range_upper\
\
\
--\'e7ouldn't fing ref ranges for aado2_calc , pao2fio2ratio. But these are formed using PO2, PCO2 and fio2. As PO1 and PCO1 already considered above, fio2 is added here, so that ok to omit aado2_calc , pao2fio2ratio.\
\
  , MAX(CASE WHEN itemid = 50816 THEN\
      CASE\
        WHEN valuenum > 20 AND valuenum <= 100 THEN valuenum\
        WHEN valuenum > 0.2 AND valuenum <= 1.0 THEN valuenum*100.0\
      ELSE NULL END\
    ELSE NULL END) AS fio2_bg ,\
    MAX(CASE WHEN itemid = 50816 THEN ref_range_lower ELSE NULL END) AS fio2_bg_ref_range_lower, MAX(CASE WHEN itemid = 50816 THEN ref_range_upper ELSE NULL END) AS fio2_bg_ref_range_upper\
\
\
 , MAX(CASE WHEN itemid = 50802 THEN valuenum ELSE NULL END) AS baseexcess_bg, MAX(CASE WHEN itemid = 50802 THEN ref_range_lower ELSE NULL END) AS baseexcess_bg_ref_range_lower, MAX(CASE WHEN itemid = 50802 THEN ref_range_upper ELSE NULL END) AS baseexcess_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50803 THEN valuenum ELSE NULL END) AS bicarbonate_bg, MAX(CASE WHEN itemid = 50803 THEN ref_range_lower ELSE NULL END) AS bicarbonate_bg_ref_range_lower, MAX(CASE WHEN itemid = 50803 THEN ref_range_upper ELSE NULL END) AS bicarbonate_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50804 THEN valuenum ELSE NULL END) AS totalco2_bg, MAX(CASE WHEN itemid = 50804 THEN ref_range_lower ELSE NULL END) AS totalco2_bg_ref_range_lower, MAX(CASE WHEN itemid = 50804 THEN ref_range_upper ELSE NULL END) AS totalco2_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50810 THEN valuenum ELSE NULL END) AS hematocrit_bg, MAX(CASE WHEN itemid = 50810 THEN ref_range_lower ELSE NULL END) AS hematocrit_bg_ref_range_lower, MAX(CASE WHEN itemid = 50810 THEN ref_range_upper ELSE NULL END) AS hematocrit_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50811 THEN valuenum ELSE NULL END) AS hemoglobin_bg, MAX(CASE WHEN itemid = 50811 THEN ref_range_lower ELSE NULL END) AS hemoglobin_bg_ref_range_lower, MAX(CASE WHEN itemid = 50811 THEN ref_range_upper ELSE NULL END) AS hemoglobin_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50805 THEN valuenum ELSE NULL END) AS carboxyhemoglobin_bg, MAX(CASE WHEN itemid = 50805 THEN ref_range_lower ELSE NULL END) AS carboxyhemoglobin_bg_ref_range_lower, MAX(CASE WHEN itemid = 50805 THEN ref_range_upper ELSE NULL END) AS carboxyhemoglobin_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50814 THEN valuenum ELSE NULL END) AS methemoglobin_bg, MAX(CASE WHEN itemid = 50814 THEN ref_range_lower ELSE NULL END) AS methemoglobin_bg_ref_range_lower, MAX(CASE WHEN itemid = 50814 THEN ref_range_upper ELSE NULL END) AS methemoglobin_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50825 THEN valuenum ELSE NULL END) AS temperature_bg, MAX(CASE WHEN itemid = 50825 THEN ref_range_lower ELSE NULL END) AS temperature_bg_ref_range_lower, MAX(CASE WHEN itemid = 50825 THEN ref_range_upper ELSE NULL END) AS temperature_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50806 THEN valuenum ELSE NULL END) AS chloride_bg, MAX(CASE WHEN itemid = 50806 THEN ref_range_lower ELSE NULL END) AS chloride_bg_ref_range_lower, MAX(CASE WHEN itemid = 50806 THEN ref_range_upper ELSE NULL END) AS chloride_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50808 THEN valuenum ELSE NULL END) AS calcium_bg, MAX(CASE WHEN itemid = 50808 THEN ref_range_lower ELSE NULL END) AS calcium_bg_ref_range_lower, MAX(CASE WHEN itemid = 50808 THEN ref_range_upper ELSE NULL END) AS calcium_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50809 THEN valuenum ELSE NULL END) AS glucose_bg, MAX(CASE WHEN itemid = 50809 THEN ref_range_lower ELSE NULL END) AS glucose_bg_ref_range_lower, MAX(CASE WHEN itemid = 50809 THEN ref_range_upper ELSE NULL END) AS glucose_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50822 THEN valuenum ELSE NULL END) AS potassium_bg, MAX(CASE WHEN itemid = 50822 THEN ref_range_lower ELSE NULL END) AS potassium_bg_ref_range_lower, MAX(CASE WHEN itemid = 50822 THEN ref_range_upper ELSE NULL END) AS potassium_bg_ref_range_upper\
\
 , MAX(CASE WHEN itemid = 50824 THEN valuenum ELSE NULL END) AS sodium_bg, MAX(CASE WHEN itemid = 50824 THEN ref_range_lower ELSE NULL END) AS sodium_bg_ref_range_lower, MAX(CASE WHEN itemid = 50824 THEN ref_range_upper ELSE NULL END) AS sodium_bg_ref_range_upper\
\
from `physionet-data.mimic_hosp.labevents`\
\
WHERE itemid IN\
(\
51221,	--hematocrit\
51222,	--hemoglobin\
51265,	--platelet\
51301,	--wbc\
\
50862,	--albumin\
50930,	--globulin\
50976,	--total_protein\
50868,	--aniongap\
50882,	--bicarbonate\
51006,	--bun\
50893,	--calcium\
50902,	--chloride\
50912,	--creatinine\
50931,	--glucose\
50983,	--sodium\
50971,	--potassium\
\
52069,	--basophils_abs\
52073,	--eosinophils_abs\
51133,	--lymphocytes_abs\
52074,	--monocytes_abs\
52075,	--neutrophils_abs\
51143,	--atypical_lymphocytes\
51144,	--bands\
52135,	--immature_granulocytes\
51251,	--metamyelocytes\
51257,	--nrbc\
\
51196,	--d_dimer\
51214,	--fibrinogen\
51297,	--thrombin\
51237,	--inr\
51274,	--pt\
51275,	--ptt\
\
50861,	--alt\
50863,	--alp\
50878,	--ast\
50867,	--amylase\
50885,	--bilirubin_total\
50883,	--bilirubin_direct\
50884,	--bilirubin_indirect\
50910,	--ck_cpk\
50911,	--ck_mb\
50927,	--ggt\
50954, 	--ld_ldh\
\
--blood gasses\
\
 --52033,  -- specimen (='ART' for bg_ART)\
 50813,  -- lactate_bg\
 50820,  -- ph_bg\
 50817,  -- so2_bg\
 50821,  -- po2_bg\
 50818,  -- pco2_bg\
 50801,  -- aado2_bg\
 50802,  -- baseexcess_bg\
 50803,  -- bicarbonate_bg\
 50804,  -- totalco2_bg\
 50810,  -- hematocrit_bg\
 50811,  -- hemoglobin_bg\
 50805,  -- carboxyhemoglobin_bg\
 50814,  -- methemoglobin_bg\
 50825,  -- temperature_bg\
 50806,  -- chloride_bg\
 50808,  -- calcium_bg\
 50809,  -- glucose_bg\
 50822,  -- potassium_bg\
 50824,  -- sodium_bg\
 50816   -- fio2 (added above 'baseexcess_bg')\
\
\
)\
AND valuenum IS NOT NULL -- reason to issue\
\
--AND valuenum > 0 -- lab values can be negative (Ex - baseexcess. therefore, remove this line)\
\
group by specimen_id\
)\
--1.2\
\
, specimen_to_join_item_id_to_name_without_specimen AS  -- convert from itemid into a meaningful column full full blood count related parameters in labevents\
(\
  SELECT\
    MAX(subject_id) as subject_id\
  , MAX(hadm_id) as hadm_id\
  , specimen_id\
  , MAX(charttime) as charttime\
  -- convert from itemid into a meaningful column\
\
-- blood gases (vital signs in mimic derived_first24h_bg)\
\
, MAX(CASE WHEN itemid = 52033 THEN value ELSE NULL END) AS specimen  -- when this value = 'ART', can use for bg_art\
\
from `physionet-data.mimic_hosp.labevents`\
\
WHERE itemid IN\
(\
\
--blood gasses\
\
 52033  -- specimen (='ART' for bg_ART)\
\
)\
--AND valuenum IS NOT NULL -- reason to issue\
-- lab values cannot be 0 and cannot be negative\
--AND valuenum > 0\
\
group by specimen_id\
)\
\
, item_id_to_name_after_joining_specimen AS \
(\
  select without_specimen.*, specimen_only.specimen\
  from item_id_to_name_without_specimen without_specimen\
  left join specimen_to_join_item_id_to_name_without_specimen specimen_only\
  on without_specimen.specimen_id = specimen_only.specimen_id\
)\
\
-- 2. Get first 24h vital signs for each icu stay_id\
\
, first24h_vital_signs AS\
(\
  SELECT\
  min(ie.subject_id) as subject_id\
, min(ie.hadm_id) as hadm_id\
, min(ie.stay_id) as stay_id\
\
 --labs \
\
, min(hematocrit) as hematocrit_lab_min\
, max(hematocrit) as hematocrit_lab_max\
, min( hematocrit_ref_range_lower) as  hematocrit_lab_ref_range_lower\
, max( hematocrit_ref_range_upper) as  hematocrit_lab_ref_range_upper\
\
, min(hemoglobin) as hemoglobin_lab_min\
, max(hemoglobin) as hemoglobin_lab_max\
, min( hemoglobin_ref_range_lower) as  hemoglobin_lab_ref_range_lower\
, max( hemoglobin_ref_range_upper) as  hemoglobin_lab_ref_range_upper\
\
, min(platelets) as platelets_lab_min\
, max(platelets) as platelets_lab_max\
, min( platelets_ref_range_lower) as  platelets_lab_ref_range_lower\
, max( platelets_ref_range_upper) as  platelets_lab_ref_range_upper\
\
, min(wbc) as wbc_lab_min\
, max(wbc) as wbc_lab_max\
, min( wbc_ref_range_lower) as  wbc_lab_ref_range_lower  \
, max( wbc_ref_range_upper) as  wbc_lab_ref_range_upper\
\
, min(albumin) as albumin_lab_min\
, max(albumin) as albumin_lab_max\
, min( albumin_ref_range_lower) as  albumin_lab_ref_range_lower\
, max( albumin_ref_range_upper) as  albumin_lab_ref_range_upper\
\
, min(globulin) as globulin_lab_min\
, max(globulin) as globulin_lab_max\
, min( globulin_ref_range_lower) as  globulin_lab_ref_range_lower\
, max( globulin_ref_range_upper) as  globulin_lab_ref_range_upper\
\
, min(total_protein) as total_protein_lab_min\
, max(total_protein) as total_protein_lab_max\
, min( total_protein_ref_range_lower) as  total_protein_lab_ref_range_lower\
, max( total_protein_ref_range_upper) as  total_protein_lab_ref_range_upper\
\
, min(aniongap) as aniongap_lab_min\
, max(aniongap) as aniongap_lab_max\
, min( aniongap_ref_range_lower) as  aniongap_lab_ref_range_lower\
, max( aniongap_ref_range_upper) as  aniongap_lab_ref_range_upper\
\
, min(bicarbonate) as bicarbonate_lab_min\
, max(bicarbonate) as bicarbonate_lab_max\
, min( bicarbonate_ref_range_lower) as  bicarbonate_lab_ref_range_lower\
, max( bicarbonate_ref_range_upper) as  bicarbonate_lab_ref_range_upper\
\
, min(bun) as bun_lab_min\
, max(bun) as bun_lab_max\
, min( bun_ref_range_lower) as  bun_lab_ref_range_lower\
, max( bun_ref_range_upper) as  bun_lab_ref_range_upper\
\
, min(calcium) as calcium_lab_min\
, max(calcium) as calcium_lab_max\
, min( calcium_ref_range_lower) as  calcium_lab_ref_range_lower\
, max( calcium_ref_range_upper) as  calcium_lab_ref_range_upper\
\
, min(chloride) as chloride_lab_min\
, max(chloride) as chloride_lab_max\
, min( chloride_ref_range_lower) as  chloride_lab_ref_range_lower\
, max( chloride_ref_range_upper) as  chloride_lab_ref_range_upper\
\
, min(creatinine) as creatinine_lab_min\
, max(creatinine) as creatinine_lab_max\
, min( creatinine_ref_range_lower) as  creatinine_lab_ref_range_lower\
, max( creatinine_ref_range_upper) as  creatinine_lab_ref_range_upper\
\
, min(glucose) as glucose_lab_min\
, max(glucose) as glucose_lab_max\
, min( glucose_ref_range_lower) as  glucose_lab_ref_range_lower\
, max( glucose_ref_range_upper) as  glucose_lab_ref_range_upper\
\
, min(sodium) as sodium_lab_min\
, max(sodium) as sodium_lab_max\
, min( sodium_ref_range_lower) as  sodium_lab_ref_range_lower\
, max( sodium_ref_range_upper) as  sodium_lab_ref_range_upper\
\
, min(potassium) as potassium_lab_min\
, max(potassium) as potassium_lab_max\
, min( potassium_ref_range_lower) as  potassium_lab_ref_range_lower\
, max( potassium_ref_range_upper) as  potassium_lab_ref_range_upper\
\
, min(abs_basophils) as abs_basophils_lab_min\
, max(abs_basophils) as abs_basophils_lab_max\
, min( abs_basophils_ref_range_lower) as  abs_basophils_lab_ref_range_lower\
, max( abs_basophils_ref_range_upper) as  abs_basophils_lab_ref_range_upper\
\
, min(abs_eosinophils) as abs_eosinophils_lab_min\
, max(abs_eosinophils) as abs_eosinophils_lab_max\
, min( abs_eosinophils_ref_range_lower) as  abs_eosinophils_lab_ref_range_lower\
, max( abs_eosinophils_ref_range_upper) as  abs_eosinophils_lab_ref_range_upper\
\
, min(abs_lymphocytes) as abs_lymphocytes_lab_min\
, max(abs_lymphocytes) as abs_lymphocytes_lab_max\
, min( abs_lymphocytes_ref_range_lower) as  abs_lymphocytes_lab_ref_range_lower\
, max( abs_lymphocytes_ref_range_upper) as  abs_lymphocytes_lab_ref_range_upper\
\
, min(abs_monocytes) as abs_monocytes_lab_min\
, max(abs_monocytes) as abs_monocytes_lab_max\
, min( abs_monocytes_ref_range_lower) as  abs_monocytes_lab_ref_range_lower\
, max( abs_monocytes_ref_range_upper) as  abs_monocytes_lab_ref_range_upper\
\
, min(abs_neutrophils) as abs_neutrophils_lab_min\
, max(abs_neutrophils) as abs_neutrophils_lab_max\
, min( abs_neutrophils_ref_range_lower) as  abs_neutrophils_lab_ref_range_lower\
, max( abs_neutrophils_ref_range_upper) as  abs_neutrophils_lab_ref_range_upper\
\
, min(atyps) as atyps_lab_min\
, max(atyps) as atyps_lab_max\
, min( atyps_ref_range_lower) as  atyps_lab_ref_range_lower\
, max( atyps_ref_range_upper) as  atyps_lab_ref_range_upper\
\
, min(bands) as bands_lab_min\
, max(bands) as bands_lab_max\
, min( bands_ref_range_lower) as  bands_lab_ref_range_lower\
, max( bands_ref_range_upper) as  bands_lab_ref_range_upper\
\
, min(imm_granulocytes) as imm_granulocytes_lab_min\
, max(imm_granulocytes) as imm_granulocytes_lab_max\
, min( imm_granulocytes_ref_range_lower) as  imm_granulocytes_lab_ref_range_lower\
, max( imm_granulocytes_ref_range_upper) as  imm_granulocytes_lab_ref_range_upper\
\
, min(metas) as metas_lab_min\
, max(metas) as metas_lab_max\
, min( metas_ref_range_lower) as  metas_lab_ref_range_lower\
, max( metas_ref_range_upper) as  metas_lab_ref_range_upper\
\
, min(nrbc) as nrbc_lab_min\
, max(nrbc) as nrbc_lab_max\
, min( nrbc_ref_range_lower) as  nrbc_lab_ref_range_lower\
, max( nrbc_ref_range_upper) as  nrbc_lab_ref_range_upper\
\
, min(d_dimer) as d_dimer_lab_min\
, max(d_dimer) as d_dimer_lab_max\
, min( d_dimer_ref_range_lower) as  d_dimer_lab_ref_range_lower\
, max( d_dimer_ref_range_upper) as  d_dimer_lab_ref_range_upper\
\
, min(fibrinogen) as fibrinogen_lab_min\
, max(fibrinogen) as fibrinogen_lab_max\
, min( fibrinogen_ref_range_lower) as  fibrinogen_lab_ref_range_lower\
, max( fibrinogen_ref_range_upper) as  fibrinogen_lab_ref_range_upper\
\
, min(thrombin) as thrombin_lab_min\
, max(thrombin) as thrombin_lab_max\
, min( thrombin_ref_range_lower) as  thrombin_lab_ref_range_lower\
, max( thrombin_ref_range_upper) as  thrombin_lab_ref_range_upper\
\
, min(inr) as inr_lab_min\
, max(inr) as inr_lab_max\
, min( inr_ref_range_lower) as  inr_lab_ref_range_lower\
, max( inr_ref_range_upper) as  inr_lab_ref_range_upper\
\
, min(pt) as pt_lab_min\
, max(pt) as pt_lab_max\
, min( pt_ref_range_lower) as  pt_lab_ref_range_lower\
, max( pt_ref_range_upper) as  pt_lab_ref_range_upper\
\
, min(ptt) as ptt_lab_min\
, max(ptt) as ptt_lab_max\
, min( ptt_ref_range_lower) as  ptt_lab_ref_range_lower\
, max( ptt_ref_range_upper) as  ptt_lab_ref_range_upper\
\
, min(alt) as alt_lab_min\
, max(alt) as alt_lab_max\
, min( alt_ref_range_lower) as  alt_lab_ref_range_lower\
, max( alt_ref_range_upper) as  alt_lab_ref_range_upper\
\
, min(alp) as alp_lab_min\
, max(alp) as alp_lab_max\
, min( alp_ref_range_lower) as  alp_lab_ref_range_lower\
, max( alp_ref_range_upper) as  alp_lab_ref_range_upper\
\
, min(ast) as ast_lab_min\
, max(ast) as ast_lab_max\
, min( ast_ref_range_lower) as  ast_lab_ref_range_lower\
, max( ast_ref_range_upper) as  ast_lab_ref_range_upper\
\
, min(amylase) as amylase_lab_min\
, max(amylase) as amylase_lab_max\
, min( amylase_ref_range_lower) as  amylase_lab_ref_range_lower\
, max( amylase_ref_range_upper) as  amylase_lab_ref_range_upper\
\
, min(bilirubin_total) as bilirubin_total_lab_min\
, max(bilirubin_total) as bilirubin_total_lab_max\
, min( bilirubin_total_ref_range_lower) as  bilirubin_total_lab_ref_range_lower\
, max( bilirubin_total_ref_range_upper) as  bilirubin_total_lab_ref_range_upper\
\
, min(bilirubin_direct) as bilirubin_direct_lab_min\
, max(bilirubin_direct) as bilirubin_direct_lab_max\
, min( bilirubin_direct_ref_range_lower) as  bilirubin_direct_lab_ref_range_lower\
, max( bilirubin_direct_ref_range_upper) as  bilirubin_direct_lab_ref_range_upper\
\
, min(bilirubin_indirect) as bilirubin_indirect_lab_min\
, max(bilirubin_indirect) as bilirubin_indirect_lab_max\
, min( bilirubin_indirect_ref_range_lower) as  bilirubin_indirect_lab_ref_range_lower\
, max( bilirubin_indirect_ref_range_upper) as  bilirubin_indirect_lab_ref_range_upper\
\
, min(ck_cpk) as ck_cpk_lab_min\
, max(ck_cpk) as ck_cpk_lab_max\
, min( ck_cpk_ref_range_lower) as  ck_cpk_lab_ref_range_lower\
, max( ck_cpk_ref_range_upper) as  ck_cpk_lab_ref_range_upper\
\
, min(ck_mb) as ck_mb_lab_min\
, max(ck_mb) as ck_mb_lab_max\
, min( ck_mb_ref_range_lower) as  ck_mb_lab_ref_range_lower\
, max( ck_mb_ref_range_upper) as  ck_mb_lab_ref_range_upper\
\
, min(ggt) as ggt_lab_min\
, max(ggt) as ggt_lab_max\
, min( ggt_ref_range_lower) as  ggt_lab_ref_range_lower\
, max( ggt_ref_range_upper) as  ggt_lab_ref_range_upper\
\
, min(ld_ldh) as ld_ldh_lab_min\
, max(ld_ldh) as ld_ldh_lab_max\
, min( ld_ldh_ref_range_lower) as  ld_ldh_lab_ref_range_lower\
, max( ld_ldh_ref_range_upper) as  ld_ldh_lab_ref_range_upper\
\
-- blood gass\
\
  , min(lactate_bg) as lactate_bg_min , max(lactate_bg) as lactate_bg_max , min( lactate_bg_ref_range_lower) as lactate_bg_ref_range_lower , max( lactate_bg_ref_range_upper) as lactate_bg_ref_range_upper\
\
  , min(ph_bg) as ph_bg_min , max(ph_bg) as ph_bg_max , min( ph_bg_ref_range_lower) as ph_bg_ref_range_lower , max( ph_bg_ref_range_upper) as ph_bg_ref_range_upper\
\
  , min(so2_bg) as so2_bg_min , max(so2_bg) as so2_bg_max , min( so2_bg_ref_range_lower) as so2_bg_ref_range_lower , max( so2_bg_ref_range_upper) as so2_bg_ref_range_upper\
\
  , min(po2_bg) as po2_bg_min , max(po2_bg) as po2_bg_max , min( po2_bg_ref_range_lower) as po2_bg_ref_range_lower , max( po2_bg_ref_range_upper) as po2_bg_ref_range_upper\
  \
  , min(pco2_bg) as pco2_bg_min , max(pco2_bg) as pco2_bg_max , min( pco2_bg_ref_range_lower) as pco2_bg_ref_range_lower , max( pco2_bg_ref_range_upper) as pco2_bg_ref_range_upper\
  \
  , min(aado2_bg) as aado2_bg_min , max(aado2_bg) as aado2_bg_max , min( aado2_bg_ref_range_lower) as aado2_bg_ref_range_lower , max( aado2_bg_ref_range_upper) as aado2_bg_ref_range_upper\
\
\
  , min(fio2_bg) as fio2_bg_min , max(fio2_bg) as fio2_bg_max , min( fio2_bg_ref_range_lower) as fio2_bg_ref_range_lower , max( fio2_bg_ref_range_upper) as fio2_bg_ref_range_upper\
\
\
  , min(baseexcess_bg) as baseexcess_bg_min , max(baseexcess_bg) as baseexcess_bg_max , min( baseexcess_bg_ref_range_lower) as baseexcess_bg_ref_range_lower , max( baseexcess_bg_ref_range_upper) as baseexcess_bg_ref_range_upper\
\
  , min(bicarbonate_bg) as bicarbonate_bg_min , max(bicarbonate_bg) as bicarbonate_bg_max , min( bicarbonate_bg_ref_range_lower) as bicarbonate_bg_ref_range_lower , max( bicarbonate_bg_ref_range_upper) as bicarbonate_bg_ref_range_upper\
\
  , min(totalco2_bg) as totalco2_bg_min , max(totalco2_bg) as totalco2_bg_max , min( totalco2_bg_ref_range_lower) as totalco2_bg_ref_range_lower , max( totalco2_bg_ref_range_upper) as totalco2_bg_ref_range_upper\
\
  , min(hematocrit_bg) as hematocrit_bg_min , max(hematocrit_bg) as hematocrit_bg_max , min( hematocrit_bg_ref_range_lower) as hematocrit_bg_ref_range_lower , max( hematocrit_bg_ref_range_upper) as hematocrit_bg_ref_range_upper\
\
  , min(hemoglobin_bg) as hemoglobin_bg_min , max(hemoglobin_bg) as hemoglobin_bg_max , min( hemoglobin_bg_ref_range_lower) as hemoglobin_bg_ref_range_lower , max( hemoglobin_bg_ref_range_upper) as hemoglobin_bg_ref_range_upper\
\
  , min(carboxyhemoglobin_bg) as carboxyhemoglobin_bg_min , max(carboxyhemoglobin_bg) as carboxyhemoglobin_bg_max , min( carboxyhemoglobin_bg_ref_range_lower) as carboxyhemoglobin_bg_ref_range_lower , max( carboxyhemoglobin_bg_ref_range_upper) as carboxyhemoglobin_bg_ref_range_upper\
\
  , min(methemoglobin_bg) as methemoglobin_bg_min , max(methemoglobin_bg) as methemoglobin_bg_max , min( methemoglobin_bg_ref_range_lower) as methemoglobin_bg_ref_range_lower , max( methemoglobin_bg_ref_range_upper) as methemoglobin_bg_ref_range_upper\
\
  , min(temperature_bg) as temperature_bg_min , max(temperature_bg) as temperature_bg_max , min( temperature_bg_ref_range_lower) as temperature_bg_ref_range_lower , max( temperature_bg_ref_range_upper) as temperature_bg_ref_range_upper\
\
  , min(chloride_bg) as chloride_bg_min , max(chloride_bg) as chloride_bg_max , min( chloride_bg_ref_range_lower) as chloride_bg_ref_range_lower , max( chloride_bg_ref_range_upper) as chloride_bg_ref_range_upper\
\
  , min(calcium_bg) as calcium_bg_min , max(calcium_bg) as calcium_bg_max , min( calcium_bg_ref_range_lower) as calcium_bg_ref_range_lower , max( calcium_bg_ref_range_upper) as calcium_bg_ref_range_upper\
\
  , min(glucose_bg) as glucose_bg_min , max(glucose_bg) as glucose_bg_max , min( glucose_bg_ref_range_lower) as glucose_bg_ref_range_lower , max( glucose_bg_ref_range_upper) as glucose_bg_ref_range_upper\
\
  , min(potassium_bg) as potassium_bg_min , max(potassium_bg) as potassium_bg_max , min( potassium_bg_ref_range_lower) as potassium_bg_ref_range_lower , max( potassium_bg_ref_range_upper) as potassium_bg_ref_range_upper\
\
  , min(sodium_bg) as sodium_bg_min , max(sodium_bg) as sodium_bg_max , min( sodium_bg_ref_range_lower) as sodium_bg_ref_range_lower , max( sodium_bg_ref_range_upper) as sodium_bg_ref_range_upper\
\
    FROM first_hep_with_hep_type_and_treatment_type_dermographics HEP_COHORT # To get the final hep admin cohort. This contain hadm_id, but not subject_id (coz of the way we grouped)\
    LEFT JOIN `physionet-data.mimic_icu.icustays` ie # to get subject_id. To join with below table (item_id_to_name_after_joining_specimen - original table used was: mimic_hosp.labevents), 'subject_id' required, coz hadm_id of most of the rows in below table are blank.\
    ON HEP_COHORT.STAY_ID = IE.STAY_ID\
\
LEFT JOIN item_id_to_name_after_joining_specimen le  # To get lab result details\
        ON ie.subject_id = le.subject_id # some hadm_id in lab test results (`physionet-data.mimic_hosp.labevents`) are blank. That's why subject_id was used to join\
        where le.charttime >= DATETIME_SUB(HEP_COHORT.hep_start, INTERVAL '1' DAY) # 24 hours prior to first hep dose\
        AND le.charttime <= hep_start # lab tests before first hep\
\
    GROUP BY HEP_COHORT.hadm_id\
)\
\
-- 3. bg_art\
\
, first24h_bg_ART AS\
(\
  SELECT\
  min(ie.subject_id) as subject_id\
, min(ie.hadm_id) as hadm_id\
, min(ie.stay_id) as stay_id\
\
-- blood gass\
\
  , min(lactate_bg) as lactate_bg_art_min , max(lactate_bg) as lactate_bg_art_max , min( lactate_bg_ref_range_lower) as lactate_bg_art_ref_range_lower , max( lactate_bg_ref_range_upper) as lactate_bg_art_ref_range_upper\
\
  , min(ph_bg) as ph_bg_art_min , max(ph_bg) as ph_bg_art_max , min( ph_bg_ref_range_lower) as ph_bg_art_ref_range_lower , max( ph_bg_ref_range_upper) as ph_bg_art_ref_range_upper\
\
  , min(so2_bg) as so2_bg_art_min , max(so2_bg) as so2_bg_art_max , min( so2_bg_ref_range_lower) as so2_bg_art_ref_range_lower , max( so2_bg_ref_range_upper) as so2_bg_art_ref_range_upper\
\
  , min(po2_bg) as po2_bg_art_min , max(po2_bg) as po2_bg_art_max , min( po2_bg_ref_range_lower) as po2_bg_art_ref_range_lower , max( po2_bg_ref_range_upper) as po2_bg_art_ref_range_upper\
  \
  , min(pco2_bg) as pco2_bg_art_min , max(pco2_bg) as pco2_bg_art_max , min( pco2_bg_ref_range_lower) as pco2_bg_art_ref_range_lower , max( pco2_bg_ref_range_upper) as pco2_bg_art_ref_range_upper\
  \
  , min(aado2_bg) as aado2_bg_art_min , max(aado2_bg) as aado2_bg_art_max , min( aado2_bg_ref_range_lower) as aado2_bg_art_ref_range_lower , max( aado2_bg_ref_range_upper) as aado2_bg_art_ref_range_upper\
\
\
  , min(fio2_bg) as fio2_bg_art_min , max(fio2_bg) as fio2_bg_art_max , min( fio2_bg_ref_range_lower) as fio2_bg_art_ref_range_lower , max( fio2_bg_ref_range_upper) as fio2_bg_art_ref_range_upper\
\
\
  , min(baseexcess_bg) as baseexcess_bg_art_min , max(baseexcess_bg) as baseexcess_bg_art_max , min( baseexcess_bg_ref_range_lower) as baseexcess_bg_art_ref_range_lower , max( baseexcess_bg_ref_range_upper) as baseexcess_bg_art_ref_range_upper\
\
  , min(bicarbonate_bg) as bicarbonate_bg_art_min , max(bicarbonate_bg) as bicarbonate_bg_art_max , min( bicarbonate_bg_ref_range_lower) as bicarbonate_bg_art_ref_range_lower , max( bicarbonate_bg_ref_range_upper) as bicarbonate_bg_art_ref_range_upper\
\
  , min(totalco2_bg) as totalco2_bg_art_min , max(totalco2_bg) as totalco2_bg_art_max , min( totalco2_bg_ref_range_lower) as totalco2_bg_art_ref_range_lower , max( totalco2_bg_ref_range_upper) as totalco2_bg_art_ref_range_upper\
\
  , min(hematocrit_bg) as hematocrit_bg_art_min , max(hematocrit_bg) as hematocrit_bg_art_max , min( hematocrit_bg_ref_range_lower) as hematocrit_bg_art_ref_range_lower , max( hematocrit_bg_ref_range_upper) as hematocrit_bg_art_ref_range_upper\
\
  , min(hemoglobin_bg) as hemoglobin_bg_art_min , max(hemoglobin_bg) as hemoglobin_bg_art_max , min( hemoglobin_bg_ref_range_lower) as hemoglobin_bg_art_ref_range_lower , max( hemoglobin_bg_ref_range_upper) as hemoglobin_bg_art_ref_range_upper\
\
  , min(carboxyhemoglobin_bg) as carboxyhemoglobin_bg_art_min , max(carboxyhemoglobin_bg) as carboxyhemoglobin_bg_art_max , min( carboxyhemoglobin_bg_ref_range_lower) as carboxyhemoglobin_bg_art_ref_range_lower , max( carboxyhemoglobin_bg_ref_range_upper) as carboxyhemoglobin_bg_art_ref_range_upper\
\
  , min(methemoglobin_bg) as methemoglobin_bg_art_min , max(methemoglobin_bg) as methemoglobin_bg_art_max , min( methemoglobin_bg_ref_range_lower) as methemoglobin_bg_art_ref_range_lower , max( methemoglobin_bg_ref_range_upper) as methemoglobin_bg_art_ref_range_upper\
\
  , min(temperature_bg) as temperature_bg_art_min , max(temperature_bg) as temperature_bg_art_max , min( temperature_bg_ref_range_lower) as temperature_bg_art_ref_range_lower , max( temperature_bg_ref_range_upper) as temperature_bg_art_ref_range_upper\
\
  , min(chloride_bg) as chloride_bg_art_min , max(chloride_bg) as chloride_bg_art_max , min( chloride_bg_ref_range_lower) as chloride_bg_art_ref_range_lower , max( chloride_bg_ref_range_upper) as chloride_bg_art_ref_range_upper\
\
  , min(calcium_bg) as calcium_bg_art_min , max(calcium_bg) as calcium_bg_art_max , min( calcium_bg_ref_range_lower) as calcium_bg_art_ref_range_lower , max( calcium_bg_ref_range_upper) as calcium_bg_art_ref_range_upper\
\
  , min(glucose_bg) as glucose_bg_art_min , max(glucose_bg) as glucose_bg_art_max , min( glucose_bg_ref_range_lower) as glucose_bg_art_ref_range_lower , max( glucose_bg_ref_range_upper) as glucose_bg_art_ref_range_upper\
\
  , min(potassium_bg) as potassium_bg_art_min , max(potassium_bg) as potassium_bg_art_max , min( potassium_bg_ref_range_lower) as potassium_bg_art_ref_range_lower , max( potassium_bg_ref_range_upper) as potassium_bg_art_ref_range_upper\
\
  , min(sodium_bg) as sodium_bg_art_min , max(sodium_bg) as sodium_bg_art_max , min( sodium_bg_ref_range_lower) as sodium_bg_art_ref_range_lower , max( sodium_bg_ref_range_upper) as sodium_bg_art_ref_range_upper\
\
    FROM first_hep_with_hep_type_and_treatment_type_dermographics HEP_COHORT # To get the final hep admin cohort. This contain had_id, but not subject_id (coz of the way we grouped)\
    LEFT JOIN `physionet-data.mimic_icu.icustays` ie # to get subject_id. To join with below table (item_id_to_name_after_joining_specimen), 'subject_id' required, coz hadm_id of most of the rows in below table are blank.\
    ON HEP_COHORT.STAY_ID = IE.STAY_ID\
\
LEFT JOIN item_id_to_name_after_joining_specimen le  # To get lab result details\
        ON ie.subject_id = le.subject_id # some hadm_id in lab test results (`physionet-data.mimic_hosp.labevents`) are blank. That's why subject_id was used to join\
        where le.charttime >= DATETIME_SUB(HEP_COHORT.hep_start, INTERVAL '1' DAY) # 24 hours prior to first hep dose\
        AND le.charttime <= hep_start # lab tests before first hep\
        AND le.specimen = 'ART.'\
    GROUP BY HEP_COHORT.hadm_id\
\
)\
\
-- 4. set reference ranges for vital signs in mimic_derived.vitalsings table and mimic_derived.sofa table , mimic_derived.gcs tables\
-- for these vital signs, lower and upper refeerence range is doesn't change from person to person\
\
\
 , set_ranges_for_vitalsigns_and_sofa_and_gcs as \
 (\
select \
   HEP_COHORT.hadm_id\
\
 , min(heart_rate) as heart_rate_vital_min\
 , max(heart_rate) as heart_rate_vital_max\
 , avg(heart_rate) as heart_rate_vital_mean\
 , 60 as heart_rate_vital_ref_range_lower\
 , 100 as heart_rate_vital_ref_range_upper\
\
 , min(sbp) as sbp_vital_min\
 , max(sbp) as sbp_vital_max\
 , avg(sbp) as sbp_vital_mean\
 , 90 as sbp_vital_ref_range_lower\
 , 140 as sbp_vital_ref_range_upper\
\
 , min(dbp) as dbp_vital_min\
 , max(dbp) as dbp_vital_max\
 , avg(dbp) as dbp_vital_mean\
 , 60 as dbp_vital_ref_range_lower\
 , 90 as dbp_vital_ref_range_upper\
\
 , min(mbp) as mbp_vital_min\
 , max(mbp) as mbp_vital_max\
 , avg(mbp) as mbp_vital_mean\
 , 70 as mbp_vital_ref_range_lower\
 , 100 as mbp_vital_ref_range_upper\
\
 , min(resp_rate) as resp_rate_vital_min\
 , max(resp_rate) as resp_rate_vital_max\
 , avg(resp_rate) as resp_rate_vital_mean\
 , 12 as resp_rate_vital_ref_range_lower\
 , 20 as resp_rate_vital_ref_range_upper\
\
 , min(temperature) as temperature_vital_min\
 , max(temperature) as temperature_vital_max\
 , avg(temperature) as temperature_vital_mean\
 , 36.5 as temperature_vital_ref_range_lower\
 , 37.3 as temperature_vital_ref_range_upper\
\
 , min(spo2) as spo2_vital_min\
 , max(spo2) as spo2_vital_max\
 , avg(spo2) as spo2_vital_mean\
 , 95 as spo2_vital_ref_range_lower\
 , 100 as spo2_vital_ref_range_upper\
\
 , min(glucose) as glucose_vital_min\
 , max(glucose) as glucose_vital_max\
 , avg(glucose) as glucose_vital_mean\
 , 70 as glucose_vital_ref_range_lower\
 , 100 as glucose_vital_ref_range_upper\
\
\
-- sofa -- Exclude SOFA related data, as SOFA is built using already available lab tests and vital signs, those we might already used\
\
/*\
 , sofa.SOFA\
 , 0 as SOFA_ref_range_lower\
 , 0 as SOFA_ref_range_upper\
\
 , sofa.respiration\
 , 0 as respiration_ref_range_lower\
 , 0 as respiration_ref_range_upper\
\
 , sofa.coagulation\
 , 0 as coagulation_ref_range_lower\
 , 0 as coagulation_ref_range_upper\
\
 , sofa.liver\
 , 0 as liver_ref_range_lower\
 , 0 as liver_ref_range_upper\
\
 , sofa.cardiovascular\
 , 0 as cardiovascular_ref_range_lower\
 , 0 as cardiovascular_ref_range_upper\
            \
 , sofa.cns\
 , 0 as cns_ref_range_lower\
 , 0 as cns_ref_range_upper\
\
 , sofa.renal\
 , 0 as renal_ref_range_lower\
 , 0 as renal_ref_range_upper\
\
*/\
\
-- gcs\
\
, min(gcs) as gcs_min --min, max, avg\
\
\
 , 15 as gcs_min_ref_range_lower\
 , 15 as gcs_min_ref_range_upper\
\
/* Just 'gcs' is enough. No need to go to this depth.\
 , gcs.gcs_motor\
 , 6 as gcs_motor_ref_range_lower\
 , 6 as gcs_motor_ref_range_upper\
\
 , gcs.gcs_verbal\
 , 5 as gcs_verbal_ref_range_lower\
 , 5 as gcs_verbal_ref_range_upper\
\
 , gcs.gcs_eyes\
 , 4 as gcs_eyes_ref_range_lower\
 , 4 as gcs_eyes_ref_range_upper\
\
 , gcs.gcs_unable   -- this is a binary flag. This is '1', if itemid =223900 (this is the item id of gcs_verbal) and VALUE = 'No Response-ETT'.O.W, this flag = 0\
 , 0 as gcs_unable_ref_range_lower\
 , 0 as gcs_unable_ref_range_upper\
*/\
\
FROM first_hep_with_hep_type_and_treatment_type_dermographics HEP_COHORT # To get the final hep admin cohort. This contain had_id, but not subject_id (coz of the way we grouped)\
    LEFT JOIN `physionet-data.mimic_core.admissions`adm -- to get subject_id. coz vitalsigns and gcs table doesn't contain hadm_id, so that subject_id is required to join.\
\
    ON HEP_COHORT.hadm_id = adm.hadm_id\
\
    LEFT JOIN `physionet-data.mimic_derived.vitalsign` vitalsign # to get vital signs\
    ON adm.subject_id = vitalsign.subject_id\
  \
    LEFT JOIN `physionet-data.mimic_derived.gcs` gcsall # to get vital signs\
    ON adm.subject_id = gcsall.subject_id\
        where vitalsign.charttime >= DATETIME_SUB(HEP_COHORT.hep_start, INTERVAL '1' DAY) # 24 hours prior to first hep dose\
        AND gcsall.charttime >= DATETIME_SUB(HEP_COHORT.hep_start, INTERVAL '1' DAY)\
        AND vitalsign.charttime <= hep_start # lab tests before first hep\
        AND gcsall.charttime <= hep_start # lab tests before first hep\
    \
    GROUP BY HEP_COHORT.hadm_id\
 )\
\
--5. join all vital signs and ref ranges (lab, bg / bg_art / vital,gcs,sofa) , so that combine same attributes from differnt test, in next step\
, join_all as\
(\
select hep_cohort.*, \
       lab_and_bg.*except(subject_id, hadm_id, stay_id)\
       ,bg_art.* except(subject_id,hadm_id,stay_id),\
       vitalsigns_and_sofa_and_gcs.* except(hadm_id) \
\
from first_hep_with_hep_type_and_treatment_type_dermographics hep_cohort\
left join  first24h_vital_signs lab_and_bg\
on hep_cohort.hadm_id = lab_and_bg.hadm_id\
\
left join first24h_bg_ART bg_art\
on hep_cohort.hadm_id = bg_art.hadm_id\
\
left join set_ranges_for_vitalsigns_and_sofa_and_gcs vitalsigns_and_sofa_and_gcs\
on hep_cohort.hadm_id = vitalsigns_and_sofa_and_gcs.hadm_id\
)\
-- first24h_vital_signs , first24h_bg_ART , set_ranges_for_vitalsigns_and_sofa_and_gcs\
\
--output after combining\
, after_deleting_duplicate_attributes as\
(\
SELECT \
   subject_id, hadm_id, join_all.stay_id\
\
-- Merge possible records in bg and bg_art tests\
\
, Case When (hematocrit_bg_min <= hematocrit_bg_art_min OR hematocrit_bg_art_min is null) THEN hematocrit_bg_min Else hematocrit_bg_art_min End As hematocrit_bg_min , Case When (hematocrit_bg_max >= hematocrit_bg_art_max OR hematocrit_bg_art_max is null) THEN hematocrit_bg_max Else hematocrit_bg_art_max End As hematocrit_bg_max , Case When (hematocrit_bg_ref_range_lower <= hematocrit_bg_art_ref_range_lower OR hematocrit_bg_art_ref_range_lower is null) THEN hematocrit_bg_ref_range_lower Else hematocrit_bg_art_ref_range_lower End As hematocrit_bg_ref_range_lower , Case When (hematocrit_bg_ref_range_upper >= hematocrit_bg_art_ref_range_upper OR hematocrit_bg_art_ref_range_upper is null) THEN hematocrit_bg_ref_range_upper Else hematocrit_bg_art_ref_range_upper End As hematocrit_bg_ref_range_upper\
\
, Case When (hemoglobin_bg_min <= hemoglobin_bg_art_min OR hemoglobin_bg_art_min is null) THEN hemoglobin_bg_min Else hemoglobin_bg_art_min End As hemoglobin_bg_min , Case When (hemoglobin_bg_max >= hemoglobin_bg_art_max OR hemoglobin_bg_art_max is null) THEN hemoglobin_bg_max Else hemoglobin_bg_art_max End As hemoglobin_bg_max , Case When (hemoglobin_bg_ref_range_lower <= hemoglobin_bg_art_ref_range_lower OR hemoglobin_bg_art_ref_range_lower is null) THEN hemoglobin_bg_ref_range_lower Else hemoglobin_bg_art_ref_range_lower End As hemoglobin_bg_ref_range_lower , Case When (hemoglobin_bg_ref_range_upper >= hemoglobin_bg_art_ref_range_upper OR hemoglobin_bg_art_ref_range_upper is null) THEN hemoglobin_bg_ref_range_upper Else hemoglobin_bg_art_ref_range_upper End As hemoglobin_bg_ref_range_upper\
\
, Case When (bicarbonate_bg_min <= bicarbonate_bg_art_min OR bicarbonate_bg_art_min is null) THEN bicarbonate_bg_min Else bicarbonate_bg_art_min End As bicarbonate_bg_min , Case When (bicarbonate_bg_max >= bicarbonate_bg_art_max OR bicarbonate_bg_art_max is null) THEN bicarbonate_bg_max Else bicarbonate_bg_art_max End As bicarbonate_bg_max , Case When (bicarbonate_bg_ref_range_lower <= bicarbonate_bg_art_ref_range_lower OR bicarbonate_bg_art_ref_range_lower is null) THEN bicarbonate_bg_ref_range_lower Else bicarbonate_bg_art_ref_range_lower End As bicarbonate_bg_ref_range_lower , Case When (bicarbonate_bg_ref_range_upper >= bicarbonate_bg_art_ref_range_upper OR bicarbonate_bg_art_ref_range_upper is null) THEN bicarbonate_bg_ref_range_upper Else bicarbonate_bg_art_ref_range_upper End As bicarbonate_bg_ref_range_upper\
\
, Case When (calcium_bg_min <= calcium_bg_art_min OR calcium_bg_art_min is null) THEN calcium_bg_min Else calcium_bg_art_min End As calcium_bg_min , Case When (calcium_bg_max >= calcium_bg_art_max OR calcium_bg_art_max is null) THEN calcium_bg_max Else calcium_bg_art_max End As calcium_bg_max , Case When (calcium_bg_ref_range_lower <= calcium_bg_art_ref_range_lower OR calcium_bg_art_ref_range_lower is null) THEN calcium_bg_ref_range_lower Else calcium_bg_art_ref_range_lower End As calcium_bg_ref_range_lower , Case When (calcium_bg_ref_range_upper >= calcium_bg_art_ref_range_upper OR calcium_bg_art_ref_range_upper is null) THEN calcium_bg_ref_range_upper Else calcium_bg_art_ref_range_upper End As calcium_bg_ref_range_upper\
\
, Case When (chloride_bg_min <= chloride_bg_art_min OR chloride_bg_art_min is null) THEN chloride_bg_min Else chloride_bg_art_min End As chloride_bg_min , Case When (chloride_bg_max >= chloride_bg_art_max OR chloride_bg_art_max is null) THEN chloride_bg_max Else chloride_bg_art_max End As chloride_bg_max , Case When (chloride_bg_ref_range_lower <= chloride_bg_art_ref_range_lower OR chloride_bg_art_ref_range_lower is null) THEN chloride_bg_ref_range_lower Else chloride_bg_art_ref_range_lower End As chloride_bg_ref_range_lower , Case When (chloride_bg_ref_range_upper >= chloride_bg_art_ref_range_upper OR chloride_bg_art_ref_range_upper is null) THEN chloride_bg_ref_range_upper Else chloride_bg_art_ref_range_upper End As chloride_bg_ref_range_upper\
\
, Case When (sodium_bg_min <= sodium_bg_art_min OR sodium_bg_art_min is null) THEN sodium_bg_min Else sodium_bg_art_min End As sodium_bg_min , Case When (sodium_bg_max >= sodium_bg_art_max OR sodium_bg_art_max is null) THEN sodium_bg_max Else sodium_bg_art_max End As sodium_bg_max , Case When (sodium_bg_ref_range_lower <= sodium_bg_art_ref_range_lower OR sodium_bg_art_ref_range_lower is null) THEN sodium_bg_ref_range_lower Else sodium_bg_art_ref_range_lower End As sodium_bg_ref_range_lower , Case When (sodium_bg_ref_range_upper >= sodium_bg_art_ref_range_upper OR sodium_bg_art_ref_range_upper is null) THEN sodium_bg_ref_range_upper Else sodium_bg_art_ref_range_upper End As sodium_bg_ref_range_upper\
\
, Case When (potassium_bg_min <= potassium_bg_art_min OR potassium_bg_art_min is null) THEN potassium_bg_min Else potassium_bg_art_min End As potassium_bg_min , Case When (potassium_bg_max >= potassium_bg_art_max OR potassium_bg_art_max is null) THEN potassium_bg_max Else potassium_bg_art_max End As potassium_bg_max , Case When (potassium_bg_ref_range_lower <= potassium_bg_art_ref_range_lower OR potassium_bg_art_ref_range_lower is null) THEN potassium_bg_ref_range_lower Else potassium_bg_art_ref_range_lower End As potassium_bg_ref_range_lower , Case When (potassium_bg_ref_range_upper >= potassium_bg_art_ref_range_upper OR potassium_bg_art_ref_range_upper is null) THEN potassium_bg_ref_range_upper Else potassium_bg_art_ref_range_upper End As potassium_bg_ref_range_upper\
\
, Case When (lactate_bg_min <= lactate_bg_art_min OR lactate_bg_art_min is null) THEN lactate_bg_min Else lactate_bg_art_min End As lactate_min , Case When (lactate_bg_max >= lactate_bg_art_max OR lactate_bg_art_max is null) THEN lactate_bg_max Else lactate_bg_art_max End As lactate_max , Case When (lactate_bg_ref_range_lower <= lactate_bg_art_ref_range_lower OR lactate_bg_art_ref_range_lower is null) THEN lactate_bg_ref_range_lower Else lactate_bg_art_ref_range_lower End As lactate_ref_range_lower , Case When (lactate_bg_ref_range_upper >= lactate_bg_art_ref_range_upper OR lactate_bg_art_ref_range_upper is null) THEN lactate_bg_ref_range_upper Else lactate_bg_art_ref_range_upper End As lactate_ref_range_upper\
\
, Case When (ph_bg_min <= ph_bg_art_min OR ph_bg_art_min is null) THEN ph_bg_min Else ph_bg_art_min End As ph_min , Case When (ph_bg_max >= ph_bg_art_max OR ph_bg_art_max is null) THEN ph_bg_max Else ph_bg_art_max End As ph_max , Case When (ph_bg_ref_range_lower <= ph_bg_art_ref_range_lower OR ph_bg_art_ref_range_lower is null) THEN ph_bg_ref_range_lower Else ph_bg_art_ref_range_lower End As ph_ref_range_lower , Case When (ph_bg_ref_range_upper >= ph_bg_art_ref_range_upper OR ph_bg_art_ref_range_upper is null) THEN ph_bg_ref_range_upper Else ph_bg_art_ref_range_upper End As ph_ref_range_upper\
\
, Case When (baseexcess_bg_min <= baseexcess_bg_art_min OR baseexcess_bg_art_min is null) THEN baseexcess_bg_min Else baseexcess_bg_art_min End As baseexcess_min , Case When (baseexcess_bg_max >= baseexcess_bg_art_max OR baseexcess_bg_art_max is null) THEN baseexcess_bg_max Else baseexcess_bg_art_max End As baseexcess_max , Case When (baseexcess_bg_ref_range_lower <= baseexcess_bg_art_ref_range_lower OR baseexcess_bg_art_ref_range_lower is null) THEN baseexcess_bg_ref_range_lower Else baseexcess_bg_art_ref_range_lower End As baseexcess_ref_range_lower , Case When (baseexcess_bg_ref_range_upper >= baseexcess_bg_art_ref_range_upper OR baseexcess_bg_art_ref_range_upper is null) THEN baseexcess_bg_ref_range_upper Else baseexcess_bg_art_ref_range_upper End As baseexcess_ref_range_upper\
\
, Case When (carboxyhemoglobin_bg_min <= carboxyhemoglobin_bg_art_min OR carboxyhemoglobin_bg_art_min is null) THEN carboxyhemoglobin_bg_min Else carboxyhemoglobin_bg_art_min End As carboxyhemoglobin_min , Case When (carboxyhemoglobin_bg_max >= carboxyhemoglobin_bg_art_max OR carboxyhemoglobin_bg_art_max is null) THEN carboxyhemoglobin_bg_max Else carboxyhemoglobin_bg_art_max End As carboxyhemoglobin_max , Case When (carboxyhemoglobin_bg_ref_range_lower <= carboxyhemoglobin_bg_art_ref_range_lower OR carboxyhemoglobin_bg_art_ref_range_lower is null) THEN carboxyhemoglobin_bg_ref_range_lower Else carboxyhemoglobin_bg_art_ref_range_lower End As carboxyhemoglobin_ref_range_lower , Case When (carboxyhemoglobin_bg_ref_range_upper >= carboxyhemoglobin_bg_art_ref_range_upper OR carboxyhemoglobin_bg_art_ref_range_upper is null) THEN carboxyhemoglobin_bg_ref_range_upper Else carboxyhemoglobin_bg_art_ref_range_upper End As carboxyhemoglobin_ref_range_upper\
\
, Case When (methemoglobin_bg_min <= methemoglobin_bg_art_min OR methemoglobin_bg_art_min is null) THEN methemoglobin_bg_min Else methemoglobin_bg_art_min End As methemoglobin_min , Case When (methemoglobin_bg_max >= methemoglobin_bg_art_max OR methemoglobin_bg_art_max is null) THEN methemoglobin_bg_max Else methemoglobin_bg_art_max End As methemoglobin_max , Case When (methemoglobin_bg_ref_range_lower <= methemoglobin_bg_art_ref_range_lower OR methemoglobin_bg_art_ref_range_lower is null) THEN methemoglobin_bg_ref_range_lower Else methemoglobin_bg_art_ref_range_lower End As methemoglobin_ref_range_lower , Case When (methemoglobin_bg_ref_range_upper >= methemoglobin_bg_art_ref_range_upper OR methemoglobin_bg_art_ref_range_upper is null) THEN methemoglobin_bg_ref_range_upper Else methemoglobin_bg_art_ref_range_upper End As methemoglobin_ref_range_upper\
\
, Case When (temperature_bg_min <= temperature_bg_art_min OR temperature_bg_art_min is null) THEN temperature_bg_min Else temperature_bg_art_min End As temperature_bg_min , Case When (temperature_bg_max >= temperature_bg_art_max OR temperature_bg_art_max is null) THEN temperature_bg_max Else temperature_bg_art_max End As temperature_bg_max , Case When (temperature_bg_ref_range_lower <= temperature_bg_art_ref_range_lower OR temperature_bg_art_ref_range_lower is null) THEN temperature_bg_ref_range_lower Else temperature_bg_art_ref_range_lower End As temperature_bg_ref_range_lower , Case When (temperature_bg_ref_range_upper >= temperature_bg_art_ref_range_upper OR temperature_bg_art_ref_range_upper is null) THEN temperature_bg_ref_range_upper Else temperature_bg_art_ref_range_upper End As temperature_bg_ref_range_upper\
\
, Case When (glucose_bg_min <= glucose_bg_art_min OR glucose_bg_art_min is null) THEN glucose_bg_min Else glucose_bg_art_min End As glucose_bg_min , Case When (glucose_bg_max >= glucose_bg_art_max OR glucose_bg_art_max is null) THEN glucose_bg_max Else glucose_bg_art_max End As glucose_bg_max , Case When (glucose_bg_ref_range_lower <= glucose_bg_art_ref_range_lower OR glucose_bg_art_ref_range_lower is null) THEN glucose_bg_ref_range_lower Else glucose_bg_art_ref_range_lower End As glucose_bg_ref_range_lower , Case When (glucose_bg_ref_range_upper >= glucose_bg_art_ref_range_upper OR glucose_bg_art_ref_range_upper is null) THEN glucose_bg_ref_range_upper Else glucose_bg_art_ref_range_upper End As glucose_bg_ref_range_upper\
\
--old\
\
, hematocrit_lab_min,  hematocrit_lab_max, hematocrit_lab_ref_range_lower, hematocrit_lab_ref_range_upper\
, hemoglobin_lab_min,  hemoglobin_lab_max, hemoglobin_lab_ref_range_lower, hemoglobin_lab_ref_range_upper\
, bicarbonate_lab_min,  bicarbonate_lab_max, bicarbonate_lab_ref_range_lower, bicarbonate_lab_ref_range_upper\
, calcium_lab_min,  calcium_lab_max, calcium_lab_ref_range_lower, calcium_lab_ref_range_upper\
, chloride_lab_min,  chloride_lab_max, chloride_lab_ref_range_lower, chloride_lab_ref_range_upper\
, sodium_lab_min,  sodium_lab_max, sodium_lab_ref_range_lower, sodium_lab_ref_range_upper\
, potassium_lab_min,  potassium_lab_max, potassium_lab_ref_range_lower, potassium_lab_ref_range_upper\
, glucose_lab_min,  glucose_lab_max, glucose_lab_ref_range_lower, glucose_lab_ref_range_upper\
\
\
 , platelets_lab_min as platelets_min ,  platelets_lab_max as platelets_max  , platelets_lab_ref_range_lower as platelets_ref_range_lower , platelets_lab_ref_range_upper as platelets_ref_range_upper\
 , wbc_lab_min as wbc_min ,  wbc_lab_max as wbc_max  , wbc_lab_ref_range_lower as wbc_ref_range_lower , wbc_lab_ref_range_upper as wbc_ref_range_upper\
 , albumin_lab_min as albumin_min ,  albumin_lab_max as albumin_max  , albumin_lab_ref_range_lower as albumin_ref_range_lower , albumin_lab_ref_range_upper as albumin_ref_range_upper\
 , globulin_lab_min as globulin_min ,  globulin_lab_max as globulin_max  , globulin_lab_ref_range_lower as globulin_ref_range_lower , globulin_lab_ref_range_upper as globulin_ref_range_upper\
 , total_protein_lab_min as total_protein_min ,  total_protein_lab_max as total_protein_max  , total_protein_lab_ref_range_lower as total_protein_ref_range_lower , total_protein_lab_ref_range_upper as total_protein_ref_range_upper\
 , aniongap_lab_min as aniongap_min ,  aniongap_lab_max as aniongap_max  , aniongap_lab_ref_range_lower as aniongap_ref_range_lower , aniongap_lab_ref_range_upper as aniongap_ref_range_upper\
 , bun_lab_min as bun_min ,  bun_lab_max as bun_max  , bun_lab_ref_range_lower as bun_ref_range_lower , bun_lab_ref_range_upper as bun_ref_range_upper\
 , creatinine_lab_min as creatinine_min ,  creatinine_lab_max as creatinine_max  , creatinine_lab_ref_range_lower as creatinine_ref_range_lower , creatinine_lab_ref_range_upper as creatinine_ref_range_upper\
 , abs_basophils_lab_min as abs_basophils_min ,  abs_basophils_lab_max as abs_basophils_max  , abs_basophils_lab_ref_range_lower as abs_basophils_ref_range_lower , abs_basophils_lab_ref_range_upper as abs_basophils_ref_range_upper\
 , abs_eosinophils_lab_min as abs_eosinophils_min ,  abs_eosinophils_lab_max as abs_eosinophils_max  , abs_eosinophils_lab_ref_range_lower as abs_eosinophils_ref_range_lower , abs_eosinophils_lab_ref_range_upper as abs_eosinophils_ref_range_upper\
 , abs_lymphocytes_lab_min as abs_lymphocytes_min ,  abs_lymphocytes_lab_max as abs_lymphocytes_max  , abs_lymphocytes_lab_ref_range_lower as abs_lymphocytes_ref_range_lower , abs_lymphocytes_lab_ref_range_upper as abs_lymphocytes_ref_range_upper\
 , abs_monocytes_lab_min as abs_monocytes_min ,  abs_monocytes_lab_max as abs_monocytes_max  , abs_monocytes_lab_ref_range_lower as abs_monocytes_ref_range_lower , abs_monocytes_lab_ref_range_upper as abs_monocytes_ref_range_upper\
 , abs_neutrophils_lab_min as abs_neutrophils_min ,  abs_neutrophils_lab_max as abs_neutrophils_max  , abs_neutrophils_lab_ref_range_lower as abs_neutrophils_ref_range_lower , abs_neutrophils_lab_ref_range_upper as abs_neutrophils_ref_range_upper\
 , atyps_lab_min as atyps_min ,  atyps_lab_max as atyps_max  , atyps_lab_ref_range_lower as atyps_ref_range_lower , atyps_lab_ref_range_upper as atyps_ref_range_upper\
 , bands_lab_min as bands_min ,  bands_lab_max as bands_max  , bands_lab_ref_range_lower as bands_ref_range_lower , bands_lab_ref_range_upper as bands_ref_range_upper\
 , imm_granulocytes_lab_min as imm_granulocytes_min ,  imm_granulocytes_lab_max as imm_granulocytes_max  , imm_granulocytes_lab_ref_range_lower as imm_granulocytes_ref_range_lower , imm_granulocytes_lab_ref_range_upper as imm_granulocytes_ref_range_upper\
 , metas_lab_min as metas_min ,  metas_lab_max as metas_max  , metas_lab_ref_range_lower as metas_ref_range_lower , metas_lab_ref_range_upper as metas_ref_range_upper\
 , nrbc_lab_min as nrbc_min ,  nrbc_lab_max as nrbc_max  , nrbc_lab_ref_range_lower as nrbc_ref_range_lower , nrbc_lab_ref_range_upper as nrbc_ref_range_upper\
 , d_dimer_lab_min as d_dimer_min ,  d_dimer_lab_max as d_dimer_max  , d_dimer_lab_ref_range_lower as d_dimer_ref_range_lower , d_dimer_lab_ref_range_upper as d_dimer_ref_range_upper\
 , fibrinogen_lab_min as fibrinogen_min ,  fibrinogen_lab_max as fibrinogen_max  , fibrinogen_lab_ref_range_lower as fibrinogen_ref_range_lower , fibrinogen_lab_ref_range_upper as fibrinogen_ref_range_upper\
 , thrombin_lab_min as thrombin_min ,  thrombin_lab_max as thrombin_max  , thrombin_lab_ref_range_lower as thrombin_ref_range_lower , thrombin_lab_ref_range_upper as thrombin_ref_range_upper\
 , inr_lab_min as inr_min ,  inr_lab_max as inr_max  , inr_lab_ref_range_lower as inr_ref_range_lower , inr_lab_ref_range_upper as inr_ref_range_upper\
 , pt_lab_min as pt_min ,  pt_lab_max as pt_max  , pt_lab_ref_range_lower as pt_ref_range_lower , pt_lab_ref_range_upper as pt_ref_range_upper\
 , ptt_lab_min as ptt_min ,  ptt_lab_max as ptt_max  , ptt_lab_ref_range_lower as ptt_ref_range_lower , ptt_lab_ref_range_upper as ptt_ref_range_upper\
 , alt_lab_min as alt_min ,  alt_lab_max as alt_max  , alt_lab_ref_range_lower as alt_ref_range_lower , alt_lab_ref_range_upper as alt_ref_range_upper\
 , alp_lab_min as alp_min ,  alp_lab_max as alp_max  , alp_lab_ref_range_lower as alp_ref_range_lower , alp_lab_ref_range_upper as alp_ref_range_upper\
 , ast_lab_min as ast_min ,  ast_lab_max as ast_max  , ast_lab_ref_range_lower as ast_ref_range_lower , ast_lab_ref_range_upper as ast_ref_range_upper\
 , amylase_lab_min as amylase_min ,  amylase_lab_max as amylase_max  , amylase_lab_ref_range_lower as amylase_ref_range_lower , amylase_lab_ref_range_upper as amylase_ref_range_upper\
 , bilirubin_total_lab_min as bilirubin_total_min ,  bilirubin_total_lab_max as bilirubin_total_max  , bilirubin_total_lab_ref_range_lower as bilirubin_total_ref_range_lower , bilirubin_total_lab_ref_range_upper as bilirubin_total_ref_range_upper\
 , bilirubin_direct_lab_min as bilirubin_direct_min ,  bilirubin_direct_lab_max as bilirubin_direct_max  , bilirubin_direct_lab_ref_range_lower as bilirubin_direct_ref_range_lower , bilirubin_direct_lab_ref_range_upper as bilirubin_direct_ref_range_upper\
 , bilirubin_indirect_lab_min as bilirubin_indirect_min ,  bilirubin_indirect_lab_max as bilirubin_indirect_max  , bilirubin_indirect_lab_ref_range_lower as bilirubin_indirect_ref_range_lower , bilirubin_indirect_lab_ref_range_upper as bilirubin_indirect_ref_range_upper\
 , ck_cpk_lab_min as ck_cpk_min ,  ck_cpk_lab_max as ck_cpk_max  , ck_cpk_lab_ref_range_lower as ck_cpk_ref_range_lower , ck_cpk_lab_ref_range_upper as ck_cpk_ref_range_upper\
 , ck_mb_lab_min as ck_mb_min ,  ck_mb_lab_max as ck_mb_max  , ck_mb_lab_ref_range_lower as ck_mb_ref_range_lower , ck_mb_lab_ref_range_upper as ck_mb_ref_range_upper\
 , ggt_lab_min as ggt_min ,  ggt_lab_max as ggt_max  , ggt_lab_ref_range_lower as ggt_ref_range_lower , ggt_lab_ref_range_upper as ggt_ref_range_upper\
 , ld_ldh_lab_min as ld_ldh_min ,  ld_ldh_lab_max as ld_ldh_max  , ld_ldh_lab_ref_range_lower as ld_ldh_ref_range_lower , ld_ldh_lab_ref_range_upper as ld_ldh_ref_range_upper\
\
 , so2_bg_min , so2_bg_max , so2_bg_ref_range_lower , so2_bg_ref_range_upper\
 , po2_bg_min , po2_bg_max , po2_bg_ref_range_lower , po2_bg_ref_range_upper\
 , pco2_bg_min , pco2_bg_max , pco2_bg_ref_range_lower , pco2_bg_ref_range_upper\
 , aado2_bg_min , aado2_bg_max , aado2_bg_ref_range_lower , aado2_bg_ref_range_upper\
 , fio2_bg_min , fio2_bg_max , fio2_bg_ref_range_lower , fio2_bg_ref_range_upper\
 , totalco2_bg_min , totalco2_bg_max , totalco2_bg_ref_range_lower , totalco2_bg_ref_range_upper\
\
 , so2_bg_art_min , so2_bg_art_max , so2_bg_art_ref_range_lower , so2_bg_art_ref_range_upper\
 , po2_bg_art_min , po2_bg_art_max , po2_bg_art_ref_range_lower , po2_bg_art_ref_range_upper\
 , pco2_bg_art_min , pco2_bg_art_max , pco2_bg_art_ref_range_lower , pco2_bg_art_ref_range_upper\
 , aado2_bg_art_min , aado2_bg_art_max , aado2_bg_art_ref_range_lower , aado2_bg_art_ref_range_upper\
 , fio2_bg_art_min , fio2_bg_art_max , fio2_bg_art_ref_range_lower , fio2_bg_art_ref_range_upper\
 , totalco2_bg_art_min , totalco2_bg_art_max , totalco2_bg_art_ref_range_lower , totalco2_bg_art_ref_range_upper\
\
 , heart_rate_vital_min as heart_rate_min ,  heart_rate_vital_max as heart_rate_max  , heart_rate_vital_mean as heart_rate_mean ,heart_rate_vital_ref_range_lower as heart_rate_ref_range_lower , heart_rate_vital_ref_range_upper as heart_rate_ref_range_upper\
 , sbp_vital_min as sbp_min ,  sbp_vital_max as sbp_max  , sbp_vital_mean as sbp_mean ,sbp_vital_ref_range_lower as sbp_ref_range_lower , sbp_vital_ref_range_upper as sbp_ref_range_upper\
 , dbp_vital_min as dbp_min ,  dbp_vital_max as dbp_max  , dbp_vital_mean as dbp_mean ,dbp_vital_ref_range_lower as dbp_ref_range_lower , dbp_vital_ref_range_upper as dbp_ref_range_upper\
 , mbp_vital_min as mbp_min ,  mbp_vital_max as mbp_max  , mbp_vital_mean as mbp_mean ,mbp_vital_ref_range_lower as mbp_ref_range_lower , mbp_vital_ref_range_upper as mbp_ref_range_upper\
 , resp_rate_vital_min as resp_rate_min ,  resp_rate_vital_max as resp_rate_max  , resp_rate_vital_mean as resp_rate_mean ,resp_rate_vital_ref_range_lower as resp_rate_ref_range_lower , resp_rate_vital_ref_range_upper as resp_rate_ref_range_upper\
 , spo2_vital_min as spo2_min ,  spo2_vital_max as spo2_max  , spo2_vital_mean as spo2_mean ,spo2_vital_ref_range_lower as spo2_ref_range_lower , spo2_vital_ref_range_upper as spo2_ref_range_upper\
\
--new\
  , temperature_vital_min, temperature_vital_max, temperature_vital_mean, temperature_vital_ref_range_lower, temperature_vital_ref_range_upper\
  , glucose_vital_min, glucose_vital_max, glucose_vital_mean, glucose_vital_ref_range_lower, glucose_vital_ref_range_upper\
\
 --, SOFA , SOFA_ref_range_lower , SOFA_ref_range_upper\
 --, respiration , respiration_ref_range_lower , respiration_ref_range_upper\
 --, coagulation , coagulation_ref_range_lower , coagulation_ref_range_upper\
 --, liver , liver_ref_range_lower , liver_ref_range_upper\
 --, cardiovascular , cardiovascular_ref_range_lower , cardiovascular_ref_range_upper\
 --, cns , cns_ref_range_lower , cns_ref_range_upper\
 --, renal , renal_ref_range_lower , renal_ref_range_upper\
\
--old\
\
, gcs_min , gcs_min_ref_range_lower , gcs_min_ref_range_upper\
-- , gcs_motor , gcs_motor_ref_range_lower , gcs_motor_ref_range_upper\
-- , gcs_verbal , gcs_verbal_ref_range_lower , gcs_verbal_ref_range_upper\
-- , gcs_eyes , gcs_eyes_ref_range_lower , gcs_eyes_ref_range_upper\
-- , gcs_unable , gcs_unable_ref_range_lower , gcs_unable_ref_range_upper\
\
from join_all\
\
)\
\
, check_status_of_each_vital_sign_full_list as\
(\
select \
   subject_id , hadm_id , stay_id\
\
--parameters which were only in vital signs\
\
--old\
 , heart_rate_min , CASE WHEN (heart_rate_min BETWEEN heart_rate_ref_range_lower AND heart_rate_ref_range_upper) AND (heart_rate_ref_range_lower IS NOT NULL) AND (heart_rate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (heart_rate_min< heart_rate_ref_range_lower) AND (heart_rate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (heart_rate_min > heart_rate_ref_range_upper) AND (heart_rate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN heart_rate_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS heart_rate_min_status , heart_rate_max , CASE WHEN (heart_rate_max BETWEEN heart_rate_ref_range_lower AND heart_rate_ref_range_upper) AND (heart_rate_ref_range_lower IS NOT NULL) AND (heart_rate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (heart_rate_max< heart_rate_ref_range_lower) AND (heart_rate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (heart_rate_max > heart_rate_ref_range_upper) AND (heart_rate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN heart_rate_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS heart_rate_max_status , heart_rate_mean , CASE WHEN (heart_rate_mean BETWEEN heart_rate_ref_range_lower AND heart_rate_ref_range_upper) AND (heart_rate_ref_range_lower IS NOT NULL) AND (heart_rate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (heart_rate_mean < heart_rate_ref_range_lower) AND (heart_rate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (heart_rate_mean > heart_rate_ref_range_upper) AND (heart_rate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN heart_rate_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS heart_rate_mean_status\
 , sbp_min , CASE WHEN (sbp_min BETWEEN sbp_ref_range_lower AND sbp_ref_range_upper) AND (sbp_ref_range_lower IS NOT NULL) AND (sbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (sbp_min< sbp_ref_range_lower) AND (sbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (sbp_min > sbp_ref_range_upper) AND (sbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN sbp_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS sbp_min_status , sbp_max , CASE WHEN (sbp_max BETWEEN sbp_ref_range_lower AND sbp_ref_range_upper) AND (sbp_ref_range_lower IS NOT NULL) AND (sbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (sbp_max< sbp_ref_range_lower) AND (sbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (sbp_max > sbp_ref_range_upper) AND (sbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN sbp_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS sbp_max_status , sbp_mean , CASE WHEN (sbp_mean BETWEEN sbp_ref_range_lower AND sbp_ref_range_upper) AND (sbp_ref_range_lower IS NOT NULL) AND (sbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (sbp_mean < sbp_ref_range_lower) AND (sbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (sbp_mean > sbp_ref_range_upper) AND (sbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN sbp_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS sbp_mean_status\
 , dbp_min , CASE WHEN (dbp_min BETWEEN dbp_ref_range_lower AND dbp_ref_range_upper) AND (dbp_ref_range_lower IS NOT NULL) AND (dbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (dbp_min< dbp_ref_range_lower) AND (dbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (dbp_min > dbp_ref_range_upper) AND (dbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN dbp_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS dbp_min_status , dbp_max , CASE WHEN (dbp_max BETWEEN dbp_ref_range_lower AND dbp_ref_range_upper) AND (dbp_ref_range_lower IS NOT NULL) AND (dbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (dbp_max< dbp_ref_range_lower) AND (dbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (dbp_max > dbp_ref_range_upper) AND (dbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN dbp_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS dbp_max_status , dbp_mean , CASE WHEN (dbp_mean BETWEEN dbp_ref_range_lower AND dbp_ref_range_upper) AND (dbp_ref_range_lower IS NOT NULL) AND (dbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (dbp_mean < dbp_ref_range_lower) AND (dbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (dbp_mean > dbp_ref_range_upper) AND (dbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN dbp_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS dbp_mean_status\
 , mbp_min , CASE WHEN (mbp_min BETWEEN mbp_ref_range_lower AND mbp_ref_range_upper) AND (mbp_ref_range_lower IS NOT NULL) AND (mbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (mbp_min< mbp_ref_range_lower) AND (mbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (mbp_min > mbp_ref_range_upper) AND (mbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN mbp_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS mbp_min_status , mbp_max , CASE WHEN (mbp_max BETWEEN mbp_ref_range_lower AND mbp_ref_range_upper) AND (mbp_ref_range_lower IS NOT NULL) AND (mbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (mbp_max< mbp_ref_range_lower) AND (mbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (mbp_max > mbp_ref_range_upper) AND (mbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN mbp_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS mbp_max_status , mbp_mean , CASE WHEN (mbp_mean BETWEEN mbp_ref_range_lower AND mbp_ref_range_upper) AND (mbp_ref_range_lower IS NOT NULL) AND (mbp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (mbp_mean < mbp_ref_range_lower) AND (mbp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (mbp_mean > mbp_ref_range_upper) AND (mbp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN mbp_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS mbp_mean_status\
 , resp_rate_min , CASE WHEN (resp_rate_min BETWEEN resp_rate_ref_range_lower AND resp_rate_ref_range_upper) AND (resp_rate_ref_range_lower IS NOT NULL) AND (resp_rate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (resp_rate_min< resp_rate_ref_range_lower) AND (resp_rate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (resp_rate_min > resp_rate_ref_range_upper) AND (resp_rate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN resp_rate_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS resp_rate_min_status , resp_rate_max , CASE WHEN (resp_rate_max BETWEEN resp_rate_ref_range_lower AND resp_rate_ref_range_upper) AND (resp_rate_ref_range_lower IS NOT NULL) AND (resp_rate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (resp_rate_max< resp_rate_ref_range_lower) AND (resp_rate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (resp_rate_max > resp_rate_ref_range_upper) AND (resp_rate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN resp_rate_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS resp_rate_max_status , resp_rate_mean , CASE WHEN (resp_rate_mean BETWEEN resp_rate_ref_range_lower AND resp_rate_ref_range_upper) AND (resp_rate_ref_range_lower IS NOT NULL) AND (resp_rate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (resp_rate_mean < resp_rate_ref_range_lower) AND (resp_rate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (resp_rate_mean > resp_rate_ref_range_upper) AND (resp_rate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN resp_rate_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS resp_rate_mean_status\
 , spo2_min , CASE WHEN (spo2_min BETWEEN spo2_ref_range_lower AND spo2_ref_range_upper) AND (spo2_ref_range_lower IS NOT NULL) AND (spo2_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (spo2_min< spo2_ref_range_lower) AND (spo2_ref_range_lower IS NOT NULL) THEN 'low' WHEN (spo2_min > spo2_ref_range_upper) AND (spo2_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN spo2_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS spo2_min_status , spo2_max , CASE WHEN (spo2_max BETWEEN spo2_ref_range_lower AND spo2_ref_range_upper) AND (spo2_ref_range_lower IS NOT NULL) AND (spo2_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (spo2_max< spo2_ref_range_lower) AND (spo2_ref_range_lower IS NOT NULL) THEN 'low' WHEN (spo2_max > spo2_ref_range_upper) AND (spo2_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN spo2_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS spo2_max_status , spo2_mean , CASE WHEN (spo2_mean BETWEEN spo2_ref_range_lower AND spo2_ref_range_upper) AND (spo2_ref_range_lower IS NOT NULL) AND (spo2_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (spo2_mean < spo2_ref_range_lower) AND (spo2_ref_range_lower IS NOT NULL) THEN 'low' WHEN (spo2_mean > spo2_ref_range_upper) AND (spo2_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN spo2_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS spo2_mean_status\
\
--new\
\
, temperature_vital_min , CASE WHEN (temperature_vital_min BETWEEN temperature_vital_ref_range_lower AND temperature_vital_ref_range_upper) AND (temperature_vital_ref_range_lower IS NOT NULL) AND (temperature_vital_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (temperature_vital_min< temperature_vital_ref_range_lower) AND (temperature_vital_ref_range_lower IS NOT NULL) THEN 'low' WHEN (temperature_vital_min > temperature_vital_ref_range_upper) AND (temperature_vital_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN temperature_vital_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS temperature_vital_min_status , temperature_vital_max , CASE WHEN (temperature_vital_max BETWEEN temperature_vital_ref_range_lower AND temperature_vital_ref_range_upper) AND (temperature_vital_ref_range_lower IS NOT NULL) AND (temperature_vital_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (temperature_vital_max< temperature_vital_ref_range_lower) AND (temperature_vital_ref_range_lower IS NOT NULL) THEN 'low' WHEN (temperature_vital_max > temperature_vital_ref_range_upper) AND (temperature_vital_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN temperature_vital_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS temperature_vital_max_status , temperature_vital_mean , CASE WHEN (temperature_vital_mean BETWEEN temperature_vital_ref_range_lower AND temperature_vital_ref_range_upper) AND (temperature_vital_ref_range_lower IS NOT NULL) AND (temperature_vital_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (temperature_vital_mean < temperature_vital_ref_range_lower) AND (temperature_vital_ref_range_lower IS NOT NULL) THEN 'low' WHEN (temperature_vital_mean > temperature_vital_ref_range_upper) AND (temperature_vital_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN temperature_vital_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS temperature_vital_mean_status\
\
, glucose_vital_min , CASE WHEN (glucose_vital_min BETWEEN glucose_vital_ref_range_lower AND glucose_vital_ref_range_upper) AND (glucose_vital_ref_range_lower IS NOT NULL) AND (glucose_vital_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (glucose_vital_min< glucose_vital_ref_range_lower) AND (glucose_vital_ref_range_lower IS NOT NULL) THEN 'low' WHEN (glucose_vital_min > glucose_vital_ref_range_upper) AND (glucose_vital_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN glucose_vital_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS glucose_vital_min_status , glucose_vital_max , CASE WHEN (glucose_vital_max BETWEEN glucose_vital_ref_range_lower AND glucose_vital_ref_range_upper) AND (glucose_vital_ref_range_lower IS NOT NULL) AND (glucose_vital_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (glucose_vital_max< glucose_vital_ref_range_lower) AND (glucose_vital_ref_range_lower IS NOT NULL) THEN 'low' WHEN (glucose_vital_max > glucose_vital_ref_range_upper) AND (glucose_vital_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN glucose_vital_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS glucose_vital_max_status , glucose_vital_mean , CASE WHEN (glucose_vital_mean BETWEEN glucose_vital_ref_range_lower AND glucose_vital_ref_range_upper) AND (glucose_vital_ref_range_lower IS NOT NULL) AND (glucose_vital_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (glucose_vital_mean < glucose_vital_ref_range_lower) AND (glucose_vital_ref_range_lower IS NOT NULL) THEN 'low' WHEN (glucose_vital_mean > glucose_vital_ref_range_upper) AND (glucose_vital_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN glucose_vital_mean IS NULL THEN 'not ordered' ELSE 'no ref range' END AS glucose_vital_mean_status\
\
--parameters which were in bg, bg_art\
\
--new\
\
, hematocrit_bg_min , CASE WHEN (hematocrit_bg_min BETWEEN hematocrit_bg_ref_range_lower AND hematocrit_bg_ref_range_upper) AND (hematocrit_bg_ref_range_lower IS NOT NULL) AND (hematocrit_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hematocrit_bg_min< hematocrit_bg_ref_range_lower) AND (hematocrit_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hematocrit_bg_min > hematocrit_bg_ref_range_upper) AND (hematocrit_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hematocrit_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hematocrit_bg_min_status , hematocrit_bg_max , CASE WHEN (hematocrit_bg_max BETWEEN hematocrit_bg_ref_range_lower AND hematocrit_bg_ref_range_upper) AND (hematocrit_bg_ref_range_lower IS NOT NULL) AND (hematocrit_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hematocrit_bg_max< hematocrit_bg_ref_range_lower) AND (hematocrit_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hematocrit_bg_max > hematocrit_bg_ref_range_upper) AND (hematocrit_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hematocrit_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hematocrit_bg_max_status\
\
, hemoglobin_bg_min , CASE WHEN (hemoglobin_bg_min BETWEEN hemoglobin_bg_ref_range_lower AND hemoglobin_bg_ref_range_upper) AND (hemoglobin_bg_ref_range_lower IS NOT NULL) AND (hemoglobin_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hemoglobin_bg_min< hemoglobin_bg_ref_range_lower) AND (hemoglobin_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hemoglobin_bg_min > hemoglobin_bg_ref_range_upper) AND (hemoglobin_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hemoglobin_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hemoglobin_bg_min_status , hemoglobin_bg_max , CASE WHEN (hemoglobin_bg_max BETWEEN hemoglobin_bg_ref_range_lower AND hemoglobin_bg_ref_range_upper) AND (hemoglobin_bg_ref_range_lower IS NOT NULL) AND (hemoglobin_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hemoglobin_bg_max< hemoglobin_bg_ref_range_lower) AND (hemoglobin_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hemoglobin_bg_max > hemoglobin_bg_ref_range_upper) AND (hemoglobin_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hemoglobin_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hemoglobin_bg_max_status\
\
, bicarbonate_bg_min , CASE WHEN (bicarbonate_bg_min BETWEEN bicarbonate_bg_ref_range_lower AND bicarbonate_bg_ref_range_upper) AND (bicarbonate_bg_ref_range_lower IS NOT NULL) AND (bicarbonate_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bicarbonate_bg_min< bicarbonate_bg_ref_range_lower) AND (bicarbonate_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bicarbonate_bg_min > bicarbonate_bg_ref_range_upper) AND (bicarbonate_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bicarbonate_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bicarbonate_bg_min_status , bicarbonate_bg_max , CASE WHEN (bicarbonate_bg_max BETWEEN bicarbonate_bg_ref_range_lower AND bicarbonate_bg_ref_range_upper) AND (bicarbonate_bg_ref_range_lower IS NOT NULL) AND (bicarbonate_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bicarbonate_bg_max< bicarbonate_bg_ref_range_lower) AND (bicarbonate_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bicarbonate_bg_max > bicarbonate_bg_ref_range_upper) AND (bicarbonate_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bicarbonate_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bicarbonate_bg_max_status\
\
, calcium_bg_min , CASE WHEN (calcium_bg_min BETWEEN calcium_bg_ref_range_lower AND calcium_bg_ref_range_upper) AND (calcium_bg_ref_range_lower IS NOT NULL) AND (calcium_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (calcium_bg_min< calcium_bg_ref_range_lower) AND (calcium_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (calcium_bg_min > calcium_bg_ref_range_upper) AND (calcium_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN calcium_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS calcium_bg_min_status , calcium_bg_max , CASE WHEN (calcium_bg_max BETWEEN calcium_bg_ref_range_lower AND calcium_bg_ref_range_upper) AND (calcium_bg_ref_range_lower IS NOT NULL) AND (calcium_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (calcium_bg_max< calcium_bg_ref_range_lower) AND (calcium_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (calcium_bg_max > calcium_bg_ref_range_upper) AND (calcium_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN calcium_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS calcium_bg_max_status\
\
, chloride_bg_min , CASE WHEN (chloride_bg_min BETWEEN chloride_bg_ref_range_lower AND chloride_bg_ref_range_upper) AND (chloride_bg_ref_range_lower IS NOT NULL) AND (chloride_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (chloride_bg_min< chloride_bg_ref_range_lower) AND (chloride_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (chloride_bg_min > chloride_bg_ref_range_upper) AND (chloride_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN chloride_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS chloride_bg_min_status , chloride_bg_max , CASE WHEN (chloride_bg_max BETWEEN chloride_bg_ref_range_lower AND chloride_bg_ref_range_upper) AND (chloride_bg_ref_range_lower IS NOT NULL) AND (chloride_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (chloride_bg_max< chloride_bg_ref_range_lower) AND (chloride_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (chloride_bg_max > chloride_bg_ref_range_upper) AND (chloride_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN chloride_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS chloride_bg_max_status\
\
, sodium_bg_min , CASE WHEN (sodium_bg_min BETWEEN sodium_bg_ref_range_lower AND sodium_bg_ref_range_upper) AND (sodium_bg_ref_range_lower IS NOT NULL) AND (sodium_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (sodium_bg_min< sodium_bg_ref_range_lower) AND (sodium_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (sodium_bg_min > sodium_bg_ref_range_upper) AND (sodium_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN sodium_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS sodium_bg_min_status , sodium_bg_max , CASE WHEN (sodium_bg_max BETWEEN sodium_bg_ref_range_lower AND sodium_bg_ref_range_upper) AND (sodium_bg_ref_range_lower IS NOT NULL) AND (sodium_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (sodium_bg_max< sodium_bg_ref_range_lower) AND (sodium_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (sodium_bg_max > sodium_bg_ref_range_upper) AND (sodium_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN sodium_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS sodium_bg_max_status\
\
, potassium_bg_min , CASE WHEN (potassium_bg_min BETWEEN potassium_bg_ref_range_lower AND potassium_bg_ref_range_upper) AND (potassium_bg_ref_range_lower IS NOT NULL) AND (potassium_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (potassium_bg_min< potassium_bg_ref_range_lower) AND (potassium_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (potassium_bg_min > potassium_bg_ref_range_upper) AND (potassium_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN potassium_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS potassium_bg_min_status , potassium_bg_max , CASE WHEN (potassium_bg_max BETWEEN potassium_bg_ref_range_lower AND potassium_bg_ref_range_upper) AND (potassium_bg_ref_range_lower IS NOT NULL) AND (potassium_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (potassium_bg_max< potassium_bg_ref_range_lower) AND (potassium_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (potassium_bg_max > potassium_bg_ref_range_upper) AND (potassium_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN potassium_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS potassium_bg_max_status\
\
, temperature_bg_min , CASE WHEN (temperature_bg_min BETWEEN temperature_bg_ref_range_lower AND temperature_bg_ref_range_upper) AND (temperature_bg_ref_range_lower IS NOT NULL) AND (temperature_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (temperature_bg_min< temperature_bg_ref_range_lower) AND (temperature_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (temperature_bg_min > temperature_bg_ref_range_upper) AND (temperature_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN temperature_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS temperature_bg_min_status , temperature_bg_max , CASE WHEN (temperature_bg_max BETWEEN temperature_bg_ref_range_lower AND temperature_bg_ref_range_upper) AND (temperature_bg_ref_range_lower IS NOT NULL) AND (temperature_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (temperature_bg_max< temperature_bg_ref_range_lower) AND (temperature_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (temperature_bg_max > temperature_bg_ref_range_upper) AND (temperature_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN temperature_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS temperature_bg_max_status\
\
, glucose_bg_min , CASE WHEN (glucose_bg_min BETWEEN glucose_bg_ref_range_lower AND glucose_bg_ref_range_upper) AND (glucose_bg_ref_range_lower IS NOT NULL) AND (glucose_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (glucose_bg_min< glucose_bg_ref_range_lower) AND (glucose_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (glucose_bg_min > glucose_bg_ref_range_upper) AND (glucose_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN glucose_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS glucose_bg_min_status , glucose_bg_max , CASE WHEN (glucose_bg_max BETWEEN glucose_bg_ref_range_lower AND glucose_bg_ref_range_upper) AND (glucose_bg_ref_range_lower IS NOT NULL) AND (glucose_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (glucose_bg_max< glucose_bg_ref_range_lower) AND (glucose_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (glucose_bg_max > glucose_bg_ref_range_upper) AND (glucose_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN glucose_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS glucose_bg_max_status\
\
--old\
\
 , lactate_min , CASE WHEN (lactate_min BETWEEN lactate_ref_range_lower AND lactate_ref_range_upper) AND (lactate_ref_range_lower IS NOT NULL) AND (lactate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (lactate_min< lactate_ref_range_lower) AND (lactate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (lactate_min > lactate_ref_range_upper) AND (lactate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN lactate_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS lactate_min_status , lactate_max , CASE WHEN (lactate_max BETWEEN lactate_ref_range_lower AND lactate_ref_range_upper) AND (lactate_ref_range_lower IS NOT NULL) AND (lactate_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (lactate_max< lactate_ref_range_lower) AND (lactate_ref_range_lower IS NOT NULL) THEN 'low' WHEN (lactate_max > lactate_ref_range_upper) AND (lactate_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN lactate_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS lactate_max_status\
 , ph_min , CASE WHEN (ph_min BETWEEN ph_ref_range_lower AND ph_ref_range_upper) AND (ph_ref_range_lower IS NOT NULL) AND (ph_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ph_min< ph_ref_range_lower) AND (ph_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ph_min > ph_ref_range_upper) AND (ph_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ph_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ph_min_status , ph_max , CASE WHEN (ph_max BETWEEN ph_ref_range_lower AND ph_ref_range_upper) AND (ph_ref_range_lower IS NOT NULL) AND (ph_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ph_max< ph_ref_range_lower) AND (ph_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ph_max > ph_ref_range_upper) AND (ph_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ph_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ph_max_status\
 , baseexcess_min , CASE WHEN (baseexcess_min BETWEEN baseexcess_ref_range_lower AND baseexcess_ref_range_upper) AND (baseexcess_ref_range_lower IS NOT NULL) AND (baseexcess_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (baseexcess_min< baseexcess_ref_range_lower) AND (baseexcess_ref_range_lower IS NOT NULL) THEN 'low' WHEN (baseexcess_min > baseexcess_ref_range_upper) AND (baseexcess_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN baseexcess_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS baseexcess_min_status , baseexcess_max , CASE WHEN (baseexcess_max BETWEEN baseexcess_ref_range_lower AND baseexcess_ref_range_upper) AND (baseexcess_ref_range_lower IS NOT NULL) AND (baseexcess_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (baseexcess_max< baseexcess_ref_range_lower) AND (baseexcess_ref_range_lower IS NOT NULL) THEN 'low' WHEN (baseexcess_max > baseexcess_ref_range_upper) AND (baseexcess_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN baseexcess_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS baseexcess_max_status\
 , carboxyhemoglobin_min , CASE WHEN (carboxyhemoglobin_min BETWEEN carboxyhemoglobin_ref_range_lower AND carboxyhemoglobin_ref_range_upper) AND (carboxyhemoglobin_ref_range_lower IS NOT NULL) AND (carboxyhemoglobin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (carboxyhemoglobin_min< carboxyhemoglobin_ref_range_lower) AND (carboxyhemoglobin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (carboxyhemoglobin_min > carboxyhemoglobin_ref_range_upper) AND (carboxyhemoglobin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN carboxyhemoglobin_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS carboxyhemoglobin_min_status , carboxyhemoglobin_max , CASE WHEN (carboxyhemoglobin_max BETWEEN carboxyhemoglobin_ref_range_lower AND carboxyhemoglobin_ref_range_upper) AND (carboxyhemoglobin_ref_range_lower IS NOT NULL) AND (carboxyhemoglobin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (carboxyhemoglobin_max< carboxyhemoglobin_ref_range_lower) AND (carboxyhemoglobin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (carboxyhemoglobin_max > carboxyhemoglobin_ref_range_upper) AND (carboxyhemoglobin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN carboxyhemoglobin_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS carboxyhemoglobin_max_status\
 , methemoglobin_min , CASE WHEN (methemoglobin_min BETWEEN methemoglobin_ref_range_lower AND methemoglobin_ref_range_upper) AND (methemoglobin_ref_range_lower IS NOT NULL) AND (methemoglobin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (methemoglobin_min< methemoglobin_ref_range_lower) AND (methemoglobin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (methemoglobin_min > methemoglobin_ref_range_upper) AND (methemoglobin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN methemoglobin_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS methemoglobin_min_status , methemoglobin_max , CASE WHEN (methemoglobin_max BETWEEN methemoglobin_ref_range_lower AND methemoglobin_ref_range_upper) AND (methemoglobin_ref_range_lower IS NOT NULL) AND (methemoglobin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (methemoglobin_max< methemoglobin_ref_range_lower) AND (methemoglobin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (methemoglobin_max > methemoglobin_ref_range_upper) AND (methemoglobin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN methemoglobin_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS methemoglobin_max_status\
\
\
--parameters which were in lab\
\
--new\
\
, hematocrit_lab_min , CASE WHEN (hematocrit_lab_min BETWEEN hematocrit_lab_ref_range_lower AND hematocrit_lab_ref_range_upper) AND (hematocrit_lab_ref_range_lower IS NOT NULL) AND (hematocrit_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hematocrit_lab_min< hematocrit_lab_ref_range_lower) AND (hematocrit_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hematocrit_lab_min > hematocrit_lab_ref_range_upper) AND (hematocrit_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hematocrit_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hematocrit_lab_min_status , hematocrit_lab_max , CASE WHEN (hematocrit_lab_max BETWEEN hematocrit_lab_ref_range_lower AND hematocrit_lab_ref_range_upper) AND (hematocrit_lab_ref_range_lower IS NOT NULL) AND (hematocrit_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hematocrit_lab_max< hematocrit_lab_ref_range_lower) AND (hematocrit_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hematocrit_lab_max > hematocrit_lab_ref_range_upper) AND (hematocrit_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hematocrit_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hematocrit_lab_max_status\
\
, hemoglobin_lab_min , CASE WHEN (hemoglobin_lab_min BETWEEN hemoglobin_lab_ref_range_lower AND hemoglobin_lab_ref_range_upper) AND (hemoglobin_lab_ref_range_lower IS NOT NULL) AND (hemoglobin_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hemoglobin_lab_min< hemoglobin_lab_ref_range_lower) AND (hemoglobin_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hemoglobin_lab_min > hemoglobin_lab_ref_range_upper) AND (hemoglobin_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hemoglobin_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hemoglobin_lab_min_status , hemoglobin_lab_max , CASE WHEN (hemoglobin_lab_max BETWEEN hemoglobin_lab_ref_range_lower AND hemoglobin_lab_ref_range_upper) AND (hemoglobin_lab_ref_range_lower IS NOT NULL) AND (hemoglobin_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (hemoglobin_lab_max< hemoglobin_lab_ref_range_lower) AND (hemoglobin_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (hemoglobin_lab_max > hemoglobin_lab_ref_range_upper) AND (hemoglobin_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN hemoglobin_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS hemoglobin_lab_max_status\
\
, bicarbonate_lab_min , CASE WHEN (bicarbonate_lab_min BETWEEN bicarbonate_lab_ref_range_lower AND bicarbonate_lab_ref_range_upper) AND (bicarbonate_lab_ref_range_lower IS NOT NULL) AND (bicarbonate_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bicarbonate_lab_min< bicarbonate_lab_ref_range_lower) AND (bicarbonate_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bicarbonate_lab_min > bicarbonate_lab_ref_range_upper) AND (bicarbonate_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bicarbonate_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bicarbonate_lab_min_status , bicarbonate_lab_max , CASE WHEN (bicarbonate_lab_max BETWEEN bicarbonate_lab_ref_range_lower AND bicarbonate_lab_ref_range_upper) AND (bicarbonate_lab_ref_range_lower IS NOT NULL) AND (bicarbonate_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bicarbonate_lab_max< bicarbonate_lab_ref_range_lower) AND (bicarbonate_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bicarbonate_lab_max > bicarbonate_lab_ref_range_upper) AND (bicarbonate_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bicarbonate_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bicarbonate_lab_max_status\
\
, calcium_lab_min , CASE WHEN (calcium_lab_min BETWEEN calcium_lab_ref_range_lower AND calcium_lab_ref_range_upper) AND (calcium_lab_ref_range_lower IS NOT NULL) AND (calcium_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (calcium_lab_min< calcium_lab_ref_range_lower) AND (calcium_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (calcium_lab_min > calcium_lab_ref_range_upper) AND (calcium_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN calcium_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS calcium_lab_min_status , calcium_lab_max , CASE WHEN (calcium_lab_max BETWEEN calcium_lab_ref_range_lower AND calcium_lab_ref_range_upper) AND (calcium_lab_ref_range_lower IS NOT NULL) AND (calcium_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (calcium_lab_max< calcium_lab_ref_range_lower) AND (calcium_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (calcium_lab_max > calcium_lab_ref_range_upper) AND (calcium_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN calcium_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS calcium_lab_max_status\
\
, chloride_lab_min , CASE WHEN (chloride_lab_min BETWEEN chloride_lab_ref_range_lower AND chloride_lab_ref_range_upper) AND (chloride_lab_ref_range_lower IS NOT NULL) AND (chloride_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (chloride_lab_min< chloride_lab_ref_range_lower) AND (chloride_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (chloride_lab_min > chloride_lab_ref_range_upper) AND (chloride_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN chloride_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS chloride_lab_min_status , chloride_lab_max , CASE WHEN (chloride_lab_max BETWEEN chloride_lab_ref_range_lower AND chloride_lab_ref_range_upper) AND (chloride_lab_ref_range_lower IS NOT NULL) AND (chloride_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (chloride_lab_max< chloride_lab_ref_range_lower) AND (chloride_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (chloride_lab_max > chloride_lab_ref_range_upper) AND (chloride_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN chloride_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS chloride_lab_max_status\
\
, sodium_lab_min , CASE WHEN (sodium_lab_min BETWEEN sodium_lab_ref_range_lower AND sodium_lab_ref_range_upper) AND (sodium_lab_ref_range_lower IS NOT NULL) AND (sodium_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (sodium_lab_min< sodium_lab_ref_range_lower) AND (sodium_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (sodium_lab_min > sodium_lab_ref_range_upper) AND (sodium_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN sodium_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS sodium_lab_min_status , sodium_lab_max , CASE WHEN (sodium_lab_max BETWEEN sodium_lab_ref_range_lower AND sodium_lab_ref_range_upper) AND (sodium_lab_ref_range_lower IS NOT NULL) AND (sodium_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (sodium_lab_max< sodium_lab_ref_range_lower) AND (sodium_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (sodium_lab_max > sodium_lab_ref_range_upper) AND (sodium_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN sodium_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS sodium_lab_max_status\
\
, potassium_lab_min , CASE WHEN (potassium_lab_min BETWEEN potassium_lab_ref_range_lower AND potassium_lab_ref_range_upper) AND (potassium_lab_ref_range_lower IS NOT NULL) AND (potassium_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (potassium_lab_min< potassium_lab_ref_range_lower) AND (potassium_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (potassium_lab_min > potassium_lab_ref_range_upper) AND (potassium_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN potassium_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS potassium_lab_min_status , potassium_lab_max , CASE WHEN (potassium_lab_max BETWEEN potassium_lab_ref_range_lower AND potassium_lab_ref_range_upper) AND (potassium_lab_ref_range_lower IS NOT NULL) AND (potassium_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (potassium_lab_max< potassium_lab_ref_range_lower) AND (potassium_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (potassium_lab_max > potassium_lab_ref_range_upper) AND (potassium_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN potassium_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS potassium_lab_max_status\
\
, glucose_lab_min , CASE WHEN (glucose_lab_min BETWEEN glucose_lab_ref_range_lower AND glucose_lab_ref_range_upper) AND (glucose_lab_ref_range_lower IS NOT NULL) AND (glucose_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (glucose_lab_min< glucose_lab_ref_range_lower) AND (glucose_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (glucose_lab_min > glucose_lab_ref_range_upper) AND (glucose_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN glucose_lab_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS glucose_lab_min_status , glucose_lab_max , CASE WHEN (glucose_lab_max BETWEEN glucose_lab_ref_range_lower AND glucose_lab_ref_range_upper) AND (glucose_lab_ref_range_lower IS NOT NULL) AND (glucose_lab_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (glucose_lab_max< glucose_lab_ref_range_lower) AND (glucose_lab_ref_range_lower IS NOT NULL) THEN 'low' WHEN (glucose_lab_max > glucose_lab_ref_range_upper) AND (glucose_lab_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN glucose_lab_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS glucose_lab_max_status\
\
-- old\
\
\
 , platelets_min , CASE WHEN (platelets_min BETWEEN platelets_ref_range_lower AND platelets_ref_range_upper) AND (platelets_ref_range_lower IS NOT NULL) AND (platelets_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (platelets_min< platelets_ref_range_lower) AND (platelets_ref_range_lower IS NOT NULL) THEN 'low' WHEN (platelets_min > platelets_ref_range_upper) AND (platelets_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN platelets_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS platelets_min_status , platelets_max , CASE WHEN (platelets_max BETWEEN platelets_ref_range_lower AND platelets_ref_range_upper) AND (platelets_ref_range_lower IS NOT NULL) AND (platelets_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (platelets_max< platelets_ref_range_lower) AND (platelets_ref_range_lower IS NOT NULL) THEN 'low' WHEN (platelets_max > platelets_ref_range_upper) AND (platelets_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN platelets_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS platelets_max_status\
 , wbc_min , CASE WHEN (wbc_min BETWEEN wbc_ref_range_lower AND wbc_ref_range_upper) AND (wbc_ref_range_lower IS NOT NULL) AND (wbc_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (wbc_min< wbc_ref_range_lower) AND (wbc_ref_range_lower IS NOT NULL) THEN 'low' WHEN (wbc_min > wbc_ref_range_upper) AND (wbc_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN wbc_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS wbc_min_status , wbc_max , CASE WHEN (wbc_max BETWEEN wbc_ref_range_lower AND wbc_ref_range_upper) AND (wbc_ref_range_lower IS NOT NULL) AND (wbc_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (wbc_max< wbc_ref_range_lower) AND (wbc_ref_range_lower IS NOT NULL) THEN 'low' WHEN (wbc_max > wbc_ref_range_upper) AND (wbc_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN wbc_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS wbc_max_status\
 , albumin_min , CASE WHEN (albumin_min BETWEEN albumin_ref_range_lower AND albumin_ref_range_upper) AND (albumin_ref_range_lower IS NOT NULL) AND (albumin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (albumin_min< albumin_ref_range_lower) AND (albumin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (albumin_min > albumin_ref_range_upper) AND (albumin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN albumin_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS albumin_min_status , albumin_max , CASE WHEN (albumin_max BETWEEN albumin_ref_range_lower AND albumin_ref_range_upper) AND (albumin_ref_range_lower IS NOT NULL) AND (albumin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (albumin_max< albumin_ref_range_lower) AND (albumin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (albumin_max > albumin_ref_range_upper) AND (albumin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN albumin_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS albumin_max_status\
 , globulin_min , CASE WHEN (globulin_min BETWEEN globulin_ref_range_lower AND globulin_ref_range_upper) AND (globulin_ref_range_lower IS NOT NULL) AND (globulin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (globulin_min< globulin_ref_range_lower) AND (globulin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (globulin_min > globulin_ref_range_upper) AND (globulin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN globulin_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS globulin_min_status , globulin_max , CASE WHEN (globulin_max BETWEEN globulin_ref_range_lower AND globulin_ref_range_upper) AND (globulin_ref_range_lower IS NOT NULL) AND (globulin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (globulin_max< globulin_ref_range_lower) AND (globulin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (globulin_max > globulin_ref_range_upper) AND (globulin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN globulin_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS globulin_max_status\
 , total_protein_min , CASE WHEN (total_protein_min BETWEEN total_protein_ref_range_lower AND total_protein_ref_range_upper) AND (total_protein_ref_range_lower IS NOT NULL) AND (total_protein_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (total_protein_min< total_protein_ref_range_lower) AND (total_protein_ref_range_lower IS NOT NULL) THEN 'low' WHEN (total_protein_min > total_protein_ref_range_upper) AND (total_protein_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN total_protein_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS total_protein_min_status , total_protein_max , CASE WHEN (total_protein_max BETWEEN total_protein_ref_range_lower AND total_protein_ref_range_upper) AND (total_protein_ref_range_lower IS NOT NULL) AND (total_protein_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (total_protein_max< total_protein_ref_range_lower) AND (total_protein_ref_range_lower IS NOT NULL) THEN 'low' WHEN (total_protein_max > total_protein_ref_range_upper) AND (total_protein_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN total_protein_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS total_protein_max_status\
 , aniongap_min , CASE WHEN (aniongap_min BETWEEN aniongap_ref_range_lower AND aniongap_ref_range_upper) AND (aniongap_ref_range_lower IS NOT NULL) AND (aniongap_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (aniongap_min< aniongap_ref_range_lower) AND (aniongap_ref_range_lower IS NOT NULL) THEN 'low' WHEN (aniongap_min > aniongap_ref_range_upper) AND (aniongap_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN aniongap_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS aniongap_min_status , aniongap_max , CASE WHEN (aniongap_max BETWEEN aniongap_ref_range_lower AND aniongap_ref_range_upper) AND (aniongap_ref_range_lower IS NOT NULL) AND (aniongap_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (aniongap_max< aniongap_ref_range_lower) AND (aniongap_ref_range_lower IS NOT NULL) THEN 'low' WHEN (aniongap_max > aniongap_ref_range_upper) AND (aniongap_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN aniongap_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS aniongap_max_status\
 , bun_min , CASE WHEN (bun_min BETWEEN bun_ref_range_lower AND bun_ref_range_upper) AND (bun_ref_range_lower IS NOT NULL) AND (bun_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bun_min< bun_ref_range_lower) AND (bun_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bun_min > bun_ref_range_upper) AND (bun_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bun_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bun_min_status , bun_max , CASE WHEN (bun_max BETWEEN bun_ref_range_lower AND bun_ref_range_upper) AND (bun_ref_range_lower IS NOT NULL) AND (bun_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bun_max< bun_ref_range_lower) AND (bun_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bun_max > bun_ref_range_upper) AND (bun_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bun_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bun_max_status\
 , creatinine_min , CASE WHEN (creatinine_min BETWEEN creatinine_ref_range_lower AND creatinine_ref_range_upper) AND (creatinine_ref_range_lower IS NOT NULL) AND (creatinine_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (creatinine_min< creatinine_ref_range_lower) AND (creatinine_ref_range_lower IS NOT NULL) THEN 'low' WHEN (creatinine_min > creatinine_ref_range_upper) AND (creatinine_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN creatinine_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS creatinine_min_status , creatinine_max , CASE WHEN (creatinine_max BETWEEN creatinine_ref_range_lower AND creatinine_ref_range_upper) AND (creatinine_ref_range_lower IS NOT NULL) AND (creatinine_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (creatinine_max< creatinine_ref_range_lower) AND (creatinine_ref_range_lower IS NOT NULL) THEN 'low' WHEN (creatinine_max > creatinine_ref_range_upper) AND (creatinine_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN creatinine_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS creatinine_max_status\
 , abs_basophils_min , CASE WHEN (abs_basophils_min BETWEEN abs_basophils_ref_range_lower AND abs_basophils_ref_range_upper) AND (abs_basophils_ref_range_lower IS NOT NULL) AND (abs_basophils_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_basophils_min< abs_basophils_ref_range_lower) AND (abs_basophils_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_basophils_min > abs_basophils_ref_range_upper) AND (abs_basophils_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_basophils_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_basophils_min_status , abs_basophils_max , CASE WHEN (abs_basophils_max BETWEEN abs_basophils_ref_range_lower AND abs_basophils_ref_range_upper) AND (abs_basophils_ref_range_lower IS NOT NULL) AND (abs_basophils_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_basophils_max< abs_basophils_ref_range_lower) AND (abs_basophils_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_basophils_max > abs_basophils_ref_range_upper) AND (abs_basophils_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_basophils_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_basophils_max_status\
 , abs_eosinophils_min , CASE WHEN (abs_eosinophils_min BETWEEN abs_eosinophils_ref_range_lower AND abs_eosinophils_ref_range_upper) AND (abs_eosinophils_ref_range_lower IS NOT NULL) AND (abs_eosinophils_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_eosinophils_min< abs_eosinophils_ref_range_lower) AND (abs_eosinophils_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_eosinophils_min > abs_eosinophils_ref_range_upper) AND (abs_eosinophils_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_eosinophils_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_eosinophils_min_status , abs_eosinophils_max , CASE WHEN (abs_eosinophils_max BETWEEN abs_eosinophils_ref_range_lower AND abs_eosinophils_ref_range_upper) AND (abs_eosinophils_ref_range_lower IS NOT NULL) AND (abs_eosinophils_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_eosinophils_max< abs_eosinophils_ref_range_lower) AND (abs_eosinophils_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_eosinophils_max > abs_eosinophils_ref_range_upper) AND (abs_eosinophils_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_eosinophils_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_eosinophils_max_status\
 , abs_lymphocytes_min , CASE WHEN (abs_lymphocytes_min BETWEEN abs_lymphocytes_ref_range_lower AND abs_lymphocytes_ref_range_upper) AND (abs_lymphocytes_ref_range_lower IS NOT NULL) AND (abs_lymphocytes_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_lymphocytes_min< abs_lymphocytes_ref_range_lower) AND (abs_lymphocytes_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_lymphocytes_min > abs_lymphocytes_ref_range_upper) AND (abs_lymphocytes_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_lymphocytes_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_lymphocytes_min_status , abs_lymphocytes_max , CASE WHEN (abs_lymphocytes_max BETWEEN abs_lymphocytes_ref_range_lower AND abs_lymphocytes_ref_range_upper) AND (abs_lymphocytes_ref_range_lower IS NOT NULL) AND (abs_lymphocytes_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_lymphocytes_max< abs_lymphocytes_ref_range_lower) AND (abs_lymphocytes_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_lymphocytes_max > abs_lymphocytes_ref_range_upper) AND (abs_lymphocytes_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_lymphocytes_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_lymphocytes_max_status\
 , abs_monocytes_min , CASE WHEN (abs_monocytes_min BETWEEN abs_monocytes_ref_range_lower AND abs_monocytes_ref_range_upper) AND (abs_monocytes_ref_range_lower IS NOT NULL) AND (abs_monocytes_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_monocytes_min< abs_monocytes_ref_range_lower) AND (abs_monocytes_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_monocytes_min > abs_monocytes_ref_range_upper) AND (abs_monocytes_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_monocytes_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_monocytes_min_status , abs_monocytes_max , CASE WHEN (abs_monocytes_max BETWEEN abs_monocytes_ref_range_lower AND abs_monocytes_ref_range_upper) AND (abs_monocytes_ref_range_lower IS NOT NULL) AND (abs_monocytes_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_monocytes_max< abs_monocytes_ref_range_lower) AND (abs_monocytes_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_monocytes_max > abs_monocytes_ref_range_upper) AND (abs_monocytes_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_monocytes_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_monocytes_max_status\
 , abs_neutrophils_min , CASE WHEN (abs_neutrophils_min BETWEEN abs_neutrophils_ref_range_lower AND abs_neutrophils_ref_range_upper) AND (abs_neutrophils_ref_range_lower IS NOT NULL) AND (abs_neutrophils_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_neutrophils_min< abs_neutrophils_ref_range_lower) AND (abs_neutrophils_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_neutrophils_min > abs_neutrophils_ref_range_upper) AND (abs_neutrophils_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_neutrophils_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_neutrophils_min_status , abs_neutrophils_max , CASE WHEN (abs_neutrophils_max BETWEEN abs_neutrophils_ref_range_lower AND abs_neutrophils_ref_range_upper) AND (abs_neutrophils_ref_range_lower IS NOT NULL) AND (abs_neutrophils_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (abs_neutrophils_max< abs_neutrophils_ref_range_lower) AND (abs_neutrophils_ref_range_lower IS NOT NULL) THEN 'low' WHEN (abs_neutrophils_max > abs_neutrophils_ref_range_upper) AND (abs_neutrophils_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN abs_neutrophils_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS abs_neutrophils_max_status\
 , atyps_min , CASE WHEN (atyps_min BETWEEN atyps_ref_range_lower AND atyps_ref_range_upper) AND (atyps_ref_range_lower IS NOT NULL) AND (atyps_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (atyps_min< atyps_ref_range_lower) AND (atyps_ref_range_lower IS NOT NULL) THEN 'low' WHEN (atyps_min > atyps_ref_range_upper) AND (atyps_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN atyps_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS atyps_min_status , atyps_max , CASE WHEN (atyps_max BETWEEN atyps_ref_range_lower AND atyps_ref_range_upper) AND (atyps_ref_range_lower IS NOT NULL) AND (atyps_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (atyps_max< atyps_ref_range_lower) AND (atyps_ref_range_lower IS NOT NULL) THEN 'low' WHEN (atyps_max > atyps_ref_range_upper) AND (atyps_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN atyps_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS atyps_max_status\
 , bands_min , CASE WHEN (bands_min BETWEEN bands_ref_range_lower AND bands_ref_range_upper) AND (bands_ref_range_lower IS NOT NULL) AND (bands_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bands_min< bands_ref_range_lower) AND (bands_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bands_min > bands_ref_range_upper) AND (bands_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bands_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bands_min_status , bands_max , CASE WHEN (bands_max BETWEEN bands_ref_range_lower AND bands_ref_range_upper) AND (bands_ref_range_lower IS NOT NULL) AND (bands_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bands_max< bands_ref_range_lower) AND (bands_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bands_max > bands_ref_range_upper) AND (bands_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bands_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bands_max_status\
 , imm_granulocytes_min , CASE WHEN (imm_granulocytes_min BETWEEN imm_granulocytes_ref_range_lower AND imm_granulocytes_ref_range_upper) AND (imm_granulocytes_ref_range_lower IS NOT NULL) AND (imm_granulocytes_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (imm_granulocytes_min< imm_granulocytes_ref_range_lower) AND (imm_granulocytes_ref_range_lower IS NOT NULL) THEN 'low' WHEN (imm_granulocytes_min > imm_granulocytes_ref_range_upper) AND (imm_granulocytes_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN imm_granulocytes_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS imm_granulocytes_min_status , imm_granulocytes_max , CASE WHEN (imm_granulocytes_max BETWEEN imm_granulocytes_ref_range_lower AND imm_granulocytes_ref_range_upper) AND (imm_granulocytes_ref_range_lower IS NOT NULL) AND (imm_granulocytes_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (imm_granulocytes_max< imm_granulocytes_ref_range_lower) AND (imm_granulocytes_ref_range_lower IS NOT NULL) THEN 'low' WHEN (imm_granulocytes_max > imm_granulocytes_ref_range_upper) AND (imm_granulocytes_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN imm_granulocytes_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS imm_granulocytes_max_status\
 , metas_min , CASE WHEN (metas_min BETWEEN metas_ref_range_lower AND metas_ref_range_upper) AND (metas_ref_range_lower IS NOT NULL) AND (metas_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (metas_min< metas_ref_range_lower) AND (metas_ref_range_lower IS NOT NULL) THEN 'low' WHEN (metas_min > metas_ref_range_upper) AND (metas_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN metas_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS metas_min_status , metas_max , CASE WHEN (metas_max BETWEEN metas_ref_range_lower AND metas_ref_range_upper) AND (metas_ref_range_lower IS NOT NULL) AND (metas_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (metas_max< metas_ref_range_lower) AND (metas_ref_range_lower IS NOT NULL) THEN 'low' WHEN (metas_max > metas_ref_range_upper) AND (metas_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN metas_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS metas_max_status\
 , nrbc_min , CASE WHEN (nrbc_min BETWEEN nrbc_ref_range_lower AND nrbc_ref_range_upper) AND (nrbc_ref_range_lower IS NOT NULL) AND (nrbc_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (nrbc_min< nrbc_ref_range_lower) AND (nrbc_ref_range_lower IS NOT NULL) THEN 'low' WHEN (nrbc_min > nrbc_ref_range_upper) AND (nrbc_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN nrbc_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS nrbc_min_status , nrbc_max , CASE WHEN (nrbc_max BETWEEN nrbc_ref_range_lower AND nrbc_ref_range_upper) AND (nrbc_ref_range_lower IS NOT NULL) AND (nrbc_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (nrbc_max< nrbc_ref_range_lower) AND (nrbc_ref_range_lower IS NOT NULL) THEN 'low' WHEN (nrbc_max > nrbc_ref_range_upper) AND (nrbc_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN nrbc_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS nrbc_max_status\
 , d_dimer_min , CASE WHEN (d_dimer_min BETWEEN d_dimer_ref_range_lower AND d_dimer_ref_range_upper) AND (d_dimer_ref_range_lower IS NOT NULL) AND (d_dimer_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (d_dimer_min< d_dimer_ref_range_lower) AND (d_dimer_ref_range_lower IS NOT NULL) THEN 'low' WHEN (d_dimer_min > d_dimer_ref_range_upper) AND (d_dimer_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN d_dimer_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS d_dimer_min_status , d_dimer_max , CASE WHEN (d_dimer_max BETWEEN d_dimer_ref_range_lower AND d_dimer_ref_range_upper) AND (d_dimer_ref_range_lower IS NOT NULL) AND (d_dimer_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (d_dimer_max< d_dimer_ref_range_lower) AND (d_dimer_ref_range_lower IS NOT NULL) THEN 'low' WHEN (d_dimer_max > d_dimer_ref_range_upper) AND (d_dimer_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN d_dimer_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS d_dimer_max_status\
 , fibrinogen_min , CASE WHEN (fibrinogen_min BETWEEN fibrinogen_ref_range_lower AND fibrinogen_ref_range_upper) AND (fibrinogen_ref_range_lower IS NOT NULL) AND (fibrinogen_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (fibrinogen_min< fibrinogen_ref_range_lower) AND (fibrinogen_ref_range_lower IS NOT NULL) THEN 'low' WHEN (fibrinogen_min > fibrinogen_ref_range_upper) AND (fibrinogen_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN fibrinogen_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS fibrinogen_min_status , fibrinogen_max , CASE WHEN (fibrinogen_max BETWEEN fibrinogen_ref_range_lower AND fibrinogen_ref_range_upper) AND (fibrinogen_ref_range_lower IS NOT NULL) AND (fibrinogen_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (fibrinogen_max< fibrinogen_ref_range_lower) AND (fibrinogen_ref_range_lower IS NOT NULL) THEN 'low' WHEN (fibrinogen_max > fibrinogen_ref_range_upper) AND (fibrinogen_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN fibrinogen_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS fibrinogen_max_status\
 , thrombin_min , CASE WHEN (thrombin_min BETWEEN thrombin_ref_range_lower AND thrombin_ref_range_upper) AND (thrombin_ref_range_lower IS NOT NULL) AND (thrombin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (thrombin_min< thrombin_ref_range_lower) AND (thrombin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (thrombin_min > thrombin_ref_range_upper) AND (thrombin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN thrombin_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS thrombin_min_status , thrombin_max , CASE WHEN (thrombin_max BETWEEN thrombin_ref_range_lower AND thrombin_ref_range_upper) AND (thrombin_ref_range_lower IS NOT NULL) AND (thrombin_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (thrombin_max< thrombin_ref_range_lower) AND (thrombin_ref_range_lower IS NOT NULL) THEN 'low' WHEN (thrombin_max > thrombin_ref_range_upper) AND (thrombin_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN thrombin_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS thrombin_max_status\
 , inr_min , CASE WHEN (inr_min BETWEEN inr_ref_range_lower AND inr_ref_range_upper) AND (inr_ref_range_lower IS NOT NULL) AND (inr_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (inr_min< inr_ref_range_lower) AND (inr_ref_range_lower IS NOT NULL) THEN 'low' WHEN (inr_min > inr_ref_range_upper) AND (inr_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN inr_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS inr_min_status , inr_max , CASE WHEN (inr_max BETWEEN inr_ref_range_lower AND inr_ref_range_upper) AND (inr_ref_range_lower IS NOT NULL) AND (inr_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (inr_max< inr_ref_range_lower) AND (inr_ref_range_lower IS NOT NULL) THEN 'low' WHEN (inr_max > inr_ref_range_upper) AND (inr_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN inr_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS inr_max_status\
 , pt_min , CASE WHEN (pt_min BETWEEN pt_ref_range_lower AND pt_ref_range_upper) AND (pt_ref_range_lower IS NOT NULL) AND (pt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (pt_min< pt_ref_range_lower) AND (pt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (pt_min > pt_ref_range_upper) AND (pt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN pt_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS pt_min_status , pt_max , CASE WHEN (pt_max BETWEEN pt_ref_range_lower AND pt_ref_range_upper) AND (pt_ref_range_lower IS NOT NULL) AND (pt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (pt_max< pt_ref_range_lower) AND (pt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (pt_max > pt_ref_range_upper) AND (pt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN pt_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS pt_max_status\
 , ptt_min , CASE WHEN (ptt_min BETWEEN ptt_ref_range_lower AND ptt_ref_range_upper) AND (ptt_ref_range_lower IS NOT NULL) AND (ptt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ptt_min< ptt_ref_range_lower) AND (ptt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ptt_min > ptt_ref_range_upper) AND (ptt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ptt_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ptt_min_status , ptt_max , CASE WHEN (ptt_max BETWEEN ptt_ref_range_lower AND ptt_ref_range_upper) AND (ptt_ref_range_lower IS NOT NULL) AND (ptt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ptt_max< ptt_ref_range_lower) AND (ptt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ptt_max > ptt_ref_range_upper) AND (ptt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ptt_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ptt_max_status\
 , alt_min , CASE WHEN (alt_min BETWEEN alt_ref_range_lower AND alt_ref_range_upper) AND (alt_ref_range_lower IS NOT NULL) AND (alt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (alt_min< alt_ref_range_lower) AND (alt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (alt_min > alt_ref_range_upper) AND (alt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN alt_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS alt_min_status , alt_max , CASE WHEN (alt_max BETWEEN alt_ref_range_lower AND alt_ref_range_upper) AND (alt_ref_range_lower IS NOT NULL) AND (alt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (alt_max< alt_ref_range_lower) AND (alt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (alt_max > alt_ref_range_upper) AND (alt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN alt_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS alt_max_status\
 , alp_min , CASE WHEN (alp_min BETWEEN alp_ref_range_lower AND alp_ref_range_upper) AND (alp_ref_range_lower IS NOT NULL) AND (alp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (alp_min< alp_ref_range_lower) AND (alp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (alp_min > alp_ref_range_upper) AND (alp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN alp_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS alp_min_status , alp_max , CASE WHEN (alp_max BETWEEN alp_ref_range_lower AND alp_ref_range_upper) AND (alp_ref_range_lower IS NOT NULL) AND (alp_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (alp_max< alp_ref_range_lower) AND (alp_ref_range_lower IS NOT NULL) THEN 'low' WHEN (alp_max > alp_ref_range_upper) AND (alp_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN alp_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS alp_max_status\
 , ast_min , CASE WHEN (ast_min BETWEEN ast_ref_range_lower AND ast_ref_range_upper) AND (ast_ref_range_lower IS NOT NULL) AND (ast_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ast_min< ast_ref_range_lower) AND (ast_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ast_min > ast_ref_range_upper) AND (ast_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ast_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ast_min_status , ast_max , CASE WHEN (ast_max BETWEEN ast_ref_range_lower AND ast_ref_range_upper) AND (ast_ref_range_lower IS NOT NULL) AND (ast_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ast_max< ast_ref_range_lower) AND (ast_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ast_max > ast_ref_range_upper) AND (ast_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ast_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ast_max_status\
 , amylase_min , CASE WHEN (amylase_min BETWEEN amylase_ref_range_lower AND amylase_ref_range_upper) AND (amylase_ref_range_lower IS NOT NULL) AND (amylase_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (amylase_min< amylase_ref_range_lower) AND (amylase_ref_range_lower IS NOT NULL) THEN 'low' WHEN (amylase_min > amylase_ref_range_upper) AND (amylase_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN amylase_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS amylase_min_status , amylase_max , CASE WHEN (amylase_max BETWEEN amylase_ref_range_lower AND amylase_ref_range_upper) AND (amylase_ref_range_lower IS NOT NULL) AND (amylase_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (amylase_max< amylase_ref_range_lower) AND (amylase_ref_range_lower IS NOT NULL) THEN 'low' WHEN (amylase_max > amylase_ref_range_upper) AND (amylase_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN amylase_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS amylase_max_status\
 , bilirubin_total_min , CASE WHEN (bilirubin_total_min BETWEEN bilirubin_total_ref_range_lower AND bilirubin_total_ref_range_upper) AND (bilirubin_total_ref_range_lower IS NOT NULL) AND (bilirubin_total_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bilirubin_total_min< bilirubin_total_ref_range_lower) AND (bilirubin_total_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bilirubin_total_min > bilirubin_total_ref_range_upper) AND (bilirubin_total_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bilirubin_total_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bilirubin_total_min_status , bilirubin_total_max , CASE WHEN (bilirubin_total_max BETWEEN bilirubin_total_ref_range_lower AND bilirubin_total_ref_range_upper) AND (bilirubin_total_ref_range_lower IS NOT NULL) AND (bilirubin_total_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bilirubin_total_max< bilirubin_total_ref_range_lower) AND (bilirubin_total_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bilirubin_total_max > bilirubin_total_ref_range_upper) AND (bilirubin_total_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bilirubin_total_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bilirubin_total_max_status\
 , bilirubin_direct_min , CASE WHEN (bilirubin_direct_min BETWEEN bilirubin_direct_ref_range_lower AND bilirubin_direct_ref_range_upper) AND (bilirubin_direct_ref_range_lower IS NOT NULL) AND (bilirubin_direct_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bilirubin_direct_min< bilirubin_direct_ref_range_lower) AND (bilirubin_direct_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bilirubin_direct_min > bilirubin_direct_ref_range_upper) AND (bilirubin_direct_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bilirubin_direct_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bilirubin_direct_min_status , bilirubin_direct_max , CASE WHEN (bilirubin_direct_max BETWEEN bilirubin_direct_ref_range_lower AND bilirubin_direct_ref_range_upper) AND (bilirubin_direct_ref_range_lower IS NOT NULL) AND (bilirubin_direct_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bilirubin_direct_max< bilirubin_direct_ref_range_lower) AND (bilirubin_direct_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bilirubin_direct_max > bilirubin_direct_ref_range_upper) AND (bilirubin_direct_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bilirubin_direct_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bilirubin_direct_max_status\
 , bilirubin_indirect_min , CASE WHEN (bilirubin_indirect_min BETWEEN bilirubin_indirect_ref_range_lower AND bilirubin_indirect_ref_range_upper) AND (bilirubin_indirect_ref_range_lower IS NOT NULL) AND (bilirubin_indirect_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bilirubin_indirect_min< bilirubin_indirect_ref_range_lower) AND (bilirubin_indirect_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bilirubin_indirect_min > bilirubin_indirect_ref_range_upper) AND (bilirubin_indirect_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bilirubin_indirect_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bilirubin_indirect_min_status , bilirubin_indirect_max , CASE WHEN (bilirubin_indirect_max BETWEEN bilirubin_indirect_ref_range_lower AND bilirubin_indirect_ref_range_upper) AND (bilirubin_indirect_ref_range_lower IS NOT NULL) AND (bilirubin_indirect_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (bilirubin_indirect_max< bilirubin_indirect_ref_range_lower) AND (bilirubin_indirect_ref_range_lower IS NOT NULL) THEN 'low' WHEN (bilirubin_indirect_max > bilirubin_indirect_ref_range_upper) AND (bilirubin_indirect_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN bilirubin_indirect_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS bilirubin_indirect_max_status\
 , ck_cpk_min , CASE WHEN (ck_cpk_min BETWEEN ck_cpk_ref_range_lower AND ck_cpk_ref_range_upper) AND (ck_cpk_ref_range_lower IS NOT NULL) AND (ck_cpk_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ck_cpk_min< ck_cpk_ref_range_lower) AND (ck_cpk_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ck_cpk_min > ck_cpk_ref_range_upper) AND (ck_cpk_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ck_cpk_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ck_cpk_min_status , ck_cpk_max , CASE WHEN (ck_cpk_max BETWEEN ck_cpk_ref_range_lower AND ck_cpk_ref_range_upper) AND (ck_cpk_ref_range_lower IS NOT NULL) AND (ck_cpk_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ck_cpk_max< ck_cpk_ref_range_lower) AND (ck_cpk_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ck_cpk_max > ck_cpk_ref_range_upper) AND (ck_cpk_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ck_cpk_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ck_cpk_max_status\
 , ck_mb_min , CASE WHEN (ck_mb_min BETWEEN ck_mb_ref_range_lower AND ck_mb_ref_range_upper) AND (ck_mb_ref_range_lower IS NOT NULL) AND (ck_mb_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ck_mb_min< ck_mb_ref_range_lower) AND (ck_mb_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ck_mb_min > ck_mb_ref_range_upper) AND (ck_mb_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ck_mb_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ck_mb_min_status , ck_mb_max , CASE WHEN (ck_mb_max BETWEEN ck_mb_ref_range_lower AND ck_mb_ref_range_upper) AND (ck_mb_ref_range_lower IS NOT NULL) AND (ck_mb_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ck_mb_max< ck_mb_ref_range_lower) AND (ck_mb_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ck_mb_max > ck_mb_ref_range_upper) AND (ck_mb_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ck_mb_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ck_mb_max_status\
 , ggt_min , CASE WHEN (ggt_min BETWEEN ggt_ref_range_lower AND ggt_ref_range_upper) AND (ggt_ref_range_lower IS NOT NULL) AND (ggt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ggt_min< ggt_ref_range_lower) AND (ggt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ggt_min > ggt_ref_range_upper) AND (ggt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ggt_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ggt_min_status , ggt_max , CASE WHEN (ggt_max BETWEEN ggt_ref_range_lower AND ggt_ref_range_upper) AND (ggt_ref_range_lower IS NOT NULL) AND (ggt_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ggt_max< ggt_ref_range_lower) AND (ggt_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ggt_max > ggt_ref_range_upper) AND (ggt_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ggt_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ggt_max_status\
 , ld_ldh_min , CASE WHEN (ld_ldh_min BETWEEN ld_ldh_ref_range_lower AND ld_ldh_ref_range_upper) AND (ld_ldh_ref_range_lower IS NOT NULL) AND (ld_ldh_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ld_ldh_min< ld_ldh_ref_range_lower) AND (ld_ldh_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ld_ldh_min > ld_ldh_ref_range_upper) AND (ld_ldh_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ld_ldh_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ld_ldh_min_status , ld_ldh_max , CASE WHEN (ld_ldh_max BETWEEN ld_ldh_ref_range_lower AND ld_ldh_ref_range_upper) AND (ld_ldh_ref_range_lower IS NOT NULL) AND (ld_ldh_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (ld_ldh_max< ld_ldh_ref_range_lower) AND (ld_ldh_ref_range_lower IS NOT NULL) THEN 'low' WHEN (ld_ldh_max > ld_ldh_ref_range_upper) AND (ld_ldh_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN ld_ldh_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS ld_ldh_max_status\
\
--parameters which were in bg\
\
 , so2_bg_min , CASE WHEN (so2_bg_min BETWEEN so2_bg_ref_range_lower AND so2_bg_ref_range_upper) AND (so2_bg_ref_range_lower IS NOT NULL) AND (so2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (so2_bg_min< so2_bg_ref_range_lower) AND (so2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (so2_bg_min > so2_bg_ref_range_upper) AND (so2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN so2_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS so2_bg_min_status , so2_bg_max , CASE WHEN (so2_bg_max BETWEEN so2_bg_ref_range_lower AND so2_bg_ref_range_upper) AND (so2_bg_ref_range_lower IS NOT NULL) AND (so2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (so2_bg_max< so2_bg_ref_range_lower) AND (so2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (so2_bg_max > so2_bg_ref_range_upper) AND (so2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN so2_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS so2_bg_max_status\
 , po2_bg_min , CASE WHEN (po2_bg_min BETWEEN po2_bg_ref_range_lower AND po2_bg_ref_range_upper) AND (po2_bg_ref_range_lower IS NOT NULL) AND (po2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (po2_bg_min< po2_bg_ref_range_lower) AND (po2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (po2_bg_min > po2_bg_ref_range_upper) AND (po2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN po2_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS po2_bg_min_status , po2_bg_max , CASE WHEN (po2_bg_max BETWEEN po2_bg_ref_range_lower AND po2_bg_ref_range_upper) AND (po2_bg_ref_range_lower IS NOT NULL) AND (po2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (po2_bg_max< po2_bg_ref_range_lower) AND (po2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (po2_bg_max > po2_bg_ref_range_upper) AND (po2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN po2_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS po2_bg_max_status\
 , pco2_bg_min , CASE WHEN (pco2_bg_min BETWEEN pco2_bg_ref_range_lower AND pco2_bg_ref_range_upper) AND (pco2_bg_ref_range_lower IS NOT NULL) AND (pco2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (pco2_bg_min< pco2_bg_ref_range_lower) AND (pco2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (pco2_bg_min > pco2_bg_ref_range_upper) AND (pco2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN pco2_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS pco2_bg_min_status , pco2_bg_max , CASE WHEN (pco2_bg_max BETWEEN pco2_bg_ref_range_lower AND pco2_bg_ref_range_upper) AND (pco2_bg_ref_range_lower IS NOT NULL) AND (pco2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (pco2_bg_max< pco2_bg_ref_range_lower) AND (pco2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (pco2_bg_max > pco2_bg_ref_range_upper) AND (pco2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN pco2_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS pco2_bg_max_status\
 , aado2_bg_min , CASE WHEN (aado2_bg_min BETWEEN aado2_bg_ref_range_lower AND aado2_bg_ref_range_upper) AND (aado2_bg_ref_range_lower IS NOT NULL) AND (aado2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (aado2_bg_min< aado2_bg_ref_range_lower) AND (aado2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (aado2_bg_min > aado2_bg_ref_range_upper) AND (aado2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN aado2_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS aado2_bg_min_status , aado2_bg_max , CASE WHEN (aado2_bg_max BETWEEN aado2_bg_ref_range_lower AND aado2_bg_ref_range_upper) AND (aado2_bg_ref_range_lower IS NOT NULL) AND (aado2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (aado2_bg_max< aado2_bg_ref_range_lower) AND (aado2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (aado2_bg_max > aado2_bg_ref_range_upper) AND (aado2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN aado2_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS aado2_bg_max_status\
 , fio2_bg_min , CASE WHEN (fio2_bg_min BETWEEN fio2_bg_ref_range_lower AND fio2_bg_ref_range_upper) AND (fio2_bg_ref_range_lower IS NOT NULL) AND (fio2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (fio2_bg_min< fio2_bg_ref_range_lower) AND (fio2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (fio2_bg_min > fio2_bg_ref_range_upper) AND (fio2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN fio2_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS fio2_bg_min_status , fio2_bg_max , CASE WHEN (fio2_bg_max BETWEEN fio2_bg_ref_range_lower AND fio2_bg_ref_range_upper) AND (fio2_bg_ref_range_lower IS NOT NULL) AND (fio2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (fio2_bg_max< fio2_bg_ref_range_lower) AND (fio2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (fio2_bg_max > fio2_bg_ref_range_upper) AND (fio2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN fio2_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS fio2_bg_max_status\
 , totalco2_bg_min , CASE WHEN (totalco2_bg_min BETWEEN totalco2_bg_ref_range_lower AND totalco2_bg_ref_range_upper) AND (totalco2_bg_ref_range_lower IS NOT NULL) AND (totalco2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (totalco2_bg_min< totalco2_bg_ref_range_lower) AND (totalco2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (totalco2_bg_min > totalco2_bg_ref_range_upper) AND (totalco2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN totalco2_bg_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS totalco2_bg_min_status , totalco2_bg_max , CASE WHEN (totalco2_bg_max BETWEEN totalco2_bg_ref_range_lower AND totalco2_bg_ref_range_upper) AND (totalco2_bg_ref_range_lower IS NOT NULL) AND (totalco2_bg_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (totalco2_bg_max< totalco2_bg_ref_range_lower) AND (totalco2_bg_ref_range_lower IS NOT NULL) THEN 'low' WHEN (totalco2_bg_max > totalco2_bg_ref_range_upper) AND (totalco2_bg_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN totalco2_bg_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS totalco2_bg_max_status\
\
--parameters which were in bg_art\
\
 , so2_bg_art_min , CASE WHEN (so2_bg_art_min BETWEEN so2_bg_art_ref_range_lower AND so2_bg_art_ref_range_upper) AND (so2_bg_art_ref_range_lower IS NOT NULL) AND (so2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (so2_bg_art_min< so2_bg_art_ref_range_lower) AND (so2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (so2_bg_art_min > so2_bg_art_ref_range_upper) AND (so2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN so2_bg_art_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS so2_bg_art_min_status , so2_bg_art_max , CASE WHEN (so2_bg_art_max BETWEEN so2_bg_art_ref_range_lower AND so2_bg_art_ref_range_upper) AND (so2_bg_art_ref_range_lower IS NOT NULL) AND (so2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (so2_bg_art_max< so2_bg_art_ref_range_lower) AND (so2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (so2_bg_art_max > so2_bg_art_ref_range_upper) AND (so2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN so2_bg_art_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS so2_bg_art_max_status\
 , po2_bg_art_min , CASE WHEN (po2_bg_art_min BETWEEN po2_bg_art_ref_range_lower AND po2_bg_art_ref_range_upper) AND (po2_bg_art_ref_range_lower IS NOT NULL) AND (po2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (po2_bg_art_min< po2_bg_art_ref_range_lower) AND (po2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (po2_bg_art_min > po2_bg_art_ref_range_upper) AND (po2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN po2_bg_art_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS po2_bg_art_min_status , po2_bg_art_max , CASE WHEN (po2_bg_art_max BETWEEN po2_bg_art_ref_range_lower AND po2_bg_art_ref_range_upper) AND (po2_bg_art_ref_range_lower IS NOT NULL) AND (po2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (po2_bg_art_max< po2_bg_art_ref_range_lower) AND (po2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (po2_bg_art_max > po2_bg_art_ref_range_upper) AND (po2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN po2_bg_art_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS po2_bg_art_max_status\
 , pco2_bg_art_min , CASE WHEN (pco2_bg_art_min BETWEEN pco2_bg_art_ref_range_lower AND pco2_bg_art_ref_range_upper) AND (pco2_bg_art_ref_range_lower IS NOT NULL) AND (pco2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (pco2_bg_art_min< pco2_bg_art_ref_range_lower) AND (pco2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (pco2_bg_art_min > pco2_bg_art_ref_range_upper) AND (pco2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN pco2_bg_art_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS pco2_bg_art_min_status , pco2_bg_art_max , CASE WHEN (pco2_bg_art_max BETWEEN pco2_bg_art_ref_range_lower AND pco2_bg_art_ref_range_upper) AND (pco2_bg_art_ref_range_lower IS NOT NULL) AND (pco2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (pco2_bg_art_max< pco2_bg_art_ref_range_lower) AND (pco2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (pco2_bg_art_max > pco2_bg_art_ref_range_upper) AND (pco2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN pco2_bg_art_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS pco2_bg_art_max_status\
 , aado2_bg_art_min , CASE WHEN (aado2_bg_art_min BETWEEN aado2_bg_art_ref_range_lower AND aado2_bg_art_ref_range_upper) AND (aado2_bg_art_ref_range_lower IS NOT NULL) AND (aado2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (aado2_bg_art_min< aado2_bg_art_ref_range_lower) AND (aado2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (aado2_bg_art_min > aado2_bg_art_ref_range_upper) AND (aado2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN aado2_bg_art_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS aado2_bg_art_min_status , aado2_bg_art_max , CASE WHEN (aado2_bg_art_max BETWEEN aado2_bg_art_ref_range_lower AND aado2_bg_art_ref_range_upper) AND (aado2_bg_art_ref_range_lower IS NOT NULL) AND (aado2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (aado2_bg_art_max< aado2_bg_art_ref_range_lower) AND (aado2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (aado2_bg_art_max > aado2_bg_art_ref_range_upper) AND (aado2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN aado2_bg_art_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS aado2_bg_art_max_status\
 , fio2_bg_art_min , CASE WHEN (fio2_bg_art_min BETWEEN fio2_bg_art_ref_range_lower AND fio2_bg_art_ref_range_upper) AND (fio2_bg_art_ref_range_lower IS NOT NULL) AND (fio2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (fio2_bg_art_min< fio2_bg_art_ref_range_lower) AND (fio2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (fio2_bg_art_min > fio2_bg_art_ref_range_upper) AND (fio2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN fio2_bg_art_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS fio2_bg_art_min_status , fio2_bg_art_max , CASE WHEN (fio2_bg_art_max BETWEEN fio2_bg_art_ref_range_lower AND fio2_bg_art_ref_range_upper) AND (fio2_bg_art_ref_range_lower IS NOT NULL) AND (fio2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (fio2_bg_art_max< fio2_bg_art_ref_range_lower) AND (fio2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (fio2_bg_art_max > fio2_bg_art_ref_range_upper) AND (fio2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN fio2_bg_art_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS fio2_bg_art_max_status\
 , totalco2_bg_art_min , CASE WHEN (totalco2_bg_art_min BETWEEN totalco2_bg_art_ref_range_lower AND totalco2_bg_art_ref_range_upper) AND (totalco2_bg_art_ref_range_lower IS NOT NULL) AND (totalco2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (totalco2_bg_art_min< totalco2_bg_art_ref_range_lower) AND (totalco2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (totalco2_bg_art_min > totalco2_bg_art_ref_range_upper) AND (totalco2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN totalco2_bg_art_min IS NULL THEN 'not ordered' ELSE 'no ref range' END AS totalco2_bg_art_min_status , totalco2_bg_art_max , CASE WHEN (totalco2_bg_art_max BETWEEN totalco2_bg_art_ref_range_lower AND totalco2_bg_art_ref_range_upper) AND (totalco2_bg_art_ref_range_lower IS NOT NULL) AND (totalco2_bg_art_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (totalco2_bg_art_max< totalco2_bg_art_ref_range_lower) AND (totalco2_bg_art_ref_range_lower IS NOT NULL) THEN 'low' WHEN (totalco2_bg_art_max > totalco2_bg_art_ref_range_upper) AND (totalco2_bg_art_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN totalco2_bg_art_max IS NULL THEN 'not ordered' ELSE 'no ref range' END AS totalco2_bg_art_max_status\
\
--parameters which were in SOFA - we don't use them here, as these are made of the parameters we have already used\
\
 --, SOFA , CASE WHEN (SOFA BETWEEN SOFA_ref_range_lower AND SOFA_ref_range_upper) AND (SOFA_ref_range_lower IS NOT NULL) AND (SOFA_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (SOFA< SOFA_ref_range_lower) AND (SOFA_ref_range_lower IS NOT NULL) THEN 'low' WHEN (SOFA > SOFA_ref_range_upper) AND (SOFA_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN SOFA IS NULL THEN 'not ordered' ELSE 'no ref range' END AS SOFA_status\
-- , respiration , CASE WHEN (respiration BETWEEN respiration_ref_range_lower AND respiration_ref_range_upper) AND (respiration_ref_range_lower IS NOT NULL) AND (respiration_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (respiration< respiration_ref_range_lower) AND (respiration_ref_range_lower IS NOT NULL) THEN 'low' WHEN (respiration > respiration_ref_range_upper) AND (respiration_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN respiration IS NULL THEN 'not ordered' ELSE 'no ref range' END AS respiration_status\
-- , coagulation , CASE WHEN (coagulation BETWEEN coagulation_ref_range_lower AND coagulation_ref_range_upper) AND (coagulation_ref_range_lower IS NOT NULL) AND (coagulation_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (coagulation< coagulation_ref_range_lower) AND (coagulation_ref_range_lower IS NOT NULL) THEN 'low' WHEN (coagulation > coagulation_ref_range_upper) AND (coagulation_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN coagulation IS NULL THEN 'not ordered' ELSE 'no ref range' END AS coagulation_status\
 --, liver , CASE WHEN (liver BETWEEN liver_ref_range_lower AND liver_ref_range_upper) AND (liver_ref_range_lower IS NOT NULL) AND (liver_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (liver< liver_ref_range_lower) AND (liver_ref_range_lower IS NOT NULL) THEN 'low' WHEN (liver > liver_ref_range_upper) AND (liver_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN liver IS NULL THEN 'not ordered' ELSE 'no ref range' END AS liver_status\
 --, cardiovascular , CASE WHEN (cardiovascular BETWEEN cardiovascular_ref_range_lower AND cardiovascular_ref_range_upper) AND (cardiovascular_ref_range_lower IS NOT NULL) AND (cardiovascular_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (cardiovascular< cardiovascular_ref_range_lower) AND (cardiovascular_ref_range_lower IS NOT NULL) THEN 'low' WHEN (cardiovascular > cardiovascular_ref_range_upper) AND (cardiovascular_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN cardiovascular IS NULL THEN 'not ordered' ELSE 'no ref range' END AS cardiovascular_status\
 --, cns , CASE WHEN (cns BETWEEN cns_ref_range_lower AND cns_ref_range_upper) AND (cns_ref_range_lower IS NOT NULL) AND (cns_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (cns< cns_ref_range_lower) AND (cns_ref_range_lower IS NOT NULL) THEN 'low' WHEN (cns > cns_ref_range_upper) AND (cns_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN cns IS NULL THEN 'not ordered' ELSE 'no ref range' END AS cns_status\
 --, renal , CASE WHEN (renal BETWEEN renal_ref_range_lower AND renal_ref_range_upper) AND (renal_ref_range_lower IS NOT NULL) AND (renal_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (renal< renal_ref_range_lower) AND (renal_ref_range_lower IS NOT NULL) THEN 'low' WHEN (renal > renal_ref_range_upper) AND (renal_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN renal IS NULL THEN 'not ordered' ELSE 'no ref range' END AS renal_status\
\
--parameters which were in GCS - we only use HCS_min (total) and exclude individual components\
\
, gcs_min \
\
, CASE WHEN gcs_min = 15 THEN 'normal'\
WHEN (gcs_min BETWEEN 13 AND 14) THEN 'less abnormal-level 1'\
WHEN (gcs_min BETWEEN 10 AND 12) THEN 'moderately abnormal-level 2'\
WHEN (gcs_min BETWEEN 6 AND 9) THEN 'highly abnormal-level 3'\
WHEN (gcs_min < 6) THEN 'severely abnormal-level 4'\
WHEN (gcs_min IS NULL) THEN 'not ordered'\
ELSE 'no ref range' END AS gcs_min_status\
\
-- , gcs_motor , CASE WHEN (gcs_motor BETWEEN gcs_motor_ref_range_lower AND gcs_motor_ref_range_upper) AND (gcs_motor_ref_range_lower IS NOT NULL) AND (gcs_motor_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (gcs_motor< gcs_motor_ref_range_lower) AND (gcs_motor_ref_range_lower IS NOT NULL) THEN 'low' WHEN (gcs_motor > gcs_motor_ref_range_upper) AND (gcs_motor_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN gcs_motor IS NULL THEN 'not ordered' ELSE 'no ref range' END AS gcs_motor_status\
-- , gcs_verbal , CASE WHEN (gcs_verbal BETWEEN gcs_verbal_ref_range_lower AND gcs_verbal_ref_range_upper) AND (gcs_verbal_ref_range_lower IS NOT NULL) AND (gcs_verbal_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (gcs_verbal< gcs_verbal_ref_range_lower) AND (gcs_verbal_ref_range_lower IS NOT NULL) THEN 'low' WHEN (gcs_verbal > gcs_verbal_ref_range_upper) AND (gcs_verbal_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN gcs_verbal IS NULL THEN 'not ordered' ELSE 'no ref range' END AS gcs_verbal_status\
-- , gcs_eyes , CASE WHEN (gcs_eyes BETWEEN gcs_eyes_ref_range_lower AND gcs_eyes_ref_range_upper) AND (gcs_eyes_ref_range_lower IS NOT NULL) AND (gcs_eyes_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (gcs_eyes< gcs_eyes_ref_range_lower) AND (gcs_eyes_ref_range_lower IS NOT NULL) THEN 'low' WHEN (gcs_eyes > gcs_eyes_ref_range_upper) AND (gcs_eyes_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN gcs_eyes IS NULL THEN 'not ordered' ELSE 'no ref range' END AS gcs_eyes_status\
-- , gcs_unable , CASE WHEN (gcs_unable BETWEEN gcs_unable_ref_range_lower AND gcs_unable_ref_range_upper) AND (gcs_unable_ref_range_lower IS NOT NULL) AND (gcs_unable_ref_range_upper IS NOT NULL) THEN 'normal' WHEN (gcs_unable< gcs_unable_ref_range_lower) AND (gcs_unable_ref_range_lower IS NOT NULL) THEN 'low' WHEN (gcs_unable > gcs_unable_ref_range_upper) AND (gcs_unable_ref_range_upper IS NOT NULL) THEN 'elevated' WHEN gcs_unable IS NULL THEN 'not ordered' ELSE 'no ref range' END AS gcs_unable_status\
\
from after_deleting_duplicate_attributes\
)\
\
#--------------------------------------------------------------------------------------------------------------------------------------------------\
# Export results\
\
# Tabel #1 - row count - 13416 , distinct hadm_id count - 13416\
# one row per one hadm_id \
\
#select * from first_hep_with_hep_type_and_treatment_type_dermographics\
\
#---\
\
# Tabel #2 - multiple rows per one hadm_id # row count - 179230 , distinct hadm_id count - 13394\
#select * from JOIN_HEAPRIN_PLATELET_LEFT\
\
#---\
\
# Tabel #3 - one row per one hadm_id # 13416\
# select * from check_status_of_each_vital_sign_full_list\
\
#---------------------------------------------------------------------------------------------------------------------------------------------------\
\
select * from first_hep_with_hep_type_and_treatment_type_dermographics\
\
}