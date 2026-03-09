"""
AMR knowledge base containing gene and antibiotic class facts.
"""

GENE_FACTS = {
    "CTX-M-15": (
        "CTX-M-15 is the most prevalent plasmid-encoded ESBL worldwide, "
        "hydrolyses third-generation cephalosporins and aztreonam, and is "
        "typically associated with pandemic E. coli clonal lineage ST131."
    ),
    "KPC-1": (
        "KPC-1 is an Ambler class A serine carbapenemase that inactivates "
        "all beta-lactam antibiotics including carbapenems, conferring "
        "pan-resistance to this antibiotic class."
    ),
    "MCR-1": (
        "MCR-1 encodes a plasmid-mediated phosphoethanolamine transferase "
        "that modifies lipid A, reducing colistin binding and conferring "
        "resistance to the last-resort antibiotic colistin."
    ),
    "CMY-59": (
        "CMY-59 is a plasmid-mediated AmpC beta-lactamase that hydrolyses "
        "penicillins, cephalosporins and cephamycins and is not inhibited "
        "by clavulanate or tazobactam."
    ),
    "acrB": (
        "AcrB is the inner membrane transporter of the AcrAB-TolC efflux "
        "pump, the primary multidrug efflux system in E. coli, and exports "
        "fluoroquinolones, tetracyclines, beta-lactams and chloramphenicol."
    ),
    "QnrS1": (
        "QnrS1 is a pentapeptide repeat protein that protects DNA gyrase "
        "from fluoroquinolone inhibition and is encoded on conjugative "
        "plasmids, facilitating horizontal dissemination."
    ),
    "tet(A)": (
        "Tet(A) is an MFS tetracycline efflux pump that confers resistance "
        "to tetracycline; tigecycline (glycylcycline) retains activity "
        "against tet(A)-expressing strains."
    ),
    "tet(B)": (
        "Tet(B) is a broad tetracycline efflux pump including minocycline "
        "resistance, located on broad-host-range Tn10 transposons enabling "
        "inter-species spread."
    ),
    "sul1": (
        "Sul1 encodes a drug-insensitive dihydropteroate synthase conferring "
        "sulfonamide resistance, typically located on class 1 integrons "
        "alongside additional resistance gene cassettes."
    ),
    "sul2": (
        "Sul2 encodes an alternative sulfonamide-resistant DHPS on small "
        "mobilisable plasmids with high horizontal gene transfer potential "
        "across Enterobacteriaceae."
    ),
    "dfrA17": (
        "DfrA17 encodes a trimethoprim-resistant DHFR that eliminates "
        "co-trimoxazole therapy for UTI and enteric infections when combined "
        "with sul genes."
    ),
    "aadA5": (
        "AadA5 is aminoglycoside nucleotidyltransferase ANT(3'')-Ia that "
        "inactivates streptomycin and spectinomycin by adenylation and is "
        "commonly found in class 1 integrons."
    ),
    "floR": (
        "FloR is an MFS efflux pump conferring chloramphenicol and "
        "florfenicol resistance with clinical relevance in both human "
        "medicine and veterinary settings."
    ),
    "mphA": (
        "MphA is a macrolide 2-phosphotransferase that inactivates "
        "azithromycin and erythromycin and is plasmid-mediated with "
        "increasing prevalence in Enterobacteriaceae."
    ),
    "mphB": (
        "MphB is a macrolide phosphotransferase active against 14 and "
        "15-membered ring macrolides and is often co-carried with mphA "
        "on conjugative plasmids."
    ),
    "marA": (
        "MarA is a global multiple antibiotic resistance regulator that "
        "upregulates AcrAB-TolC efflux and downregulates OmpF porin "
        "expression to reduce intracellular drug concentrations."
    ),
    "tolC": (
        "TolC is the outer membrane channel essential for AcrAB-TolC "
        "efflux pump assembly and is required for intrinsic multidrug "
        "resistance in E. coli."
    ),
    "CRP": (
        "CRP is a global transcriptional activator of efflux genes and "
        "biofilm determinants, contributing to antibiotic tolerance and "
        "multidrug resistance phenotypes."
    ),
    "H-NS": (
        "H-NS is a histone-like nucleoid structuring protein that silences "
        "acquired resistance gene clusters; loss-of-function mutations "
        "de-repress resistance pathways."
    ),
    "bacA": (
        "BacA is an undecaprenyl pyrophosphate phosphatase that confers "
        "bacitracin resistance and supports cell wall precursor "
        "biosynthesis in E. coli."
    ),
    "Escherichia_coli_ampC1_beta-lactamase": (
        "AmpC1 is the chromosomal AmpC beta-lactamase of E. coli conferring "
        "intrinsic resistance to aminopenicillins and early cephalosporins; "
        "derepression causes broad cephalosporin and cephamycin resistance."
    ),
    "APH(3'')-Ib": (
        "APH(3'')-Ib is an aminoglycoside phosphotransferase that "
        "inactivates streptomycin and is commonly found on resistance "
        "plasmids alongside APH(6)-Id."
    ),
    "APH(3')-Ia": (
        "APH(3')-Ia is an aminoglycoside 3-phosphotransferase conferring "
        "kanamycin and neomycin resistance via phosphorylation."
    ),
    "APH(6)-Id": (
        "APH(6)-Id is a streptomycin-inactivating phosphotransferase "
        "frequently co-located with APH(3'')-Ib on the same resistance "
        "module."
    ),
    "Escherichia_coli_acrA": (
        "AcrA is the periplasmic adaptor protein of AcrAB-TolC; essential "
        "for pump assembly and efflux function."
    ),
    "Escherichia_coli_emrE": (
        "EmrE is a small multidrug resistance protein conferring resistance "
        "to ethidium bromide and low-MW antiseptics via proton-coupled "
        "efflux."
    ),
    "Escherichia_coli_mdfA": (
        "MdfA is an MFS multidrug efflux transporter with broad substrate "
        "range including fluoroquinolones and chloramphenicol."
    ),
}

CLASS_FACTS = {
    "carbapenem": (
        "Carbapenem resistance is a critical public health emergency. "
        "Options: colistin/polymyxin B, ceftazidime-avibactam (KPC), "
        "aztreonam-avibactam (MBL producers), or cefiderocol."
    ),
    "cephalosporin": (
        "ESBL or AmpC eliminates third-generation cephalosporins. "
        "Carbapenem required for systemic MDR infections."
    ),
    "fluoroquinolone": (
        "Ciprofloxacin and levofloxacin eliminated by PMQR plus chromosomal "
        "topoisomerase mutations conferring high-level resistance."
    ),
    "tetracycline": (
        "Tet(A)/tet(B) efflux eliminates tetracycline and doxycycline; "
        "tigecycline typically retains activity."
    ),
    "aminoglycoside": (
        "Gentamicin resistance reduces synergistic combination options; "
        "amikacin may retain activity depending on enzyme type."
    ),
    "sulfonamide": (
        "Co-trimoxazole eliminated for UTI, travellers diarrhoea and "
        "prophylaxis in immunocompromised patients."
    ),
    "macrolide": (
        "Azithromycin and erythromycin inactivated by phosphotransferases; "
        "clinically significant for enteric infections."
    ),
    "phenicol": (
        "Chloramphenicol resistance eliminates a key drug in resource-limited "
        "settings for typhoid and bacterial meningitis."
    ),
    "peptide": (
        "Colistin resistance eliminates the last-resort antibiotic class; "
        "immediate infectious disease specialist consultation required."
    ),
    "penam": (
        "Near-universal E. coli penicillin resistance via intrinsic AmpC; "
        "empirical penicillin therapy is not appropriate."
    ),
    "cephamycin": (
        "AmpC-mediated cefoxitin and cefotetan resistance; carbapenems "
        "required for definitive therapy."
    ),
    "diaminopyrimidine": (
        "Trimethoprim resistance via dfr genes eliminates co-trimoxazole "
        "for urinary and enteric infections."
    ),
    "glycylcycline": (
        "Tigecycline resistance emerging; MIC determination recommended "
        "before clinical use."
    ),
    "monobactam": (
        "Aztreonam affected by ESBLs; aztreonam-avibactam restores activity "
        "for MBL-producing strains."
    ),
    "fosfomycin": (
        "Fosfomycin retains activity in many MDR E. coli UTI cases; "
        "resistance emerging."
    ),
    "rifamycin": (
        "Rifampicin resistance limits combination therapy options in XDR "
        "gram-negative infections."
    ),
    "acridine_dye": (
        "Efflux-mediated disinfectant resistance with antibiotic "
        "cross-resistance implications."
    ),
    "aminocoumarin": (
        "Broad efflux upregulation marker; limited direct clinical impact."
    ),
    "triclosan": (
        "Biocide resistance via efflux and FabI modification; potential "
        "antibiotic cross-resistance."
    ),
    "benzalkonium_chloride": (
        "Quaternary ammonium compound efflux resistance; healthcare "
        "decontamination failure risk."
    ),
    "penem": (
        "Resistance to penems via AmpC or carbapenemase production."
    ),
    "nitroimidazole": (
        "Metronidazole resistance; relevant in polymicrobial infections."
    ),
    "lincosamide": (
        "Clindamycin resistance indicating broad resistance plasmid "
        "acquisition."
    ),
    "nucleoside": (
        "Nucleoside antibiotic resistance marking broad MDR plasmid "
        "carriage."
    ),
    "rhodamine": (
        "Disinfectant co-resistance via multidrug efflux pumps."
    ),
}
