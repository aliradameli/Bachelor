"""2-D numerical breast phantom (speed-of-sound + density maps) -- BREAST ONLY, depth 0-10 cm.

Maps have shape (Nx, Ny): axis 0 = lateral (dx), axis 1 = depth (dy); depth D = Ny*dy [m].

SCOPE (REF2 §0.1 #1, §1, §2, §9):
  * z = 0 is the skin surface (or the gel standoff); z = D is the POSTERIOR BREAST BOUNDARY (end of the
    retromammary fat). There is NO chest wall: no pectoralis, intercostal muscle, ribs, cartilage, pleura, lung.
  * A larger D means a LARGER BREAST, not a deeper look into the thorax. D = 4, 6, 8, 10 cm are size classes.
    Layer thicknesses scale with D (see _get_layout). If your transducer sees deeper than D, use a PML/absorbing
    boundary or a fat half-space below the grid (REF2 §9).

Reference tags used in comments
  REF2   = "Acoustic Reference for 2D Numerical Breast Phantom Modeling - Breast-Only, Size Classes 4-10 cm (v2)", 2026-11-23
           (the research document that accompanies this code; section numbers are its sections)
  REF2b  = the short "Comprehensive Acoustic Reference ... (0-10 cm)" working draft (breast-only policy, contour model
           r(theta) = r0[1 + sum a_k cos(k theta + phi_k)], eta depth scaling, lesion generator requirements §3.2-3.4, §4.5)
  Li22   = Li F, Villa U, Park S, Anastasio MA. IEEE TUFFC 2022;69(1):135-146 (3-D stochastic USCT phantoms)
  Littrup21 = Littrup PJ, Duric N, Sak M, et al. J Breast Imaging 2021, doi:10.1093/jbi/wbab032
  Anzola25  = Anzola ML et al. Tomography 2025;11(10):111        Bezek25 = Bezek CD et al. Eur Radiol 2025;35(3):1490
  Gity21    = Gity M et al. Arch Iran Med 2021;24(5):383         Shao23  = Shao S et al. Front Oncol 2023;13:1103397
  Watanabe21= Watanabe T et al. J Med Ultrason 2021;48:71        Khalaf20= Khalaf LMR, Herdan RA. Egypt J Radiol Nucl Med 2020;51:138
  Fakhry25  = Fakhry S et al. Egypt J Radiol Nucl Med 2025;56:226 Pan24   = Pan H et al. Quant Imaging Med Surg 2024;14(5):3381
  Zhang22   = Zhang Q et al. Front Oncol 2022;12:868164          Salvatorelli20 = Salvatorelli L et al. Cancers 2020;12(3):609
  Wang25    = Wang Y et al. Quant Imaging Med Surg 2025;15(6):5218 (skin thickness)
  Sizikova23= Sizikova E et al. arXiv:2310.18494 (Gaussian random sphere + spicules)
  VanCamp23 = Van Camp A et al. Prog Biomed Eng 2023;5:012003 (lesion-model review)
  Sengupta21/24 = Sengupta A, Sharma D, Badano A. SPIE 11595 (2021); Phys Med Biol 69(7) (2024) (stiffness-driven tumour growth)
  PMC10965701 = 2024 paper, Table 2, n = 130 cancers (tumour-to-skin 8.6 +/- 4.4 mm, tumour-to-nipple 41.1 +/- 30.0 mm)
  Pawlowska = Pawlowska, Zolek, Litniewski, modified-VICTRE acoustic table (IEEE IUS; year unconfirmed in REF2 §10)
Evidence tags (REF2 §0.2):  [P] 2020+ primary study | [R] 2020+ review / quoted | [L] legacy value (pre-2020 source,
unverified, kept so the model is complete) | [A] modelling assumption of this code | [C] calibration run for REF2 §7.4.4.
Where a range is given without SD the SD used is (max-min)/6 and is marked (*).

Usage:
    bm = BreastMap(Nx=400, Ny=400, dx=2e-4, dy=2e-4, rndseed=1, birads='C')   # D = 8 cm breast
    bm.disable('vessels'); bm.set_entities(tumor_cyst=False)
    bm.build()
    sos, rho, lab = bm.get_sos(), bm.get_rho(), bm.get_labels()
Old chain style (bm.addGel().addEpidermis()...) still works; each add* is a no-op when its entity is disabled.
Use build() for the correct order.
"""


MAX_DEPTH_M = 0.10   # REF2 §2: size classes 4, 6, 8, 10 cm; REF2b §1: configurable extent 0-10 cm
MIN_DEPTH_M = 0.02   # [A] below this there is no room for skin + subcutaneous fat + gland + retromammary fat


def _const(v):
    """Constant property (no spread): used where REF2 §3 gives a single value (Pawlowska rows)."""
    return (float(v), 0.0, float(v), float(v))


# ----------------------------------------------------------------------------------------------
# Acoustic properties: (mean, sd, min, max). c in m/s, rho in kg/m^3.  Source: REF2 §3 master table
# ----------------------------------------------------------------------------------------------
ACOUSTIC_PROPS = {
    # Coupling water/gel: ~1500 m/s at 26 C, rho 994-1000 (REF2 §3; Li22 citing Greenspan & Tschiegg 1959 [L]).
    # SOS spread = temperature range 1482-1525 [L, v1 table].
    'water_gel': {'c': (1500, 8, 1482, 1525),        'rho': (997, 1.5, 994, 1000)},                  # [P]/[L] (*)
    # Bulk skin: Li22 Table II (REF2 §3): 1555 +/- 10 (1530-1580), 1109 +/- 14 (1100-1125); Pawlowska 1537 / 1100
    'skin':       {'c': (1555, 10, 1530, 1580),      'rho': (1109, 14, 1100, 1125)},                 # [P]
    # Epidermis / dermis sub-layers: Moran 1995 (REF2 §3) [L]. Used only when skin_model='layered'.
    'epidermis':  {'c': (1645, 34, 1532, 1736),      'rho': (1150, 13, 1110, 1190)},                 # [L] (*)
    'dermis':     {'c': (1595, 19, 1569, 1681),      'rho': (1100, 14, 1090, 1130)},                 # [L] (*)
    # Fat (subcutaneous + retromammary): Li22 Table II = 1440.2 +/- 20.9 (1412-1485), 911 +/- 53 (812-961)
    # (Pawlowska: 1440 / 940)
    'fat':        {'c': (1440.2, 20.9, 1412, 1485),  'rho': (911, 53, 812, 961)},                    # [P]
    # Fibroglandular: Li22 Table II = 1540 +/- 15 (1517-1567), 1041 +/- 45.3 (990-1092) (Pawlowska 1560 / 1050)
    'gland':      {'c': (1540, 15, 1517, 1567),      'rho': (1041, 45.3, 990, 1092)},                # [P]
    # Lactiferous ducts: Pawlowska 1545 / 1030 (REF2 §3), single values
    'duct':       {'c': _const(1545),                'rho': _const(1030)},                           # [P, year unconfirmed]
    # Cooper's ligaments, grid-dependent (REF2 §3 + §3.1 #2: volume averaging explains the 1457 vs 1750 gap)
    'cooper_coarse': {'c': (1457, 18.5, 1422, 1496), 'rho': (1142, 45, 1110, 1174)},                 # [P] Li22, grid step > 0.25 mm
    'cooper_fine':   {'c': _const(1750),             'rho': _const(1170)},                           # [P] Pawlowska, grid step <= 0.25 mm
    # Blood (vessel lumen): Pawlowska 1570 / 1050 (REF2 §3)
    'blood':      {'c': _const(1570),                'rho': _const(1050)},                           # [P, year unconfirmed]
    # Vessel wall: not in REF2 §3; v1 value (IT'IS blood-vessel wall, Hasgall 2022) kept [L]
    'vessel_wall': {'c': (1584, 15, 1555, 1610),     'rho': (1102, 53, 1049, 1155)},                 # [L] (*) c
    # Invasive ductal carcinoma: Li22 Table II = 1548 +/- 10.3 (1531-1565); rho 945 +/- 20 (911-999), see REF2 §3.1 #1
    # (Pawlowska: 1580 / 1070). In-vivo check value (not sampled): malignant mass mean 1554 +/- 27 (Anzola25).
    'idc':        {'c': (1548, 10.3, 1531, 1565),    'rho': (945, 20, 911, 999)},                    # [P]
    # DCIS and fibroadenoma: Li 2009 / Ruby 2019 via v1 (REF2 §3: no 2020+ value) [L].
    # Check value for benign masses: 1499 +/- 73 m/s (Anzola25; fibroepithelial + apocrine metaplasia group).
    'dcis':       {'c': (1535, 15, 1510, 1565),      'rho': (1055, 8, 1035, 1070)},                  # [L]
    'fibroadenoma': {'c': (1513, 27, 1486, 1540),    'rho': (1045, 9, 1020, 1065)},                  # [L]
    # Cyst fluid: plasma/serum-like (IT'IS v4.1 via v1; REF2 §3 quotes ~1500-1525 m/s, ~1020 kg/m3) [L]
    'cyst':       {'c': (1510, 7, 1495, 1530),       'rho': (1020, 3, 1010, 1025)},                  # [L]
}

# REF2 §3.1 #1: no 2020+ acoustic density measurement for tumours exists. Two options exposed via tumor_density_model:
#   'li22'  -> 945 +/- 20 (Li22, reproduces the primary basis)   'solid' -> U(1045, 1070) so that Z_tumour > Z_gland [A]
IDC_RHO_SOLID_RANGE = (1045.0, 1070.0)   # [A] REF2 §3.1 #1 (upper end = Pawlowska 1070)

# Constant inclusions. Calcium hydroxyapatite c = 3200 m/s, rho = 2800 kg/m^3 (Wear & Laib 2003 via v1; REF2 §3) [L]
MICROCALC_C, MICROCALC_RHO = 3200.0, 2800.0
# Peritumoral halo: tumour/collagen-like properties. No 2020+ acoustic value found (REF2 §7.5) -> [L]/[A] from v1.
HALO_C, HALO_RHO = 1560.0, 1070.0

# ----------------------------------------------------------------------------------------------
# Macro-scale correlated texture, Li22 Table III (REF2 §4.1). v1 mislabelled the SDs as correlation lengths (REF2 §0.1 #4).
# ----------------------------------------------------------------------------------------------
TEXTURE_SD = {('fat', 'c'): 28.8, ('fat', 'rho'): 18.22,          # [P] Li22 Table III (m/s, kg/m3)
              ('gland', 'c'): 30.4, ('gland', 'rho'): 20.82}      # [P] Li22 Table III
TEXTURE_TRUNC_SIGMA = {'fat': 0.9, 'gland': None}                 # [P] fatty texture truncated at +/-0.9 sigma (Li22 Table III)
TEXTURE_CORR_LEN_M = 0.21e-3                                      # [P] correlation length 0.21 mm (Li22 Table III; REF2 §4.1)
# Gaussian covariance exp(-d^2/(2 l^2)) is produced by Gaussian-smoothing white noise with std l/sqrt(2)... [A] (convention)

# BI-RADS adipose fraction of the parenchymal zone: v1 convention [L] (REF2 §5)
BIRADS_KFAT = {'A': 0.95, 'B': 0.85, 'C': 0.66, 'D': 0.40}
# Global breast SOS by density (mean, SD) m/s, Bezek25, n = 100: VALIDATION TARGET ONLY (REF2 §5). A global pulse-echo SOS is not
# the area mean of a map, so use for ordering / spacing, not as exact targets.
BEZEK25_GLOBAL_SOS = {'A': (1459.6, 0.74), 'B': (1475.6, 15.92), 'C': (1515.6, 27.10), 'D': (1545.7, 20.62)}   # [P]

# ----------------------------------------------------------------------------------------------
# Skin / depth blueprint (REF2 §2.2). All thickness choices below are in meters.
# ----------------------------------------------------------------------------------------------
GEL_THICKNESS_M = (0.5e-3, 1.0e-3)              # [L] 0.5-1.0 mm standoff (REF2 §2.3 / v1)
EPI_THICKNESS_M = (0.104e-3, 0.019e-3, 0.066e-3, 0.142e-3)    # [P, proxy] Wang25 chest-wall skin 0.104 +/- 0.019 mm; +/-2 SD clip [A]
DERMIS_THICKNESS_M = (1.96e-3, 0.50e-3, 1.5e-3, 2.5e-3)       # [P, proxy] Wang25 1.96 +/- 0.50 mm; clip 1.5-2.5 mm = REF2 §2.2 rule [A]
SUBCUT_FRACTION_OF_D = 0.10                     # [A] REF2 §2.2: subcutaneous fat t_sc = 0.10*D ...
SUBCUT_SCALE_RANGE = (0.6, 1.5)                 # [A] ... with range 0.6-1.5x (REF2 §2.2)
RETRO_FAT_M = (3e-3, 5e-3)                      # [L] retromammary fat 3-5 mm, last layer of the grid (REF2 §2.2/§2.3)
MIN_CORPUS_M = 5e-3                             # [A] minimum fibroglandular zone thickness
LESION_MARGIN_M = 1e-3                          # [A] lesions stay >= 1 mm from skin and from the retromammary fat (REF2 §2.2 footnote)

# ----------------------------------------------------------------------------------------------
# Lesion population
# ----------------------------------------------------------------------------------------------
# Relative frequency: 78 cancers : 105 fibroadenomas : 91 cysts = 0.285 : 0.383 : 0.332 (Littrup21, REF2 §6.1) [P].
# Split of the cancer share into IDC 80 % / DCIS 20 % is [A] (REF2 §6.1).  (Trial population, not general prevalence.)
TUMOR_WEIGHTS = {'idc': 0.228, 'dcis': 0.057, 'fibroadenoma': 0.383, 'cyst': 0.332}

# Largest-dimension size, truncated log-normal: (median mm, sigma of ln, min mm, max mm). REF2 §6.2 (20 mm is NOT a hard cap).
#  screening IDC : mean 15.7 mm (5-36), Gity21; SD 7.8 mm (Watanabe21 1.6 +/- 0.78 cm; PMC10965701 1.3 +/- 0.78 cm)
#                  -> moment fit sigma = sqrt(ln(1+(7.8/15.7)^2)) = 0.469, median = 15.7/sqrt(1+cv^2) = 14.06 [A, anchored to P]
#  symptomatic IDC: mean 31 +/- 16.1 mm (1-9 cm), Khalaf20 -> sigma 0.489, median 27.5 [A, anchored to P]
#  DCIS          : no 2020+ size data; v1 fit to Rauch 2013 (US size 16 +/- 14 mm) median 12.9, sigma 0.739 [L]; upper clip 36 mm = Gity21 max
#  fibroadenoma  : screening: Littrup21 mean 11 mm, 88/105 < 15 mm, median 10, sigma 0.485 (v1 fit) [P/L];
#                  symptomatic/biopsy: mean 28 mm, max 100 mm (Fakhry25) -> median 24.7, sigma 0.5 [A]
#  cyst          : Littrup21 mean 10 mm, 80/91 < 15 mm, median 9.2 mm (REF2 §6.2) -> sigma 0.40 [P/L]
LESION_SIZE_MM = {
    'screening':   {'idc': (14.06, 0.469, 5.0, 36.0), 'dcis': (12.9, 0.739, 1.5, 36.0),
                    'fibroadenoma': (10.0, 0.485, 4.0, 50.0), 'cyst': (9.2, 0.40, 2.0, 30.0)},
    'symptomatic': {'idc': (27.5, 0.489, 10.0, 90.0), 'dcis': (12.9, 0.739, 1.5, 36.0),
                    'fibroadenoma': (24.7, 0.50, 5.0, 100.0), 'cyst': (9.2, 0.40, 2.0, 30.0)},
}

# Where a lesion sits relative to the fat-glandular interface (FGI): (at FGI, inside gland, inside fat). Littrup21, REF2 §6.1 [P]
#   cancer 74/78 = 95 / 0 / 5 %;  fibroadenoma 66/105 = 63 / 35 / 2 %;  cyst 33/91 = 36 / 64 / 0 %. DCIS follows the cancer row [A].
PLACEMENT = {'idc': (0.95, 0.0, 0.05), 'dcis': (0.95, 0.0, 0.05),
             'fibroadenoma': (0.63, 0.35, 0.02), 'cyst': (0.36, 0.64, 0.0)}

# Distance priors, PMC10965701 Table 2 (n = 130), REF2 §6.3 [P]:
#  tumour-to-skin 8.6 +/- 4.4 mm -> left-truncated normal (REF2 §6.3 implementation note). Measured from the dermis bottom to the
#  nearest tumour margin (literature definition), NOT scaled with D (REF2 §2.2 footnote / §6.3).
#  tumour-to-nipple 41.1 +/- 30.0 mm is a 3-D distance; used as in-plane lateral offset = [A] approximation (REF2 §6.3);
#  gamma distribution moment-matched to mean/SD [A] (keeps the mode away from 0).
TUMOR_TO_SKIN_MM = (8.6, 4.4)
TUMOR_TO_NIPPLE_MM = (41.1, 30.0)

# ---- Shape / margin / orientation ------------------------------------------------------------
# Malignant profiles (REF2 §7.2). Default Gity21 for general IDC; Shao23 for early (< 20 mm) cancers (REF2 §7.2 bullet 1).
#  gity21: 78 IDC, shape irregular 55.1 / oval 34.6 / round 10.2 %, non-parallel 59 % -> parallel 41 %  [P]
#  shao23: 790 cancers < 20 mm, irregular 88.9 / round 6.3 / oval 4.8 %, parallel 83.5 %                [P]
MALIGNANT_PROFILES = {
    'gity21': {'shape': {'irregular': 0.551, 'oval': 0.346, 'round': 0.102}, 'p_parallel': 0.41},
    'shao23': {'shape': {'irregular': 0.889, 'oval': 0.048, 'round': 0.063}, 'p_parallel': 0.835},
}
# IDC margin mix: Gity21 (indistinct 43.5, spiculated ~24.3, circumscribed ~19.2, angular ~6.4, microlobulated ~6.4 %) [P].
# (Shao23 gives no margin split -> same mix used for both profiles [A].)
IDC_MARGIN_PROBS = {'indistinct': 0.435, 'spiculated': 0.243, 'circumscribed': 0.192, 'angular': 0.064, 'microlobulated': 0.064}
# DCIS shape: v1 [L] Rauch 2013 (irregular 68 / oval 27 / round 5 %); no 2020+ frequencies (REF2 §7.2).
# DCIS margin: Salvatorelli20 describe an irregular hypoechoic mass with indistinct margins [R] -> weights are [A].
DCIS_SHAPE_PROBS = {'irregular': 0.682, 'oval': 0.265, 'round': 0.053}
DCIS_MARGIN_PROBS = {'indistinct': 0.70, 'microlobulated': 0.15, 'angular': 0.15}
# Fibroadenoma shape: Clauser 2017 oval 69.7 / lobulated 17.6 / round 12.7 % [L] (no 2020+ percentages; Fakhry25 reports no
# shape difference between simple and complex FA, REF2 §7.2).  Cyst round/oval 50:50 [A] (REF2 §7.2).
FA_SHAPE_PROBS = {'oval': 0.697, 'lobulated': 0.176, 'round': 0.127}
CYST_SHAPE_PROBS = {'oval': 0.5, 'round': 0.5}
# Margin classes allowed with each shape class (so that a 'round'/'oval' lesion never gets spicules): [A]
MARGINS_REGULAR = ('circumscribed', 'indistinct', 'microlobulated')            # round / oval
MARGINS_IRREGULAR = ('indistinct', 'angular', 'microlobulated', 'spiculated')   # irregular

# Depth/width ratio (AP/lateral), (mean, sd, min, max). "Parallel" = wider than tall (BI-RADS lexicon, Choi 2023 used for
# terminology only). Parallel fractions come from MALIGNANT_PROFILES; the D/W values themselves are [A].
DW_PARALLEL = (0.65, 0.15, 0.35, 0.95)       # [A]
DW_NONPARALLEL = (1.15, 0.15, 1.0, 1.6)      # [A]
DW_ROUND = (0.93, 0.05, 0.85, 1.0)           # [A] round -> D/W ~ 1 (REF2 §7.4.3)
DW_FIBROADENOMA = (0.62, 0.12, 0.5, 0.95)    # [A] REF2 §7.4.3: FA D/W 0.5-1; Dogan 2017 long/short 1.98 +/- 0.41 -> D/W ~0.5 [L]
DW_CYST_OVAL = (0.8, 0.10, 0.6, 0.95)        # [A] REF2 §7.4.3: cyst anisotropy 0.6-0.95

# Calcifications (REF2 §7.6)
CALC_PROB_DCIS = 0.80          # [R] calcifications are the detection feature in ~70-90 % of DCIS (Salvatorelli20); midpoint [A]
CALC_CLUSTER_PROB = 0.57       # [L] compact cluster vs scattered (Rauch 2013)
CALC_SIZE_M = (0.1e-3, 1.0e-3) # [L] 0.1-1 mm (REF2 §7.6)
CALC_MAX_PIXEL_M = 0.3e-3      # [A] REF2 §7.6: if dx > 0.3 mm microcalcifications are sub-pixel -> dropped

DEFAULT_ENTITIES = {
    'gel': True, 'epidermis': True, 'dermis': True,
    'fibroglandular': True, 'cooper_ligaments': True, 'ducts': True, 'vessels': True,
    'tumor_idc': True, 'tumor_dcis': True, 'tumor_fibroadenoma': True, 'tumor_cyst': True,
    'peritumoral_halo': True, 'microcalcifications': True,
    'lesion_distance_priors': True,   # weight lesion centres by tumour-to-skin / nipple distance priors (REF2 §6.3)
    'macro_heterogeneity': True, 'speckle': True,
}
# Entities removed with the chest wall (REF2 §0.1 #1, §9). Passing one as False is silently accepted; True raises.
REMOVED_ENTITIES = ('retromammary_fascia', 'pectoralis', 'muscle_striations', 'intercostal_muscle', 'ribs',
                    'costal_cartilage', 'pleura_lung')

# Label numbers of the previous version are kept for the retained tissues (so old label maps stay comparable); 16-22 were chest wall.
LABELS = {'gel': 1, 'epidermis': 2, 'dermis': 3, 'fat': 4, 'fibroglandular': 5, 'cooper': 6,
          'duct': 7, 'vessel_wall': 8, 'blood': 9, 'idc': 10, 'dcis': 11, 'fibroadenoma': 12,
          'cyst': 13, 'halo': 14, 'microcalc': 15}
TUMOR_LABELS = [LABELS[k] for k in ('idc', 'dcis', 'fibroadenoma', 'cyst')]

# Relative stiffness for the growth model: ligaments high, gland medium, fat low (ordering from Sengupta24, REF2 §7.3/§7.4.2);
# the numbers themselves are [A].
GROWTH_STIFFNESS = {'fat': 1.0, 'fibroglandular': 2.0, 'duct': 2.0, 'cooper': 6.0, 'vessel_wall': 6.0, 'blood': 6.0,
                    'gel': 1.0, 'epidermis': 6.0, 'dermis': 6.0}
GROWTH_GAMMA, GROWTH_BETA = 1.0, 2.0   # p(c) ~ 1/S^gamma * (1 + beta * occupied_neighbours)  REF2 §7.4.2 [A]


def _choice(rng, probs):
    """Draw a key from a {key: weight} dict."""
    keys = list(probs)
    w = np.array([probs[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


class BreastMap(GridUnits):
    def __init__(self, Nx, Ny, dx, dy, rndseed, max_num_tumors=3, entities=None, birads='C',
                 center_frequency=5e6, skin_model='bulk', tumor_shape_model='contour',
                 malignant_profile='gity21', lesion_cohort='screening', tumor_density_model='li22', **legacy):
        """
        Nx, Ny, dx, dy : grid size and pixel size [m]; breast depth D = Ny*dy must be 2-10 cm
        max_num_tumors : 1..max lesions are drawn (0 = none)
        entities       : dict of flags overriding DEFAULT_ENTITIES
        birads         : 'A'|'B'|'C'|'D' -> adipose fraction of the parenchymal zone (REF2 §5)
        center_frequency : [Hz] only used to set the sub-wavelength scatterer density (speckle)
        skin_model     : 'bulk' (Li22 skin, REF2 §3 [P]) | 'layered' (Moran 1995 epidermis/dermis, [L])
        tumor_shape_model : 'contour' (Fourier-radial / spiculated / angular polygons, REF2 §7.4.1, default)
                            | 'growth' (stiffness-biased growth for IDC/DCIS, REF2 §7.4.2; others use 'contour')
                            | 'ellipse' (test shape only, REF2b §3.3 -- cannot represent malignant solidity, REF2 §7.1)
        malignant_profile : 'gity21' | 'shao23' shape/orientation mix for IDC & DCIS (REF2 §7.2)
        lesion_cohort  : 'screening' | 'symptomatic' size distributions (REF2 §6.2)
        tumor_density_model : 'li22' (IDC rho 945 +/- 20) | 'solid' (U(1045,1070)) (REF2 §3.1 #1)
        Removed arguments (chest wall / in-vivo override) are accepted and ignored with a warning.
        """
        for k in list(legacy):
            if k in ('physiological_state', 'muscle_fiber_orientation', 'rib_core'):
                warnings.warn(f"'{k}' was removed (breast-only model, single temperature convention; REF2 §0.1, §3.1 #3) - ignored.")
                legacy.pop(k)
        if legacy:
            raise TypeError(f"Unexpected arguments: {sorted(legacy)}")
        self.Nx, self.Ny, self.dx, self.dy = Nx, Ny, dx, dy
        self.rndseed = rndseed
        self.max_num_tumors = max_num_tumors
        self.depth = Ny * dy                                    # D = posterior breast boundary (REF2 §1)
        if self.depth > MAX_DEPTH_M + 1e-9:
            warnings.warn(f"Depth {self.depth*100:.1f} cm > 10 cm: outside the 4-10 cm size classes (REF2 §2); extrapolating.")
        if self.depth < MIN_DEPTH_M:
            raise ValueError(f"Depth {self.depth*100:.1f} cm is too small for a breast layer stack (min {MIN_DEPTH_M*100:.0f} cm).")
        for name, val, ok in (('birads', birads, BIRADS_KFAT), ('skin_model', skin_model, ('bulk', 'layered')),
                              ('tumor_shape_model', tumor_shape_model, ('contour', 'growth', 'ellipse')),
                              ('malignant_profile', malignant_profile, MALIGNANT_PROFILES),
                              ('lesion_cohort', lesion_cohort, LESION_SIZE_MM),
                              ('tumor_density_model', tumor_density_model, ('li22', 'solid'))):
            if val not in ok:
                raise ValueError(f"{name} must be one of {list(ok)}")
        self.birads, self.skin_model = birads, skin_model
        self.tumor_shape_model, self.malignant_profile = tumor_shape_model, malignant_profile
        self.lesion_cohort, self.tumor_density_model = lesion_cohort, tumor_density_model
        self.center_frequency = center_frequency

        self.entities = dict(DEFAULT_ENTITIES)
        if entities:
            self.set_entities(**entities)

        self.xx = np.arange(Nx) * dx          # lateral coordinates [m]
        self.zz = np.arange(Ny) * dy          # depth coordinates [m]
        self._reset()

    # ------------------------------------------------------------------ entity switches
    def set_entities(self, **flags):
        for k, v in flags.items():
            if k in REMOVED_ENTITIES:
                if v:
                    raise ValueError(f"Entity '{k}' was removed: the phantom is breast-only (REF2 §9).")
                warnings.warn(f"Entity '{k}' no longer exists (breast-only model) - ignored.")
                continue
            if k not in self.entities:
                raise KeyError(f"Unknown entity '{k}'. Available: {sorted(self.entities)}")
            self.entities[k] = bool(v)
        self._layout = None
        return self

    def enable(self, *names):
        return self.set_entities(**{n: True for n in names})

    def disable(self, *names):
        return self.set_entities(**{n: False for n in names})

    # ------------------------------------------------------------------ internals
    def _reset(self):
        self.sos_map = np.zeros((self.Nx, self.Ny))
        self.rho_map = np.zeros((self.Nx, self.Ny))
        self.label_map = np.zeros((self.Nx, self.Ny), dtype=np.int8)
        self.speckle_mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        self.gland_mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        self.lesions = []
        self.there_is_a_tumor = False
        self.gel_thickness = self.epi_thickness = self.total_skin_thickness = 0  # pixels
        self._layout = None

    def _rng(self, key):
        """Independent generator per entity -> toggling one entity doesn't change the others."""
        return np.random.default_rng([self.rndseed, zlib.crc32(key.encode())])

    @staticmethod
    def _tn(rng, spec, size=None):
        mean, sd, lo, hi = spec
        if sd <= 0:                                          # constant property
            return np.full(size, mean, dtype=float) if size is not None else float(mean)
        return truncnorm.rvs((lo - mean) / sd, (hi - mean) / sd, loc=mean, scale=sd,
                             size=size, random_state=rng)

    def _sample(self, name, rng):
        p = ACOUSTIC_PROPS[name]
        return float(self._tn(rng, p['c'])), float(self._tn(rng, p['rho']))

    def _paint(self, mask, c, rho, label, speckle=True):
        if isinstance(c, np.ndarray) and c.shape == mask.shape:
            c = c[mask]
        if isinstance(rho, np.ndarray) and rho.shape == mask.shape:
            rho = rho[mask]
        self.sos_map[mask] = c
        self.rho_map[mask] = rho
        self.label_map[mask] = LABELS[label]
        self.speckle_mask[mask] = speckle

    def _rows_mask(self, r0, r1):
        m = np.zeros((self.Nx, self.Ny), dtype=bool)
        m[:, max(0, r0):max(0, r1)] = True
        return m

    def _smooth2d(self, noise, sx, sz=None):
        """Gaussian smoothing with std sx (lateral) / sz (depth) in meters, via FFT (fast for big kernels)."""
        sz = sx if sz is None else sz
        F = ndimage.fourier_gaussian(np.fft.fft2(noise), sigma=(sx / self.dx, sz / self.dy))
        return np.fft.ifft2(F).real

    def _macro_field(self, tissue, which):
        """Spatially correlated texture on fat / gland: mean + zero-mean Gaussian field with SD from Li22 Table III and
        correlation length 0.21 mm (REF2 §4.1 [P]). Fat texture truncated at +/-0.9 sigma (Li22); the result is also clipped
        to the tissue's min-max of REF2 §3. Constant (mean) if macro_heterogeneity is disabled."""
        mean, _, lo, hi = ACOUSTIC_PROPS[tissue][which]
        if not self.entities['macro_heterogeneity']:
            return np.full((self.Nx, self.Ny), mean)
        rng = self._rng(f'macro_{tissue}_{which}')
        f = self._smooth2d(rng.standard_normal((self.Nx, self.Ny)), TEXTURE_CORR_LEN_M / np.sqrt(2.0))
        f = (f - f.mean()) / (f.std() + 1e-12)
        sd = TEXTURE_SD[(tissue, which)]
        tex = sd * f
        t = TEXTURE_TRUNC_SIGMA[tissue]
        if t is not None:
            tex = np.clip(tex, -t * sd, t * sd)
        return np.clip(mean + tex, lo, hi)

    def _ellipse_frame(self, cx, cz, reach):
        """Local window around (cx, cz) and the coordinates (xs, zs) of its pixels relative to the centre."""
        i0, i1 = max(0, int((cx - reach) / self.dx)), min(self.Nx, int((cx + reach) / self.dx) + 2)
        j0, j1 = max(0, int((cz - reach) / self.dy)), min(self.Ny, int((cz + reach) / self.dy) + 2)
        if i1 <= i0 or j1 <= j0:
            return None
        return (slice(i0, i1), slice(j0, j1)), (self.xx[i0:i1] - cx)[:, None], (self.zz[j0:j1] - cz)[None, :]

    def _ellipse_mask(self, cx, cz, a, b, theta):
        full = np.zeros((self.Nx, self.Ny), dtype=bool)
        fr = self._ellipse_frame(cx, cz, max(a, b) * 1.05)
        if fr is not None:
            sl, xs, zs = fr
            co, si = np.cos(theta), np.sin(theta)
            full[sl] = ((xs * co + zs * si) / a) ** 2 + ((-xs * si + zs * co) / b) ** 2 <= 1.0
        return full

    def _cooper_key(self):
        return 'cooper_fine' if max(self.dx, self.dy) <= 0.25e-3 else 'cooper_coarse'  # REF2 §3 / §3.1 #2

    # ------------------------------------------------------------------ geometry plan
    def _get_layout(self):
        """Random but reproducible geometry (meters) for a breast of depth D = Ny*dy (REF2 §2.2).
        skin(gel+epi+dermis) | subcutaneous fat t_sc = 0.10 D * U(0.6,1.5) | fibroglandular zone | retromammary fat 3-5 mm | z = D."""
        if self._layout is not None:
            return self._layout
        e, dy, D = self.entities, self.dy, self.depth
        rng = self._rng('layout')

        def nrows(t):
            return max(1, int(round(t / dy)))   # a layer is never thinner than 1 pixel

        t_gel = rng.uniform(*GEL_THICKNESS_M)
        t_epi = float(self._tn(rng, EPI_THICKNESS_M))
        t_der = float(self._tn(rng, DERMIS_THICKNESS_M))
        r = 0
        gel = (r, r + nrows(t_gel)) if e['gel'] else (r, r)
        r = gel[1]
        epi = (r, r + nrows(t_epi)) if e['epidermis'] else (r, r)
        r = epi[1]
        der = (r, r + nrows(t_der)) if e['dermis'] else (r, r)
        r = der[1]
        L = {'gel_rows': gel, 'epi_rows': epi, 'derm_rows': der, 'skin_rows': r, 'skin_bottom': r * dy}

        L['retro_t'] = rng.uniform(*RETRO_FAT_M)
        L['corpus_end'] = D - L['retro_t']                 # posterior end of fibroglandular zone; breast ends at D
        t_sc = SUBCUT_FRACTION_OF_D * D * rng.uniform(*SUBCUT_SCALE_RANGE)
        L['z_sub_end'] = min(L['skin_bottom'] + t_sc, L['corpus_end'] - MIN_CORPUS_M)
        if L['z_sub_end'] < L['skin_bottom']:
            raise ValueError("Breast too shallow for the chosen skin/fat layers; increase Ny*dy.")
        L['x_nipple'] = self.Nx * self.dx * rng.uniform(0.35, 0.65)   # duct convergence / gland cone axis [A]
        self._layout = L
        return L

    # ------------------------------------------------------------------ skin & background
    def addGel(self, var=None):
        """Coupling gel/water standoff 0.5-1.0 mm (REF2 §2.3 [L]); properties REF2 §3. (`var` kept for API compatibility.)"""
        if not self.entities['gel']:
            return self
        r0, r1 = self._get_layout()['gel_rows']
        c, rho = self._sample('water_gel', self._rng('gel'))
        self._paint(self._rows_mask(r0, r1), c, rho, 'gel', speckle=False)
        self.gel_thickness = r1 - r0
        return self

    def _skin_props(self, layer):
        """bulk skin: one sample shared by epidermis+dermis (Li22 [P]); layered: Moran 1995 sub-layers [L] (REF2 §3)."""
        if self.skin_model == 'bulk':
            return self._sample('skin', self._rng('skin'))
        return self._sample(layer, self._rng(layer))

    def addEpidermis(self, var=None):
        """Epidermis ~0.10 mm (Wang25 proxy, REF2 §2.2)."""
        if not self.entities['epidermis']:
            return self
        r0, r1 = self._get_layout()['epi_rows']
        c, rho = self._skin_props('epidermis')
        self._paint(self._rows_mask(r0, r1), c, rho, 'epidermis')
        self.epi_thickness = r1 - r0
        return self

    def addDermis(self, var=None):
        """Dermis 1.5-2.5 mm (Wang25 proxy 1.96 +/- 0.50 mm, REF2 §2.2)."""
        L = self._get_layout()
        self.total_skin_thickness = L['skin_rows']
        if not self.entities['dermis']:
            return self
        r0, r1 = L['derm_rows']
        c, rho = self._skin_props('dermis')
        self._paint(self._rows_mask(r0, r1), c, rho, 'dermis')
        return self

    def addSubstance(self, var=None):
        """Background fat from the skin to the posterior breast boundary (subcutaneous + retromammary fat), with the
        correlated texture of Li22 (REF2 §4.1). Other entities are painted on top."""
        L = self._get_layout()
        self.total_skin_thickness = L['skin_rows']
        mask = self._rows_mask(L['skin_rows'], self.Ny)
        self._paint(mask, self._macro_field('fat', 'c'), self._macro_field('fat', 'rho'), 'fat')
        return self

    addFatBackground = addSubstance

    # ------------------------------------------------------------------ breast corpus
    def addFibroglandular(self):
        """Fibroglandular parenchyma between the subcutaneous and retromammary fat (REF2 §2.3). The adipose fraction of
        that zone follows BI-RADS (v1 convention [L], REF2 §5). Morphology = thresholded smooth random field (lobule
        scale 4-8 mm) with a nipple-centred cone envelope [A: REF2 gives fractions/depths, not the exact morphology]."""
        if not self.entities['fibroglandular']:
            return self
        L, rng = self._get_layout(), self._rng('gland')
        j0 = int(round(L['z_sub_end'] / self.dy))
        j1 = min(self.Ny, int(round(L['corpus_end'] / self.dy)))
        if j1 - j0 < 2:
            return self
        zone = self._rows_mask(j0, j1)
        f = self._smooth2d(rng.standard_normal((self.Nx, self.Ny)), rng.uniform(4e-3, 8e-3))
        f = (f - f.mean()) / (f.std() + 1e-12)
        zrel = np.clip(self.zz[None, :] - L['z_sub_end'], 0, None)
        env = np.exp(-0.5 * ((self.xx[:, None] - L['x_nipple']) / (0.02 + 0.35 * zrel)) ** 2)
        f = f + env
        thr = np.quantile(f[zone], BIRADS_KFAT[self.birads])
        gland = zone & (f > thr)
        self.gland_mask = gland
        self._paint(gland, self._macro_field('gland', 'c'), self._macro_field('gland', 'rho'), 'fibroglandular')
        return self

    def addCooperLigaments(self):
        """Thin septa (0.2-0.5 mm) bounding oval fat lobules (5-15 mm), REF2 §2.3 [L]. Voronoi-edge construction [A];
        properties depend on the grid step (REF2 §3, §3.1 #2)."""
        if not self.entities['cooper_ligaments']:
            return self
        L, rng = self._get_layout(), self._rng('cooper')
        j0 = int(round(L['skin_bottom'] / self.dy))
        j1 = min(self.Ny, int(round(L['corpus_end'] / self.dy)))
        if j1 - j0 < 2:
            return self
        width, ex = self.Nx * self.dx, 1.5            # lobules elongated laterally [A]
        cell = rng.uniform(5e-3, 15e-3)
        z0, z1 = L['skin_bottom'] - cell, L['corpus_end'] + cell
        n = max(4, int((z1 - z0) * (width + 2 * cell) / cell ** 2))
        seeds = np.column_stack([rng.uniform(-cell, width + cell, n) / ex, rng.uniform(z0, z1, n)])
        px, pz = np.meshgrid(self.xx / ex, self.zz[j0:j1], indexing='ij')
        d, idx = cKDTree(seeds).query(np.column_stack([px.ravel(), pz.ravel()]), k=2)
        t = max(rng.uniform(0.2e-3, 0.5e-3), max(self.dx, self.dy))     # at least 1 pixel
        lo, hi = np.minimum(idx[:, 0], idx[:, 1]).astype(np.int64), np.maximum(idx[:, 0], idx[:, 1]).astype(np.int64)
        keep = (((lo * 73856093) ^ (hi * 19349663)) % 1000) < 700      # ~70 % of edges -> not a perfect honeycomb [A]
        edge = ((d[:, 1] - d[:, 0]) < t) & keep
        mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        mask[:, j0:j1] = edge.reshape(self.Nx, j1 - j0)
        mask &= ~self.gland_mask
        c, rho = self._sample(self._cooper_key(), rng)
        self._paint(mask, c, rho, 'cooper')
        return self

    def addDucts(self):
        """Lactiferous ducts, 0.5-2.0 mm, converging toward the nipple axis (REF2 §2.3 [L]); properties REF2 §3 (Pawlowska).
        Branching path = [A]."""
        if not self.entities['ducts'] or not self.gland_mask.any():
            return self
        L, rng = self._get_layout(), self._rng('ducts')
        canvas = np.zeros((self.Ny, self.Nx), dtype=np.uint8)   # cv uses [row = depth, col = lateral]
        cand = np.argwhere(self.gland_mask)
        z_top = L['skin_bottom'] + 0.5e-3

        def draw(p0, p1, diam):
            n = 10
            t = np.linspace(0, 1, n)[:, None]
            path = p0[None, :] * (1 - t) + p1[None, :] * t
            norm_ = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]])
            norm_ = norm_ / (np.linalg.norm(norm_) + 1e-12)
            path = path + norm_[None, :] * (rng.normal(0, 2e-3, (n, 1)) * np.sin(np.pi * t))
            pts = np.column_stack([path[:, 0] / self.dx, path[:, 1] / self.dy]).astype(np.int32)
            th = max(1, int(round(diam / min(self.dx, self.dy))))
            cv.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 255, th)
            return path

        for _ in range(int(rng.integers(2, 5))):
            ix, iz = cand[rng.integers(len(cand))]
            p0 = np.array([ix * self.dx, iz * self.dy])
            p1 = np.array([L['x_nipple'] + rng.normal(0, 3e-3), z_top])
            path = draw(p0, p1, rng.uniform(0.5e-3, 2.0e-3))
            for _ in range(int(rng.integers(0, 3))):                   # side branches [A]
                k = int(rng.integers(2, 8))
                pb = path[k] + np.array([rng.normal(0, 6e-3), rng.uniform(3e-3, 10e-3)])
                draw(path[k], pb, rng.uniform(0.5e-3, 1.0e-3))
        mask = canvas.T > 0
        mask &= self._rows_mask(int(round(L['skin_bottom'] / self.dy)), int(round(L['corpus_end'] / self.dy)))
        c, rho = self._sample('duct', rng)
        self._paint(mask, c, rho, 'duct')
        return self

    def addVessels(self, n_vessels=None):
        """3-8 vessel cross-sections, 0.5-2.5 mm (REF2 §2.3 [L]); wall [L] + blood lumen (Pawlowska, REF2 §3)."""
        if not self.entities['vessels']:
            return self
        L, rng = self._get_layout(), self._rng('vessels')
        n = int(rng.integers(3, 9)) if n_vessels is None else n_vessels
        z_lo, z_hi = L['skin_bottom'] + 1e-3, min(L['corpus_end'], self.depth) - 1e-3
        if z_hi <= z_lo:
            return self
        cw, rw = self._sample('vessel_wall', rng)
        cb, rb = self._sample('blood', rng)
        pix = max(self.dx, self.dy)
        for _ in range(n):
            a = max(rng.uniform(0.25e-3, 1.25e-3), 0.75 * pix)
            b = max(a * rng.uniform(0.6, 1.0), 0.75 * pix)
            cx, cz, th = rng.uniform(a, self.Nx * self.dx - a), rng.uniform(z_lo, z_hi), rng.uniform(0, np.pi)
            self._paint(self._ellipse_mask(cx, cz, a, b, th), cw, rw, 'vessel_wall')
            w = 0.2 * b                                     # wall = 20 % of radius [A]
            if b - w >= 0.5 * pix:
                self._paint(self._ellipse_mask(cx, cz, a - w, b - w, th), cb, rb, 'blood', speckle=False)
        return self

    def addChestWall(self):
        """Deprecated no-op: the chest wall was removed (REF2 §9). Kept so old chains still run."""
        warnings.warn("addChestWall() is a no-op: the phantom is breast-only (REF2 §9).")
        return self

    # ------------------------------------------------------------------ lesion shape generators
    def _draw_morphology(self, kind, rng):
        """(shape_class, margin_class) of a lesion. Malignant: shape from the chosen profile (REF2 §7.2), margin from the
        Gity21 mix restricted to margins compatible with the shape [A]. Benign: FA / cyst shape tables above."""
        if kind in ('idc', 'dcis'):
            shape_p = MALIGNANT_PROFILES[self.malignant_profile]['shape'] if kind == 'idc' else DCIS_SHAPE_PROBS
            shape = _choice(rng, shape_p)
            allowed = MARGINS_IRREGULAR if shape == 'irregular' else MARGINS_REGULAR
            base = IDC_MARGIN_PROBS if kind == 'idc' else DCIS_MARGIN_PROBS
            return shape, _choice(rng, {m: p for m, p in base.items() if m in allowed})
        if kind == 'fibroadenoma':
            shape = _choice(rng, FA_SHAPE_PROBS)
            return shape, ('lobulated' if shape == 'lobulated' else 'circumscribed')
        return _choice(rng, CYST_SHAPE_PROBS), 'circumscribed'

    def _depth_width_ratio(self, kind, cls, rng):
        """Depth(AP)/width ratio of the lesion. Round -> ~1; malignant: parallel with the profile's probability (REF2 §7.2)."""
        if cls == 'round':
            return float(self._tn(rng, DW_ROUND))
        if kind in ('idc', 'dcis'):
            par = rng.random() < MALIGNANT_PROFILES[self.malignant_profile]['p_parallel']
            return float(self._tn(rng, DW_PARALLEL if par else DW_NONPARALLEL))
        if kind == 'fibroadenoma':
            return float(self._tn(rng, DW_FIBROADENOMA))
        return float(self._tn(rng, DW_CYST_OVAL))

    def _contour_recipe(self, kind, cls, margin, rng):
        """Generator-A parameters (REF2 §7.4.1 / §7.4.3). Alpha ranges per class follow the solidity calibration of REF2 §7.4.4 [C]:
          smooth/oval alpha 0.02-0.06 K=6 (solidity 1.00) | lobulated alpha 0.15-0.20 p=0.5 K=5 (0.98-0.96) |
          irregular alpha 0.25-0.35 K=10 (0.96-0.93)      | spiculated alpha 0.10, 6-8 spicules, h 0.30-0.35, w 0.12 (0.93-0.88).
        Microlobulation (harmonics 12-20, alpha 0.025-0.05) and the angular polygon (5-8 vertices) are [A]; the lexicon terms
        (round/oval/lobulated/microlobulated/angular/irregular/spiculated) follow REF2b §3.2."""
        R = dict(name='smooth', alpha=0.0, K=2, p=1.0, micro=None, n_sp=0, h=0.0, w=0.12, polygon=0)
        if self.tumor_shape_model == 'ellipse':
            R['name'] = 'ellipse'                                   # alpha = 0 -> exact ellipse (test shape)
            return R
        if kind == 'cyst':                                           # REF2 §7.4.3: alpha ~0.02, K <= 4
            R.update(alpha=0.02, K=4)
        elif kind == 'fibroadenoma':                                 # REF2 §7.4.3: K <= 5, p = 0.5, alpha 0.02-0.20
            if margin == 'lobulated':
                R.update(name='lobulated', alpha=rng.uniform(0.15, 0.20), K=5, p=0.5)
            else:
                R.update(alpha=rng.uniform(0.02, 0.08), K=5, p=0.5)
        else:                                                        # IDC / DCIS, REF2 §7.4.3
            regular = cls != 'irregular'
            if margin in ('circumscribed', 'indistinct') and regular:
                R.update(alpha=rng.uniform(0.02, 0.06), K=6)         # blur only, contour stays smooth
            elif margin == 'indistinct':
                R.update(name='irregular', alpha=rng.uniform(0.25, 0.35), K=10)
            elif margin == 'angular':
                R.update(name='angular', polygon=int(rng.integers(5, 9)))
            elif margin == 'microlobulated':
                base = dict(alpha=rng.uniform(0.02, 0.06), K=6) if regular else dict(alpha=rng.uniform(0.15, 0.25), K=10)
                R.update(name='microlobulated', micro=(rng.uniform(0.025, 0.05), 12, 20), **base)
            elif margin == 'spiculated':
                R.update(name='spiculated', alpha=0.10, K=10, n_sp=int(rng.integers(6, 9)), h=rng.uniform(0.30, 0.35), w=0.12)
        return R

    def _contour_points(self, rng, R, a, b, n=720):
        """Closed contour in the lesion frame [m]:
            r(th) = 1 + sum_{k=2..K} alpha k^-p u_k cos(k th + phi_k) [+ micro harmonics] + sum_j h_j exp(-dth_j^2 / 2w^2)
            x = a r cos th,  z = b r sin th    (b/a = depth/width scaling = eta in REF2b §3.2)
        u_k ~ U(0.5,1), phi_k ~ U(0,2pi), h_j = h U(0.5,1), r >= 0.2.  REF2 §7.4.1 (radial perturbation idea: Gaussian random
        sphere, Sizikova23 / VanCamp23 [R]; exact parameterisation [A])."""
        if R['polygon']:                                             # angular: straight-edged polygon [A]
            m = R['polygon']
            ang = (np.arange(m) + rng.uniform(-0.3, 0.3, m)) * 2 * np.pi / m + rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0.7, 1.15, m)
            return np.column_stack([a * rad * np.cos(ang), b * rad * np.sin(ang)])
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = np.ones(n)
        for k in range(2, R['K'] + 1):
            r += R['alpha'] * k ** (-R['p']) * rng.uniform(0.5, 1.0) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
        if R['micro'] is not None:
            am, k0, k1 = R['micro']
            for k in range(k0, k1 + 1):
                r += am * rng.uniform(0.5, 1.0) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
        for _ in range(R['n_sp']):
            d = np.angle(np.exp(1j * (th - rng.uniform(0, 2 * np.pi))))      # wrapped angular distance
            r += R['h'] * rng.uniform(0.5, 1.0) * np.exp(-d ** 2 / (2 * R['w'] ** 2))
        r = np.maximum(r, 0.2)
        return np.column_stack([a * r * np.cos(th), b * r * np.sin(th)])

    @staticmethod
    def _polygon_solidity(pts):
        """area / convex-hull area of a closed contour (the 'concavity' of Pan24, REF2 §7.1)."""
        x, z = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
        try:
            return float(area / ConvexHull(pts).volume)
        except Exception:
            return None

    def _rasterize_polygon(self, pts_m):
        """Fill a polygon given in grid coordinates [m] (lateral, depth) into a full-size bool mask (sub-pixel accurate)."""
        full = np.zeros((self.Nx, self.Ny), dtype=bool)
        i0 = max(0, int(np.floor(pts_m[:, 0].min() / self.dx)) - 1)
        i1 = min(self.Nx, int(np.ceil(pts_m[:, 0].max() / self.dx)) + 2)
        j0 = max(0, int(np.floor(pts_m[:, 1].min() / self.dy)) - 1)
        j1 = min(self.Ny, int(np.ceil(pts_m[:, 1].max() / self.dy)) + 2)
        if i1 <= i0 or j1 <= j0:
            return full
        canvas = np.zeros((j1 - j0, i1 - i0), dtype=np.uint8)
        fixed = np.round(np.column_stack([pts_m[:, 0] / self.dx - i0, pts_m[:, 1] / self.dy - j0]) * 16).astype(np.int32)
        cv.fillPoly(canvas, [fixed.reshape(-1, 1, 2)], 1, lineType=cv.LINE_8, shift=4)
        full[i0:i1, j0:j1] = canvas.T.astype(bool)
        return full

    def _growth_mask(self, rng, cx, cz, a, b, theta, reach, rows):
        """Generator B (REF2 §7.4.2 [A], simplification of Sengupta21/24 and Lago&Badano25): grow a lesion pixel-batch by
        pixel-batch from a seed; p(c) ~ 1/S(c)^gamma * (1 + beta * occupied_neighbours) * soft ellipse envelope, so it bends
        around stiff Cooper septa and spreads through soft tissue. Then Gaussian smoothing (sigma = 1 px) and threshold 0.5."""
        full = np.zeros((self.Nx, self.Ny), dtype=bool)
        fr = self._ellipse_frame(cx, cz, reach)
        if fr is None:
            return full
        sl, xs, zs = fr
        lab = self.label_map[sl]
        inv = {v: k for k, v in LABELS.items()}
        S = np.full(lab.shape, 1.0)
        for v in np.unique(lab):
            S[lab == v] = GROWTH_STIFFNESS.get(inv.get(int(v), 'fat'), 1.0)
        co, si = np.cos(theta), np.sin(theta)
        rn = np.sqrt(((xs * co + zs * si) / a) ** 2 + ((-xs * si + zs * co) / b) ** 2)
        env = np.exp(-0.5 * (rn / 0.9) ** 2) + 1e-6                   # soft envelope keeps the target aspect ratio [A]
        ok = rows[sl]
        target = max(3, int(np.pi * a * b / (self.dx * self.dy)))
        mask = np.zeros(lab.shape, dtype=bool)
        seed = (int(np.argmin(np.abs(xs[:, 0]))), int(np.argmin(np.abs(zs[0, :]))))
        mask[seed] = True
        n, kern = 1, np.ones((3, 3), dtype=np.float32)
        kern[1, 1] = 0
        while n < target:
            nb = ndimage.convolve(mask.astype(np.float32), kern, mode='constant')
            front = (~mask) & (nb > 0) & ok
            ids = np.flatnonzero(front.ravel())
            if ids.size == 0:
                break
            w = (env / S ** GROWTH_GAMMA * (1 + GROWTH_BETA * nb)).ravel()[ids]
            batch = int(max(1, min(target - n, int(0.03 * target) + 1, ids.size)))
            pick = rng.choice(ids, size=batch, replace=False, p=w / w.sum())
            mask.flat[pick] = True
            n += batch
        m = ndimage.gaussian_filter(mask.astype(float), 1.0) > 0.5
        m = ndimage.binary_fill_holes(m)
        lbl, k = ndimage.label(m)
        if k > 1:
            m = lbl == (1 + int(np.argmax(ndimage.sum(m, lbl, range(1, k + 1)))))
        full[sl] = m
        return full

    def _halo_mask(self, core, cx, cz, rng, rows):
        """Hyperechoic peritumoral rim of an IDC (REF2 §3 table; §7.5 [L]/[A]): a stochastic 1.5-2.5 mm band whose width varies
        around the contour (so it follows the spicules/lobules of the generated shape, REF2 §7.5)."""
        ii, jj = np.nonzero(core.any(1))[0], np.nonzero(core.any(0))[0]
        pad_i, pad_j = int(6e-3 / self.dx) + 1, int(6e-3 / self.dy) + 1
        sl = (slice(max(0, ii.min() - pad_i), min(self.Nx, ii.max() + pad_i + 1)),
              slice(max(0, jj.min() - pad_j), min(self.Ny, jj.max() + pad_j + 1)))
        cw = core[sl]
        dist = ndimage.distance_transform_edt(~cw, sampling=(self.dx, self.dy))
        th = np.arctan2((self.zz[sl[1]] - cz)[None, :], (self.xx[sl[0]] - cx)[:, None])
        width, amp = rng.uniform(1.5e-3, 2.5e-3), rng.uniform(0.5e-3, 1.5e-3)     # [L] v1 values
        ks = np.arange(3, 10)
        delta = sum(np.cos(k * th + rng.uniform(0, 2 * np.pi)) for k in ks) * amp / np.sqrt(len(ks) / 2)
        extent = np.maximum(width + delta, 0.3e-3)
        rim = (dist > 0) & (dist <= extent)
        rim &= rng.random(rim.shape) < np.clip(1 - dist / extent, 0, 1)
        rim &= rows[sl] & ~cw & ~np.isin(self.label_map[sl], TUMOR_LABELS + [LABELS['microcalc']])
        full = np.zeros_like(core)
        full[sl] = rim
        return full

    # ------------------------------------------------------------------ lesions
    def addTumors(self):
        """1..max_num_tumors lesions of the enabled types (idc, dcis, fibroadenoma, cyst) with NON-ELLIPTICAL contours.
          size      : truncated log-normal largest dimension per type / cohort           (LESION_SIZE_MM, REF2 §6.2)
          morphology: shape class + margin class + depth/width ratio                      (REF2 §7.2)
          contour   : Fourier-radial (+ micro-lobules / spicules / angular polygon), or stiffness-biased growth (REF2 §7.4)
          placement : FGI / inside gland / inside fat per type (Littrup21) weighted by tumour-to-skin and tumour-to-nipple priors
                      (PMC10965701); lesion contour must lie entirely between skin (+margin) and the retromammary fat (REF2 §2.2,
                      REF2b §3.3 items 4-5). Lesion depth is NOT rescaled with breast depth D (REF2 §2.2, §6.3).
          IDC gets a halo (REF2 §7.5); DCIS gets microcalcifications with p = 0.8 (REF2 §7.6).
        The generated contour and its parameters are stored in self.lesions (REF2b §3.3 item 7)."""
        e = self.entities
        kinds = [k for k in TUMOR_WEIGHTS if e['tumor_' + k]]
        if self.max_num_tumors < 1 or not kinds:
            return self
        L, rng = self._get_layout(), self._rng('tumors')
        n_t = int(rng.integers(1, self.max_num_tumors + 1))
        w = np.array([TUMOR_WEIGHTS[k] for k in kinds], dtype=float)
        w /= w.sum()
        z_lo, z_hi = L['skin_bottom'] + LESION_MARGIN_M, min(L['corpus_end'], self.depth) - LESION_MARGIN_M
        if z_hi - z_lo < 2e-3:
            return self
        shape = (self.Nx, self.Ny)
        width = self.Nx * self.dx
        if self.gland_mask.any():
            d_out = ndimage.distance_transform_edt(~self.gland_mask, sampling=(self.dx, self.dy))
            d_in = ndimage.distance_transform_edt(self.gland_mask, sampling=(self.dx, self.dy))
        else:
            d_out, d_in = np.full(shape, np.inf), np.zeros(shape)
        sdist = np.where(self.gland_mask, -d_in, d_out)       # signed distance to the gland boundary (<0 inside)
        # rows where a lesion (and its halo) may exist: skin bottom -> start of retromammary fat
        zone_rows = self._rows_mask(int(round(L['skin_bottom'] / self.dy)), int(round(min(L['corpus_end'], self.depth) / self.dy)))
        pix = max(self.dx, self.dy)
        sizes = LESION_SIZE_MM[self.lesion_cohort]

        for _ in range(n_t):
            kind = kinds[int(rng.choice(len(kinds), p=w))]
            # ---- size (largest dimension), morphology, aspect ratio
            med, sig, lo, hi = sizes[kind]
            diam = float(np.exp(self._tn(rng, (np.log(med), sig, np.log(lo), np.log(hi))))) * 1e-3
            cls, margin = self._draw_morphology(kind, rng)
            dw = self._depth_width_ratio(kind, cls, rng)
            if dw <= 1.0:
                a = diam / 2; b = a * dw                      # wider than tall: width is the largest dimension
            else:
                b = diam / 2; a = b / dw
            a, b = max(a, 1.5 * pix), max(b, 1.5 * pix)
            theta = rng.normal(0, np.radians(10))             # small tilt [A]
            co, si = np.cos(theta), np.sin(theta)

            # ---- contour (lesion frame -> rotated by theta)
            use_growth = (self.tumor_shape_model == 'growth' and kind in ('idc', 'dcis'))
            recipe, poly = None, None
            if use_growth:
                hx = 1.3 * np.hypot(a * co, b * si)           # growth envelope extent [A]
                hz = 1.3 * np.hypot(a * si, b * co)
            else:
                recipe = self._contour_recipe(kind, cls, margin, rng)
                poly = self._contour_points(rng, recipe, a, b)
                poly = np.column_stack([poly[:, 0] * co - poly[:, 1] * si, poly[:, 0] * si + poly[:, 1] * co])
                hx, hz = float(np.abs(poly[:, 0]).max()), float(np.abs(poly[:, 1]).max())
            s = min(1.0, 0.98 * (z_hi - z_lo) / (2 * hz), 0.98 * width / (2 * hx))
            if s < 1.0:                                        # lesion must fit in the corpus (REF2b §3.3 item 5)
                a, b, hx, hz = a * s, b * s, hx * s, hz * s
                poly = poly * s if poly is not None else None

            # ---- placement: FGI / gland / fat class (Littrup21), then distance priors (PMC10965701)
            cand = ((self.xx[:, None] >= hx) & (self.xx[:, None] <= width - hx) &
                    (self.zz[None, :] >= z_lo + hz) & (self.zz[None, :] <= z_hi - hz))
            r_mean = np.sqrt(a * b)    # "at FGI" = |signed distance| <= mean lesion radius [A]
            pools = {'fgi': cand & (np.abs(sdist) <= r_mean),
                     'gland': cand & (sdist < -r_mean),
                     'fat': cand & (sdist > r_mean)}
            order = ['fgi', 'gland', 'fat']
            first = order[int(rng.choice(3, p=PLACEMENT[kind]))]
            zone = next((o for o in [first] + [o for o in order if o != first] if pools[o].any()), None)
            pool = pools[zone] if zone else cand
            if not pool.any():
                continue
            idx = np.argwhere(pool)
            if e['lesion_distance_priors']:
                m1, s1 = TUMOR_TO_SKIN_MM
                m2, s2 = TUMOR_TO_NIPPLE_MM
                d_top = np.maximum((idx[:, 1] * self.dy - hz - L['skin_bottom']) * 1e3, 0.5)   # mm, skin -> top margin
                d_nip = np.maximum(np.abs(idx[:, 0] * self.dx - L['x_nipple']) * 1e3, 0.5)     # mm
                wt = (norm.pdf(d_top, m1, s1) *                                                 # left-truncated normal (REF2 §6.3)
                      gamma.pdf(d_nip, (m2 / s2) ** 2, scale=s2 ** 2 / m2)) + 1e-12             # moment-matched gamma [A]
                wt /= wt.sum()
                ix, iz = idx[int(rng.choice(len(idx), p=wt))]
            else:
                ix, iz = idx[rng.integers(len(idx))]
            cx, cz = ix * self.dx, iz * self.dy

            # ---- rasterise
            if use_growth:
                core = self._growth_mask(rng, cx, cz, a, b, theta, max(hx, hz) + 2 * pix, zone_rows)
                recipe_rec = {'name': 'growth', 'gamma': GROWTH_GAMMA, 'beta': GROWTH_BETA}
                solidity, contour = None, None
            else:
                core = self._rasterize_polygon(poly + np.array([cx, cz]))
                recipe_rec = dict(recipe)
                solidity, contour = self._polygon_solidity(poly), poly.copy()
            core &= zone_rows
            if not core.any():
                continue

            c, rho = self._sample(kind, rng)
            if kind == 'idc' and self.tumor_density_model == 'solid':
                rho = float(rng.uniform(*IDC_RHO_SOLID_RANGE))
            self._paint(core, c, rho, kind, speckle=(kind != 'cyst'))   # cysts stay anechoic (REF2 §4.2 / v1)

            if kind == 'idc' and e['peritumoral_halo']:
                self._paint(self._halo_mask(core, cx, cz, rng, zone_rows), HALO_C, HALO_RHO, 'halo')

            has_calc = False
            if (kind == 'dcis' and e['microcalcifications'] and max(self.dx, self.dy) <= CALC_MAX_PIXEL_M
                    and rng.random() < CALC_PROB_DCIS):
                has_calc = True
                pts = np.argwhere(core)
                if rng.random() < CALC_CLUSTER_PROB:        # compact cluster
                    c0 = pts[rng.integers(len(pts))]
                    near = (((pts[:, 0] - c0[0]) * self.dx) ** 2 + ((pts[:, 1] - c0[1]) * self.dy) ** 2
                            <= (0.35 * min(a, b)) ** 2)
                    pts = pts[near]
                calc = np.zeros(shape, dtype=bool)
                for _ in range(int(rng.integers(5, 16))):   # number of foci [A]
                    px, pz = pts[rng.integers(len(pts))]
                    sz = max(1, int(round(rng.uniform(*CALC_SIZE_M) / pix)))   # 0.1-1 mm in pixels (REF2 §7.6)
                    calc[px:px + sz, pz:pz + sz] = True
                self._paint(calc & core, MICROCALC_C, MICROCALC_RHO, 'microcalc', speckle=False)

            ii, jj = np.nonzero(core.any(1))[0], np.nonzero(core.any(0))[0]
            self.there_is_a_tumor = True
            self.lesions.append({'type': kind, 'shape_class': cls, 'margin': margin, 'contour_type': recipe_rec['name'],
                                 'placement': zone, 'center_m': (cx, cz), 'semi_axes_m': (a, b),
                                 'diameter_m': max(2 * a, 2 * b), 'depth_width_ratio': b / a, 'angle_rad': float(theta),
                                 'extent_m': ((ii.max() - ii.min() + 1) * self.dx, (jj.max() - jj.min() + 1) * self.dy),
                                 'area_m2': float(core.sum() * self.dx * self.dy), 'solidity': solidity,
                                 'contour_m': contour, 'contour_params': recipe_rec,
                                 'tumor_to_skin_m': float(jj.min() * self.dy - L['skin_bottom']),
                                 'c': c, 'rho': rho, 'microcalcifications': has_calc})
        return self

    # ------------------------------------------------------------------ speckle
    def applySpeckle(self):
        """Micro-scale sub-wavelength scatterers (pulse-echo only): ~2 reflectors per lambda^2, +/-3 % density perturbation in
        every pixel holding >= 1 scatterer. Feigin 2019 / Treeby 2017 via v1 [L] (REF2 §4.2, not re-verified). Not applied to
        gel, cysts, blood lumen or microcalcifications."""
        if not self.entities['speckle']:
            return self
        rng = self._rng('speckle')
        lam = 1540.0 / self.center_frequency
        has = rng.poisson(2.0 * self.dx * self.dy / lam ** 2, (self.Nx, self.Ny)) > 0
        pert = 0.03 * rng.uniform(-1, 1, (self.Nx, self.Ny)) * has
        m = self.speckle_mask
        self.rho_map[m] *= 1 + pert[m]
        return self

    # ------------------------------------------------------------------ driver & getters
    def build(self):
        """Regenerate all maps from scratch with the currently enabled entities (correct order)."""
        self._reset()
        return (self.addGel().addEpidermis().addDermis().addSubstance().addFibroglandular()
                .addCooperLigaments().addDucts().addVessels().addTumors().applySpeckle())

    def mean_breast_sos(self):
        """Area-mean SOS of the breast (everything except gel). Compare in ORDER/SPACING (not absolute value) with
        BEZEK25_GLOBAL_SOS[self.birads] (REF2 §5)."""
        m = self.label_map > LABELS['gel']
        return float(self.sos_map[m].mean()) if m.any() else float('nan')

    def birads_sos_target(self):
        """(mean, sd) of the measured global breast SOS for this BI-RADS class, Bezek25 (REF2 §5)."""
        return BEZEK25_GLOBAL_SOS[self.birads]

    def get_sos(self):
        return self.sos_map.copy()

    def get_rho(self):
        return self.rho_map.copy()

    def get_labels(self):
        """int8 tissue label map; see LABELS."""
        return self.label_map.copy()

    def get_tumor_mask(self):
        return np.isin(self.label_map, TUMOR_LABELS)

    def get_lesions(self):
        return list(self.lesions)

    def isThereATomur(self):
        return self.there_is_a_tumor

    has_tumor = isThereATomur