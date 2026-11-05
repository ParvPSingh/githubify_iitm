"""Builds data/demo_graph.json: a small hand-curated evidence graph for the demo.

The statements are well-established facts paraphrased from the named sources,
written so a reader can see what each hop claims. They are for demonstrating
the pipeline; check the primary source before citing any of them in a paper.

It deliberately includes noise found in real literature graphs:
* a hub node ("patients") with many weak co-occurrence edges
* an analogical edge (cancer ~ a car with a stuck accelerator and failed brakes)
so the ranking and filtering can be seen working.
"""
import json
from pathlib import Path

N = lambda i, name, t, d, *al: {"id": i, "name": name, "type": t, "domain": d, "aliases": list(al)}  # noqa: E731

nodes = [
    N("cancer", "cancer", "Disease", "oncology", "malignancy", "tumour", "tumor"),
    N("lung_cancer", "lung cancer", "Disease", "oncology"),
    N("mesothelioma", "mesothelioma", "Disease", "oncology"),
    N("aml", "acute myeloid leukaemia", "Disease", "oncology", "AML", "acute myeloid leukemia"),
    N("melanoma", "melanoma", "Disease", "oncology"),
    N("testicular_cancer", "testicular cancer", "Disease", "oncology", "testicular germ-cell tumours"),
    N("chemotherapy", "chemotherapy", "Treatment", "oncology", "chemotherapeutic"),
    N("cisplatin", "cisplatin", "Chemical", "pharmacology"),
    N("immunosuppression", "immunosuppression", "Condition", "immunology", "weakened immune system", "immunocompromised"),
    N("chickenpox", "chickenpox", "Disease", "infectious disease", "varicella"),
    N("vzv", "varicella-zoster virus", "Virus", "virology", "VZV", "human alphaherpesvirus 3"),
    N("herpes_zoster", "herpes zoster", "Disease", "infectious disease", "shingles", "zoster"),
    N("alphaherpes", "Alphaherpesvirinae", "Taxon", "virology", "alphaherpesvirus", "alphaherpesviruses"),
    N("hsv1", "herpes simplex virus 1", "Virus", "virology", "HSV-1", "human alphaherpesvirus 1"),
    N("tvec", "talimogene laherparepvec", "Treatment", "oncology", "T-VEC"),
    N("rzv", "recombinant zoster vaccine", "Vaccine", "immunology", "RZV"),
    N("automobile", "automobile", "Product", "automotive", "car", "motor vehicle", "vehicle"),
    N("diesel_engine", "diesel engine", "Component", "automotive", "diesel engines"),
    N("diesel_exhaust", "diesel engine exhaust", "Chemical mixture", "environmental health", "diesel exhaust"),
    N("brake_lining", "brake lining", "Component", "automotive", "brake linings", "brakes"),
    N("asbestos", "asbestos", "Material", "materials"),
    N("gasoline", "gasoline", "Fuel", "automotive", "petrol"),
    N("benzene", "benzene", "Chemical", "chemistry"),
    N("catalytic_converter", "catalytic converter", "Component", "automotive", "catalytic converters", "autocatalyst"),
    N("platinum", "platinum", "Element", "materials", "platinum-group metals", "Pt"),
    N("patients", "patients", "Generic", "general", "patient"),
]


def E(src, dst, rel, kind, *ev):
    return {"src": src, "dst": dst, "relation": rel, "kind": kind,
            "evidence": [{"text": t, "source": s, "year": y, "url": None} for t, s, y in ev]}


ISA = "Disease hierarchy (MeSH / Disease Ontology)"
edges = [
    # ontology
    E("lung_cancer", "cancer", "is a", "mechanistic", ("Lung cancer is a cancer arising in lung tissue.", ISA, 1960)),
    E("mesothelioma", "cancer", "is a", "mechanistic", ("Mesothelioma is a cancer of the mesothelium.", ISA, 1960)),
    E("aml", "cancer", "is a", "mechanistic", ("Acute myeloid leukaemia is a cancer of the myeloid blood-cell line.", ISA, 1960)),
    E("melanoma", "cancer", "is a", "mechanistic", ("Melanoma is a cancer of melanocytes.", ISA, 1960)),
    E("testicular_cancer", "cancer", "is a", "mechanistic", ("Testicular cancer is a cancer of the testis, mostly germ-cell tumours.", ISA, 1960)),

    # automobile -> diesel exhaust -> lung cancer
    E("automobile", "diesel_engine", "may be powered by", "mechanistic",
      ("Many cars, vans and trucks — motor vehicle classes — are powered by a diesel engine.", "Automotive engineering reference", 1936)),
    E("diesel_engine", "diesel_exhaust", "emits", "causal",
      ("A diesel engine emits diesel engine exhaust, a mixture of gases and soot particles.", "IARC Monographs Vol. 105 (2014)", 2012)),
    E("diesel_exhaust", "lung_cancer", "causes", "causal",
      ("IARC classified diesel engine exhaust as carcinogenic to humans (Group 1), based on sufficient evidence that it causes lung cancer.",
       "IARC Monographs Vol. 105; IARC press release No. 213 (June 2012)", 2012)),

    # automobile -> brake lining -> asbestos -> mesothelioma
    E("automobile", "brake_lining", "has part", "mechanistic",
      ("Every automobile has brakes; drum and disc brakes use friction material such as a brake lining.", "Automotive engineering reference", 1920)),
    E("brake_lining", "asbestos", "historically contained", "mechanistic",
      ("Asbestos was widely used in vehicle brake linings and clutch facings.", "US EPA, asbestos in automotive brakes and clutches", 1986)),
    E("asbestos", "mesothelioma", "causes", "causal",
      ("Asbestos (all forms) causes mesothelioma and cancer of the lung, larynx and ovary.", "IARC Monographs Vol. 100C (2012)", 2012),
      ("Diffuse pleural mesothelioma was linked to asbestos exposure in the North Western Cape Province.", "Wagner, Sleggs & Marchand, Br J Ind Med 1960", 1960)),

    # automobile -> gasoline -> benzene -> AML
    E("automobile", "gasoline", "is fuelled by", "mechanistic",
      ("Most automobiles are fuelled by gasoline.", "Automotive engineering reference", 1900)),
    E("gasoline", "benzene", "contains", "mechanistic",
      ("US rules limit benzene in gasoline to an annual average of 0.62% by volume.", "US EPA Mobile Source Air Toxics rule (2007)", 2007)),
    E("benzene", "aml", "causes", "causal",
      ("There is sufficient evidence in humans that benzene causes acute myeloid leukaemia.", "IARC Monographs Vol. 120 (2018)", 2018)),

    # automobile -> catalytic converter -> platinum -> cisplatin -> testicular cancer
    E("automobile", "catalytic_converter", "has part", "mechanistic",
      ("Gasoline automobiles sold in the US since 1975 carry a catalytic converter.", "US emission standards history", 1975)),
    E("catalytic_converter", "platinum", "contains", "mechanistic",
      ("A catalytic converter uses platinum-group metals (platinum, palladium, rhodium) as the catalyst.", "Automotive catalysis literature", 1975)),
    E("platinum", "cisplatin", "is the active metal in", "mechanistic",
      ("Platinum electrode electrolysis products inhibited cell division in E. coli, leading to the discovery of cisplatin.",
       "Rosenberg, Van Camp & Krigas, Nature 1965", 1965)),
    E("cisplatin", "testicular_cancer", "treats", "causal",
      ("Cisplatin-based combination chemotherapy produced durable complete remissions in disseminated testicular cancer.",
       "Einhorn & Donohue, Ann Intern Med 1977", 1977)),
    E("cisplatin", "chemotherapy", "is a", "mechanistic",
      ("Cisplatin is a platinum-based chemotherapy drug.", "Pharmacology reference", 1978)),

    # cancer -> chemotherapy -> immunosuppression -> herpes zoster -> VZV -> chickenpox
    E("chemotherapy", "cancer", "treats", "causal",
      ("Chemotherapy is a standard treatment for many types of cancer.", "Oncology reference", 1950)),
    E("chemotherapy", "immunosuppression", "can cause", "causal",
      ("Chemotherapy can cause immunosuppression by depleting white blood cells.", "Oncology reference", 1960)),
    E("immunosuppression", "herpes_zoster", "increases risk of", "associative",
      ("People who are immunocompromised have a higher risk of herpes zoster (shingles).", "CDC, Shingles clinical overview", 2008)),
    E("cancer", "herpes_zoster", "increases risk of", "associative",
      ("People with cancer, especially blood cancers, are at increased risk of shingles because of a weakened immune system.",
       "CDC, Shingles clinical overview", 2008)),
    E("vzv", "herpes_zoster", "reactivates as", "causal",
      ("After chickenpox, varicella-zoster virus stays latent in sensory ganglia and can reactivate as herpes zoster.",
       "Gershon et al., Nat Rev Dis Primers 2015", 2015),
      ("Herpes zoster was proposed to be a reactivation of latent varicella-zoster virus acquired during chickenpox.",
       "Hope-Simpson, Proc R Soc Med 1965", 1965)),
    E("vzv", "chickenpox", "causes", "causal",
      ("Chickenpox is caused by primary infection with varicella-zoster virus.", "CDC, Chickenpox clinical overview", 1953)),
    E("rzv", "herpes_zoster", "prevents", "causal",
      ("ACIP recommends recombinant zoster vaccine to prevent herpes zoster in immunocompromised adults aged 19 and older.",
       "Anderson et al., MMWR 2022", 2022)),
    E("rzv", "immunosuppression", "recommended in", "associative",
      ("Recombinant zoster vaccine is recommended for adults who are immunocompromised, for example from cancer treatment.",
       "Anderson et al., MMWR 2022", 2022)),

    # chickenpox virus family -> oncolytic HSV-1 -> melanoma
    E("vzv", "alphaherpes", "member of", "mechanistic",
      ("Varicella-zoster virus (human alphaherpesvirus 3) belongs to the subfamily Alphaherpesvirinae.", "ICTV taxonomy", 1979)),
    E("hsv1", "alphaherpes", "member of", "mechanistic",
      ("Herpes simplex virus 1 (human alphaherpesvirus 1) belongs to the subfamily Alphaherpesvirinae.", "ICTV taxonomy", 1979)),
    E("tvec", "hsv1", "derived from", "mechanistic",
      ("Talimogene laherparepvec is an oncolytic virus derived from herpes simplex virus 1.", "Andtbacka et al., J Clin Oncol 2015", 2015)),
    E("tvec", "melanoma", "treats", "causal",
      ("Talimogene laherparepvec improved durable response rate in advanced melanoma (OPTiM trial).", "Andtbacka et al., J Clin Oncol 2015", 2015)),

    # noise: a hub with weak co-occurrence edges
    *[E("patients", x, "co-mentioned with", "cooccurrence", (f"Patients with {label} were included in the study.", "co-occurrence sample", 2000))
      for x, label in [("cancer", "cancer"), ("chickenpox", "chickenpox"), ("lung_cancer", "lung cancer"),
                       ("melanoma", "melanoma"), ("herpes_zoster", "herpes zoster"), ("aml", "AML"),
                       ("mesothelioma", "mesothelioma"), ("chemotherapy", "chemotherapy")]],
    E("patients", "automobile", "co-mentioned with", "cooccurrence",
      ("Patients injured in automobile collisions were admitted to the trauma unit.", "co-occurrence sample", 2000)),

    # analogy: not evidence, excluded unless asked for
    E("cancer", "automobile", "is like", "analogical",
      ("Cancer is often explained as a car with the accelerator stuck (oncogenes) and the brakes broken (tumour suppressors).",
       "Common teaching analogy", 2000)),
]

out = Path(__file__).with_name("demo_graph.json")
out.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2, ensure_ascii=False))
print(f"wrote {out} ({len(nodes)} nodes, {len(edges)} edges)")
