import streamlit as st
import fitz
import docx
import re
import os
import json
from datetime import datetime
import spacy
from langdetect import detect
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

# ---------------- File Reading ----------------
def read_file(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([p.get_text() for p in doc])
    if file.name.endswith(".docx"):
        d = docx.Document(file)
        return "\n".join([p.text for p in d.paragraphs])
    return file.read().decode("utf-8")

# ---------------- Hindi Normalization ----------------
def normalize_text(text):
    try:
        if detect(text) == "hi":
            repl = {
                "समाप्त": "terminate",
                "स्वतः नवीनीकरण": "automatically renew",
                "क्षतिपूर्ति": "indemnity",
                "गोपनीय": "confidential",
                "अधिकार": "rights"
            }
            for k in repl:
                text = text.replace(k, repl[k])
    except:
        pass
    return text

# ---------------- Clause & Subclause Split ----------------
def split_clauses(text):
    parts = re.split(r'\n\s*\n', text)
    clauses = [p.strip() for p in parts if len(p.strip()) > 40]
    return clauses

# ---------------- Contract Type ----------------
def classify_contract(text):
    t = text.lower()
    if "employee" in t or "salary" in t:
        return "Employment Agreement"
    if "lease" in t or "rent" in t:
        return "Lease Agreement"
    if "vendor" in t or "supply" in t:
        return "Vendor Contract"
    if "partner" in t:
        return "Partnership Agreement"
    return "Service Agreement"

# ---------------- Entity Extraction ----------------
def extract_entities(text):
    doc = nlp(text)
    parties = [e.text for e in doc.ents if e.label_ == "ORG"]
    dates = [e.text for e in doc.ents if e.label_ == "DATE"]
    money = [e.text for e in doc.ents if e.label_ == "MONEY"]
    return list(set(parties)), list(set(dates)), list(set(money))

# ---------------- Obligation / Right / Prohibition ----------------
def classify_intent(clause):
    c = clause.lower()
    if "shall not" in c or "must not" in c or "prohibited" in c:
        return "Prohibition"
    if "shall" in c or "must" in c:
        return "Obligation"
    if "may" in c or "entitled to" in c:
        return "Right"
    return "Neutral"

# ---------------- Ambiguity Detection ----------------
def detect_ambiguity(clause):
    vague = ["reasonable", "as soon as possible", "etc", "as appropriate", "from time to time"]
    found = [w for w in vague if w in clause.lower()]
    return found

# ---------------- Risk Detection ----------------
def detect_risks(clause):
    c = clause.lower()
    risks = []
    if "indemn" in c:
        risks.append("Indemnity")
    if "penalty" in c or "fine" in c:
        risks.append("Penalty")
    if "terminate at any time" in c or "without cause" in c:
        risks.append("Unilateral Termination")
    if "arbitration" in c or "jurisdiction" in c:
        risks.append("Arbitration/Jurisdiction")
    if "automatically renew" in c:
        risks.append("Auto Renewal")
    if "non-compete" in c:
        risks.append("Non Compete")
    if "intellectual property" in c and "assign" in c:
        risks.append("IP Transfer")
    if "lock-in" in c or "cannot terminate before" in c:
        risks.append("Lock-in Period")
    return risks

def risk_level(risks):
    if len(risks) == 0:
        return "Low"
    if len(risks) == 1:
        return "Medium"
    return "High"

# ---------------- Plain Explanation ----------------
def explain_risk(r):
    m = {
        "Indemnity":"You must cover losses of the other party.",
        "Penalty":"Financial punishment if terms are broken.",
        "Unilateral Termination":"Other side can end contract anytime.",
        "Arbitration/Jurisdiction":"Disputes forced to specific court.",
        "Auto Renewal":"Renews automatically if not stopped.",
        "Non Compete":"You cannot work with competitors.",
        "IP Transfer":"You lose ownership of your work.",
        "Lock-in Period":"You cannot exit before fixed time."
    }
    return m.get(r, "")

def safer_alt(r):
    m = {
        "Indemnity":"Limit only to direct proven damages.",
        "Penalty":"Add reasonable cap on penalties.",
        "Unilateral Termination":"Require equal notice from both sides.",
        "Arbitration/Jurisdiction":"Use mutually agreed neutral venue.",
        "Auto Renewal":"Require written consent before renewal.",
        "Non Compete":"Limit scope and duration.",
        "IP Transfer":"Use license instead of ownership transfer.",
        "Lock-in Period":"Allow early exit with fair notice."
    }
    return m.get(r, "")

# ---------------- Template Clauses for Similarity ----------------
templates = [
"Either party may terminate with 30 days written notice.",
"Liability is limited to direct damages only.",
"Creator retains intellectual property ownership.",
"Disputes will be resolved by mutual arbitration.",
"Contract will not auto-renew without consent."
]

vectorizer = TfidfVectorizer()

def similarity_score(clause):
    docs = templates + [clause]
    tfidf = vectorizer.fit_transform(docs)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1])
    return max(sims[0])

# ---------------- PDF Export ----------------
def create_pdf(text, path):
    c = canvas.Canvas(path, pagesize=letter)
    y = 750
    for line in text.split("\n"):
        c.drawString(40, y, line[:95])
        y -= 15
        if y < 50:
            c.showPage()
            y = 750
    c.save()

# ---------------- Audit Log ----------------
def save_log(data):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    name = "logs/log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    with open(name, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- UI ----------------
st.title("AI Contract Risk Analyzer")

file = st.file_uploader("Upload Contract (PDF/DOCX/TXT)")

if file and st.button("Analyze Contract"):

    text = normalize_text(read_file(file))
    clauses = split_clauses(text)

    st.write("Contract Type:", classify_contract(text))

    parties, dates, money = extract_entities(text)
    st.write("Parties:", parties)
    st.write("Dates:", dates)
    st.write("Amounts:", money)

    report = "CONTRACT SUMMARY\n"
    high_count = 0

    for i, clause in enumerate(clauses):

        intent = classify_intent(clause)
        risks = detect_risks(clause)
        level = risk_level(risks)
        vague = detect_ambiguity(clause)
        sim = similarity_score(clause)

        if level == "High":
            high_count += 1

        st.markdown("### Clause " + str(i+1))
        st.write("Intent:", intent)
        st.write("Risk Level:", level)
        st.write("Risks:", risks if risks else "None")

        if vague:
            st.write("Ambiguity Found:", vague)

        if risks:
            for r in risks:
                st.write("Explanation:", explain_risk(r))
                st.write("Safer Alternative:", safer_alt(r))

        st.write("Template Similarity Score:", round(sim,2))

        report += "Clause " + str(i+1) + " | " + level + " | " + ",".join(risks) + "\n"

    overall = "Low"
    if high_count > 2:
        overall = "High"
    elif high_count > 0:
        overall = "Medium"

    st.subheader("Overall Contract Risk: " + overall)

    create_pdf(report, "report.pdf")
    with open("report.pdf", "rb") as f:
        st.download_button("Download PDF Report", f, "contract_report.pdf")

    save_log({
        "time": datetime.now().isoformat(),
        "risk": overall,
        "clauses": len(clauses),
        "parties": parties
    })

    st.success("Analysis Complete")