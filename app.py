from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import os, re, json, threading
from dotenv import load_dotenv

# Read DATABASE_URL and SECRET_KEY from the local .env file before configuring Flask.
load_dotenv()
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Database timestamps are stored as naive UTC. Display them in Indian Standard Time.
def format_ist(value, fmt="%d %b %Y %H:%M"):
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(ZoneInfo("Asia/Kolkata")).strftime(fmt)


app = Flask(__name__)
app.jinja_env.filters["ist"] = format_ist
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///jobmatch.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BASE_DIR, "uploads", "jobmatch_dataset_2000_dropdown_matched.csv")
DATASETS = {
    "1": ("Dataset 1 — Original (2,000 rows)", os.path.join(BASE_DIR, "uploads", "jobmatch_dataset_2000_dropdown_matched.csv")),
    "2": ("Dataset 2 — Mixed Industries (3,000 rows)", os.path.join(BASE_DIR, "uploads", "jobmatch_dataset_2_mixed_industries.csv")),
    "3": ("Dataset 3 — Varied Candidate Profiles (4,000 rows)", os.path.join(BASE_DIR, "uploads", "jobmatch_dataset_3_varied_profiles.csv")),
}
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TRAINING_LOCK = threading.Lock()


# Shared dropdown values used by BOTH Job and Candidate forms.
JOB_TITLES = [
"Software Engineer","Software Developer","Python Developer","Java Developer","Backend Developer","Frontend Developer","Full Stack Developer","Web Developer","Mobile App Developer","Android Developer","iOS Developer","React Developer","Angular Developer","Node.js Developer","PHP Developer",".NET Developer","C++ Developer","Game Developer","DevOps Engineer","Cloud Engineer","AWS Engineer","Azure Engineer","Site Reliability Engineer","Systems Engineer","Network Engineer","Network Administrator","System Administrator","Linux Administrator","Database Administrator","SQL Developer","Data Engineer","Data Analyst","Business Analyst","BI Developer","Power BI Developer","Tableau Developer","Data Scientist","Machine Learning Engineer","AI Engineer","NLP Engineer","Computer Vision Engineer","MLOps Engineer","Research Engineer","Cybersecurity Analyst","Security Engineer","SOC Analyst","Penetration Tester","Information Security Analyst","QA Engineer","Software Tester","Automation Test Engineer","Manual Tester","Product Manager","Project Manager","Program Manager","Scrum Master","Technical Lead","Engineering Manager","Solutions Architect","Cloud Architect","Enterprise Architect","UI Designer","UX Designer","UI/UX Designer","Graphic Designer","Product Designer","Digital Marketing Specialist","SEO Specialist","Content Writer","Technical Writer","Social Media Manager","Marketing Analyst","Sales Executive","Sales Manager","Business Development Executive","Account Manager","Customer Success Manager","Customer Support Executive","HR Executive","HR Manager","Recruiter","Talent Acquisition Specialist","Operations Executive","Operations Manager","Supply Chain Analyst","Financial Analyst","Accountant","Finance Manager","Risk Analyst","Investment Analyst","Mechanical Engineer","Electrical Engineer","Electronics Engineer","Civil Engineer","Chemical Engineer","Embedded Systems Engineer","IoT Engineer","Robotics Engineer","Support Engineer","IT Support Specialist","Help Desk Technician","ERP Consultant","SAP Consultant","CRM Developer","Blockchain Developer"
]
SKILLS = ["Python","Java","JavaScript","TypeScript","C","C++","C#","PHP","R","Go","Kotlin","Swift","HTML","CSS","React","Angular","Vue.js","Node.js","Express.js","Django","Flask","FastAPI","Spring Boot",".NET","SQL","MySQL","PostgreSQL","MongoDB","Redis","Oracle","AWS","Azure","Google Cloud","Docker","Kubernetes","Jenkins","Git","GitHub","Linux","Networking","Cybersecurity","SIEM","Machine Learning","Deep Learning","NLP","Computer Vision","Pandas","NumPy","scikit-learn","TensorFlow","PyTorch","Power BI","Tableau","Excel","REST API","Microservices","Data Structures","Algorithms","Spark","Hadoop","ETL","Communication","Leadership","Problem Solving","Teamwork","Time Management"]
SKILLS += ["Negotiation", "Social Media", "Financial Analysis", "Accounting", "Taxation", "Risk Management", "Recruitment", "Supply Chain Management", "Quality Control", "Creativity", "SEO"]
SKILLS = list(dict.fromkeys(SKILLS))

EDUCATION_LEVELS = ["High School","Diploma","Bachelor","Master","PhD"]
DEGREES = ["B.Tech","B.E.","B.Sc","BCA","BBA","B.Com","BA","M.Tech","M.E.","M.Sc","MCA","MBA","M.Com","MA","PhD","Diploma","Other"]
SALARY_RANGES = ["₹0-₹3L","₹3L-₹5L","₹5L-₹7L","₹7L-₹10L","₹7L-₹14L","₹10L-₹15L","₹12L-₹18L","₹15L-₹20L","₹18L-₹25L","₹20L-₹30L","₹25L-₹35L","₹30L-₹40L","₹40L+"]
EXPERIENCE_LEVELS = list(range(0, 21))
JOB_TYPES = ["Full Time","Part Time","Contract","Internship","Remote"]

# ============================================================
# DATABASE MODELS
# ============================================================


SOFT_SKILLS = [
    "Communication", "Teamwork", "Leadership", "Problem Solving",
    "Time Management", "Adaptability", "Critical Thinking", "Creativity",
    "Decision Making", "Conflict Resolution", "Attention to Detail",
    "Organization", "Collaboration", "Presentation", "Negotiation"
]

LANGUAGES = [
    "English", "Hindi", "Gujarati", "Marathi", "Bengali", "Tamil",
    "Telugu", "Kannada", "Malayalam", "Punjabi", "Urdu",
    "French", "German", "Spanish", "Japanese", "Mandarin"
]


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    job_title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200))
    required_skills = db.Column(db.Text)
    preferred_skills = db.Column(db.Text)
    education = db.Column(db.String(200))
    degree = db.Column(db.String(200))
    min_experience = db.Column(db.Float, default=0)
    location = db.Column(db.String(200))
    salary_range = db.Column(db.String(100))
    certifications = db.Column(db.Text)
    job_type = db.Column(db.String(100))
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(255))
    job_title = db.Column(db.String(200))
    education = db.Column(db.String(200))
    degree = db.Column(db.String(200))
    experience = db.Column(db.Float, default=0)
    salary_range = db.Column(db.String(100))
    technical_skills = db.Column(db.Text)
    soft_skills = db.Column(db.Text)
    certifications = db.Column(db.Text)
    projects = db.Column(db.Text)
    previous_job_titles = db.Column(db.Text)
    resume_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey("job.id"), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate.id"), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(50))
    model_name = db.Column(db.String(100))
    breakdown = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Training(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    dataset_name = db.Column(db.String(255))
    rows = db.Column(db.Integer)
    columns = db.Column(db.Integer)
    target = db.Column(db.String(100))
    best_model = db.Column(db.String(100))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1 = db.Column(db.Float)
    confusion = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# Initialize tables on startup rather than at import time. This keeps public
# pages reachable when PostgreSQL is temporarily offline and makes errors visible.
def initialize_database():
    try:
        with app.app_context():
            db.create_all()
        print("[JobMatch] Database connection successful.")
    except Exception as exc:
        print("[JobMatch] DATABASE ERROR: Could not connect or create tables.")
        print(f"[JobMatch] {type(exc).__name__}: {exc}")
        print("[JobMatch] Check PostgreSQL service, .env DATABASE_URL and credentials.")




# Sample jobs are inserted once per account; deleted jobs are not recreated on page visits.
DEFAULT_JOBS = [('Python Developer', 'Python, Flask, SQL', 'B.Tech', 2, '₹5L-₹7L'), ('Data Analyst', 'Excel, SQL, Python', 'B.Sc', 1, '₹5L-₹7L'), ('Cybersecurity Analyst', 'Cybersecurity, Networking, Linux', 'B.Tech', 2, '₹7L-₹10L'), ('Sales Executive', 'Communication, Negotiation, Excel', 'BBA', 1, '₹3L-₹5L'), ('Sales Manager', 'Leadership, Communication, Negotiation', 'MBA', 5, '₹10L-₹15L'), ('Digital Marketing Specialist', 'SEO, Social Media, Excel', 'BBA', 2, '₹5L-₹7L'), ('SEO Specialist', 'SEO, HTML, Excel', 'BA', 1, '₹3L-₹5L'), ('Financial Analyst', 'Excel, SQL, Financial Analysis', 'B.Com', 2, '₹7L-₹10L'), ('Accountant', 'Excel, Accounting, Taxation', 'B.Com', 1, '₹3L-₹5L'), ('Risk Analyst', 'Excel, SQL, Risk Management', 'B.Com', 2, '₹7L-₹10L'), ('Business Analyst', 'SQL, Excel, Communication', 'BBA', 2, '₹7L-₹10L'), ('HR Executive', 'Communication, Recruitment, Excel', 'BBA', 1, '₹3L-₹5L'), ('Recruiter', 'Communication, Recruitment, Negotiation', 'BBA', 1, '₹3L-₹5L'), ('Operations Manager', 'Leadership, Problem Solving, Excel', 'MBA', 5, '₹10L-₹15L'), ('Supply Chain Analyst', 'Excel, SQL, Supply Chain Management', 'BBA', 2, '₹5L-₹7L'), ('UI/UX Designer', 'HTML, CSS, Creativity', 'B.Sc', 2, '₹5L-₹7L'), ('Customer Support Executive', 'Communication, Problem Solving, Teamwork', 'BA', 0, '₹3L-₹5L'), ('Business Development Executive', 'Communication, Negotiation, Excel', 'BBA', 1, '₹5L-₹7L'), ('Mechanical Engineer', 'Problem Solving, Quality Control, Excel', 'B.E.', 2, '₹5L-₹7L'), ('Project Manager', 'Leadership, Time Management, Communication', 'MBA', 5, '₹12L-₹18L')]

def add_default_jobs(user_id):
    existing = {job.job_title for job in Job.query.filter_by(user_id=user_id).all()}
    for title, skills, degree, years, salary in DEFAULT_JOBS:
        if title not in existing:
            db.session.add(Job(user_id=user_id, job_title=title, company="Demo Company",
                required_skills=skills, preferred_skills="", education="Master" if degree == "MBA" else "Bachelor",
                degree=degree, min_experience=years, salary_range=salary, location="India",
                job_type="Full Time", description=f"Sample {title} role"))
    db.session.commit()

# ============================================================
# HELPERS
# ============================================================

def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Login failed. Please log in first.", "danger")
            return redirect(url_for("home"))
        return function(*args, **kwargs)
    return wrapper


def current_user():
    if "user_id" not in session:
        return None
    try:
        user = db.session.get(User, session["user_id"])
        if user is None:
            session.pop("user_id", None)
        return user
    except Exception:
        db.session.rollback()
        session.pop("user_id", None)
        return None


@app.context_processor
def inject_user():
    return {"current_user": current_user(), "JOB_TITLES": JOB_TITLES, "SKILLS": SKILLS, "SOFT_SKILLS": SOFT_SKILLS, "EDUCATION_LEVELS": EDUCATION_LEVELS, "DEGREES": DEGREES, "SALARY_RANGES": SALARY_RANGES, "EXPERIENCE_LEVELS": EXPERIENCE_LEVELS, "JOB_TYPES": JOB_TYPES}


def tokens(value):
    return set(re.findall(r"[a-zA-Z0-9+#.\-]+", str(value or "").lower()))


def similarity(a, b):
    first, second = tokens(a), tokens(b)
    if not first and not second:
        return 100.0
    return round(100 * len(first & second) / max(1, len(first | second)), 1)


def _norm(value):
    return str(value or "").strip().lower().replace(".", "")


def salary_bounds(value):
    """Return approximate LPA (low, high). Supports dropdown values such as ₹7L-₹10L and ₹40L+."""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]
    if not nums:
        return None
    if "+" in str(value):
        return nums[0], nums[0] + 15.0
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums[0], nums[1]), max(nums[0], nums[1])


def salary_compatibility(job_salary, candidate_salary):
    """High score when expectation is inside/below the offered range; gradually penalize higher expectations."""
    jb, cb = salary_bounds(job_salary), salary_bounds(candidate_salary)
    if not jb or not cb:
        return 100.0
    jlo, jhi = jb
    clo, chi = cb
    expected = (clo + chi) / 2
    # Candidate expectation is comfortably affordable.
    if expected <= jhi:
        if expected >= jlo:
            return 100.0
        # Below budget is still a very good compatibility signal.
        return 96.0
    # Smooth penalty for expectation above the job's maximum.
    gap = expected - jhi
    return round(max(20.0, 100.0 - (gap / max(jhi, 1.0)) * 120.0), 1)


EDU_RANK = {"high school": 0, "diploma": 1, "bachelor": 2, "master": 3, "phd": 4}

def education_compatibility(required, actual):
    if not required:
        return 100.0
    r, a = EDU_RANK.get(_norm(required)), EDU_RANK.get(_norm(actual))
    if r is None or a is None:
        return similarity(required, actual)
    if a >= r:
        return 100.0
    return max(35.0, 100.0 - (r-a)*30.0)


DEGREE_FAMILIES = {
    "computing": {"btech","be","bsc","bca","mtech","me","msc","mca"},
    "business": {"bba","bcom","ba","mba","mcom","ma"},
    "masters_tech": {"mtech","me","msc","mca"},
    "bachelors_tech": {"btech","be","bsc","bca"},
}

def degree_compatibility(required, actual):
    if not required:
        return 100.0
    r, a = _norm(required), _norm(actual)
    if r == a:
        return 100.0
    # Closely related postgraduate technical degrees (e.g. M.Sc <-> MCA) are strong substitutes.
    if r in DEGREE_FAMILIES["masters_tech"] and a in DEGREE_FAMILIES["masters_tech"]:
        return 90.0
    if r in DEGREE_FAMILIES["bachelors_tech"] and a in DEGREE_FAMILIES["masters_tech"]:
        return 100.0
    if r in DEGREE_FAMILIES["bachelors_tech"] and a in DEGREE_FAMILIES["bachelors_tech"]:
        return 88.0
    if r in DEGREE_FAMILIES["business"] and a in DEGREE_FAMILIES["business"]:
        return 82.0
    if a == "phd":
        return 95.0
    return 40.0


TITLE_GROUPS = [
    {"software engineer","software developer","python developer","java developer","backend developer","full stack developer","web developer","nodejs developer","php developer","net developer","c++ developer"},
    {"frontend developer","react developer","angular developer","web developer","ui designer","ui/ux designer"},
    {"data scientist","machine learning engineer","ai engineer","nlp engineer","computer vision engineer","mlops engineer","research engineer"},
    {"data engineer","data analyst","business analyst","bi developer","power bi developer","tableau developer","sql developer"},
    {"devops engineer","cloud engineer","aws engineer","azure engineer","site reliability engineer","systems engineer"},
    {"cybersecurity analyst","security engineer","soc analyst","penetration tester","information security analyst"},
    {"qa engineer","software tester","automation test engineer","manual tester"},
    {"network engineer","network administrator","system administrator","linux administrator","it support specialist","support engineer","help desk technician"},
]

def title_compatibility(required, actual):
    r, a = _norm(required), _norm(actual)
    if not r:
        return 100.0
    if r == a:
        return 100.0
    for group in TITLE_GROUPS:
        if r in group and a in group:
            return 85.0
    lexical = similarity(required, actual)
    return max(25.0, lexical)


def experience_compatibility(required, actual):
    req, have = float(required or 0), float(actual or 0)
    if req <= 0 or have >= req:
        return 100.0
    # One year short should not collapse the score; larger gaps are penalized progressively.
    ratio = have / req
    return round(max(20.0, 35.0 + 65.0 * ratio), 1)


def skill_compatibility(required_text, candidate_text, related_credit=False):
    required, available = tokens(required_text), tokens(candidate_text)
    if not required:
        return 100.0
    exact = len(required & available) / len(required)
    score = exact * 100.0
    # Related-stack credit gives modest partial credit without treating unrelated skills as exact matches.
    if related_credit and score < 100:
        tech_groups = [
            {"python","django","flask","fastapi"}, {"java","spring","springboot"},
            {"javascript","typescript","react","angular","vuejs","nodejs","expressjs"},
            {"mysql","postgresql","sql","oracle","mongodb","redis"},
            {"aws","azure","google","cloud","docker","kubernetes"},
            {"tensorflow","pytorch","scikit-learn","machine","learning","deep"}
        ]
        missing = required - available
        for item in missing:
            if any(item in g and (g & available) for g in tech_groups):
                score += 45.0 / len(required)
    return round(min(100.0, score), 1)


def matching_breakdown(job, candidate):
    candidate_skills = " ".join([candidate.technical_skills or "", candidate.resume_text or ""])
    actual_title = candidate.job_title or candidate.previous_job_titles
    return {
        "Job Title": title_compatibility(job.job_title, actual_title),
        "Required Skills": skill_compatibility(job.required_skills, candidate_skills, True),
        "Experience": experience_compatibility(job.min_experience, candidate.experience),
        "Education": education_compatibility(job.education, candidate.education),
        "Degree": degree_compatibility(job.degree, candidate.degree),
        "Salary": salary_compatibility(job.salary_range, candidate.salary_range),
    }


def structured_score(b):
    # Required skills/title matter most. Salary and equivalent degrees affect the result smoothly.
    weights = {"Job Title": .20, "Required Skills": .30, "Experience": .18, "Education": .09, "Degree": .12, "Salary": .11}
    return sum(b[k] * weights[k] for k in weights)

def match_probability(job, candidate, bundle):
    """One scoring formula for the main result and alternative jobs."""
    breakdown = matching_breakdown(job, candidate)
    rule_score = structured_score(breakdown)
    frame = form_record(job, candidate, bundle["columns"], bundle.get("numeric_columns", []))
    ml_score = float(bundle["model"].predict_proba(frame)[0][1]) * 100
    return round(max(0, min(100, ml_score * 0.60 + rule_score * 0.40)), 2)


def nearest_jobs_for(candidate, current_job, user_id, bundle, limit=5):
    ranked = []
    for other in Job.query.filter_by(user_id=user_id).all():
        if other.id == current_job.id:
            continue
        ranked.append({"job": other, "score": match_probability(other, candidate, bundle)})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


# ============================================================
# GENERIC CSV ML TRAINING
# ============================================================

TARGET_NAMES = [
    "eligible", "is_eligible", "hired", "selected",
    "match", "matched", "label", "target", "status", "outcome"
]


def find_target(data):
    for wanted in TARGET_NAMES:
        for column in data.columns:
            if str(column).strip().lower() == wanted:
                return column

    for column in data.columns:
        values = data[column].dropna().astype(str).str.strip().str.lower().unique()
        if len(values) == 2:
            return column

    raise ValueError(
        "No binary target column found. Use a column such as "
        "Eligible, Hired, Selected, Match, Label, Target or Status."
    )


def normalize_target(series):
    positive = {
        "1", "yes", "y", "true", "eligible", "hired",
        "selected", "match", "matched", "approved", "accepted"
    }
    negative = {
        "0", "no", "n", "false", "not eligible", "rejected",
        "reject", "unselected", "not matched"
    }

    result = []
    for value in series.astype(str).str.strip().str.lower():
        if value in positive:
            result.append(1)
        elif value in negative:
            result.append(0)
        else:
            try:
                number = int(float(value))
                result.append(number if number in (0, 1) else np.nan)
            except ValueError:
                result.append(np.nan)

    return pd.Series(result, index=series.index)


def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


def train_models(file_path):
    """Train five models with bounded memory on a small Render instance."""
    import gc
    import tempfile
    from sklearn.base import clone

    data = pd.read_csv(file_path).drop_duplicates().dropna(how="all")
    if "Eligible" not in data.columns:
        raise ValueError("Dataset must contain the Eligible target column.")
    y = normalize_target(data["Eligible"])
    valid = y.notna()
    data = data.loc[valid].copy()
    y = y.loc[valid].astype(int)
    if y.nunique() != 2:
        raise ValueError("Eligible must contain both Yes and No examples.")

    X = data.drop(columns=["Eligible"])
    X = X.drop(columns=[c for c in X if "certif" in c.lower() or c.lower() in ("preferred_skills", "soft_skills")], errors="ignore")
    X = X.drop(columns=[c for c in ["Transaction_ID", "id"] if c in X], errors="ignore")
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()
    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical_columns:
        X[col] = X[col].fillna("").astype("category")

    # Fit preprocessing ONCE; sparse one-hot encoding avoids huge dense matrices.
    preprocessor = ColumnTransformer([
        ("numbers", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_columns),
        ("categories", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), categorical_columns)
    ], sparse_threshold=1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    del X_train, X_test, data
    gc.collect()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=600, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=60, max_depth=14, min_samples_leaf=2,
            random_state=42, class_weight="balanced", n_jobs=1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20, min_samples_leaf=2, random_state=42, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(
            kernel="linear", probability=False, random_state=42, class_weight="balanced"
        )
    }

    results = {}
    best_model = None
    best_score = -1
    best_fitted = None
    # MultinomialNB requires nonnegative features; use sparse absolute values
    # (the scaler can create negative numeric values).
    for name, model in models.items():
        if name == "Naive Bayes":
            train_matrix = abs(X_train_encoded)
            test_matrix = abs(X_test_encoded)
        else:
            train_matrix = X_train_encoded
            test_matrix = X_test_encoded
        model.fit(train_matrix, y_train)
        prediction = model.predict(test_matrix)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, prediction), 4),
            "precision": round(precision_score(y_test, prediction, zero_division=0), 4),
            "recall": round(recall_score(y_test, prediction, zero_division=0), 4),
            "f1": round(f1_score(y_test, prediction, zero_division=0), 4),
            "confusion": confusion_matrix(y_test, prediction).tolist()
        }
        if results[name]["f1"] > best_score:
            best_score = results[name]["f1"]
            best_model = name
            best_fitted = model
        del prediction, train_matrix, test_matrix
        gc.collect()

    # Prediction expects a pipeline; keep a preprocessing transform inside it.
    # Naive Bayes is trained on absolute-valued features, so persist that step.
    if best_model == "Naive Bayes":
        from sklearn.preprocessing import FunctionTransformer
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("nonnegative", FunctionTransformer(np.abs, validate=False)),
            ("model", best_fitted)
        ])
    else:
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_fitted)])

    payload = {
        "model": pipeline,
        "columns": X.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target": "Eligible",
        "best_model": best_model,
        "models": results
    }
    # Write a temporary file and replace only after serialization succeeds.
    fd, temp_path = tempfile.mkstemp(prefix="jobmatch_model_", suffix=".pkl", dir=BASE_DIR)
    os.close(fd)
    try:
        joblib.dump(payload, temp_path, compress=3)
        os.replace(temp_path, MODEL_PATH)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "rows": len(X), "columns": len(X.columns) + 1,
        "target": "Eligible", "results": results, "best_model": best_model
    }


def form_record(job, candidate, columns, numeric_columns=None):
    base = {
        "job_title": job.job_title,
        "company": job.company,
        "required_skills": job.required_skills,
        "preferred_skills": job.preferred_skills,
        "education": job.education,
        "job_degree": job.degree,
        "min_experience": job.min_experience,
        "location": job.location,
        "salary_range": job.salary_range,
        "job_type": job.job_type,
        "description": job.description,
        "candidate_name": candidate.full_name,
        "candidate_email": candidate.email,
        "candidate_job_title": candidate.job_title,
        "candidate_education": candidate.education,
        "degree": candidate.degree,
        "experience": candidate.experience,
        "candidate_salary": candidate.salary_range,
        "candidate_salary_range": candidate.salary_range,
        "technical_skills": candidate.technical_skills,
        "soft_skills": candidate.soft_skills,
        "projects": candidate.projects,
        "previous_job_titles": candidate.previous_job_titles,
        "resume_text": candidate.resume_text
    }

    record = {}
    for column in columns:
        record[column] = base.get(column, "")

    frame = pd.DataFrame([record], columns=columns)
    # Preserve the same numeric dtypes that were used during model training.
    for column in (numeric_columns or []):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


# ============================================================
# PUBLIC PAGES + AUTHENTICATION
# ============================================================

@app.route("/")
def home():
    latest_prediction = None
    if "user_id" in session:
        latest_prediction = Prediction.query.filter_by(user_id=session["user_id"]).order_by(Prediction.id.desc()).first()
    return render_template("home.html", latest_prediction=latest_prediction)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/signup", methods=["POST"])
def signup():
    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")

    if not full_name or not email or len(password) < 8:
        flash("Please complete all fields. Password must be at least 8 characters.", "danger")
        return redirect(url_for("home"))

    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("home"))

    if User.query.filter_by(email=email).first():
        flash("An account with this email already exists.", "danger")
        return redirect(url_for("home"))

    user = User(
        full_name=full_name,
        email=email,
        password_hash=generate_password_hash(password)
    )
    db.session.add(user)
    db.session.commit()

    session["user_id"] = user.id
    add_default_jobs(user.id)
    flash("Account created successfully.", "success")
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password_hash, password):
        flash("Invalid email or password.", "danger")
        return redirect(url_for("home"))

    session["user_id"] = user.id
    flash("Welcome back!", "success")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# ============================================================
# DASHBOARD + PROFILE
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(
        Prediction.created_at.desc()
    ).all()

    average = round(
        np.mean([prediction.probability for prediction in predictions]), 1
    ) if predictions else 0

    return render_template(
        "dashboard.html",
        jobs=Job.query.filter_by(user_id=user_id).count(),
        candidates=Candidate.query.filter_by(user_id=user_id).count(),
        predictions=len(predictions),
        average=average,
        recent=predictions[:5]
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()

        existing = User.query.filter(
            User.email == email,
            User.id != user.id
        ).first()

        if existing:
            flash("That email is already in use.", "danger")
            return redirect(url_for("profile"))

        user.full_name = request.form.get("full_name", "").strip()
        user.email = email
        db.session.commit()
        flash("Profile updated.", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html")


# ============================================================
# JOBS
# ============================================================

@app.route("/jobs/load-samples", methods=["POST"])
@login_required
def load_sample_jobs():
    add_default_jobs(session["user_id"])
    flash("Sample jobs loaded (existing job titles were kept).", "success")
    return redirect(url_for("jobs"))


@app.route("/jobs", methods=["GET", "POST"])
@login_required
def jobs():
    if request.method == "POST":
        if any(not request.form.get(field, "").strip() for field in ["job_title", "education", "degree", "salary_range"]) or not request.form.getlist("required_skills"):
            flash("Please fill all required fields, including at least one skill.", "danger")
            return redirect(url_for("jobs"))
        form = request.form

        job = Job(
            user_id=session["user_id"],
            job_title=form.get("job_title", "").strip(),
            company=form.get("company", "").strip(),
            required_skills=", ".join(form.getlist("required_skills")),
            preferred_skills="",
            education=form.get("education", ""),
            degree=form.get("degree", ""),
            min_experience=float(form.get("min_experience") or 0),
            location=form.get("location", ""),
            salary_range=form.get("salary_range", ""),
            certifications=form.get("certifications", ""),
            job_type=form.get("job_type", ""),
            description=form.get("description", "")
        )

        db.session.add(job)
        db.session.commit()
        flash("Job saved.", "success")
        return redirect(url_for("jobs"))

    jobs_list = Job.query.filter_by(user_id=session["user_id"]).order_by(
        Job.created_at.desc()
    ).all()

    return render_template("jobs.html", jobs=jobs_list)


@app.route("/jobs/edit/<int:job_id>", methods=["GET", "POST"])
@login_required
def edit_job(job_id):
    job = Job.query.filter_by(
        id=job_id,
        user_id=session["user_id"]
    ).first_or_404()

    if request.method == "POST":
        if any(not request.form.get(field, "").strip() for field in ["job_title", "education", "degree", "salary_range"]) or not request.form.getlist("required_skills"):
            flash("Please fill all required fields, including at least one skill.", "danger")
            return redirect(url_for("jobs"))
        form = request.form

        job.job_title = form.get("job_title", "").strip()
        job.company = form.get("company", "").strip()
        job.required_skills = ", ".join(form.getlist("required_skills"))
        job.preferred_skills = ""
        job.education = form.get("education", "")
        job.degree = form.get("degree", "")
        job.min_experience = float(form.get("min_experience") or 0)
        job.location = form.get("location", "")
        job.salary_range = form.get("salary_range", "")
        job.certifications = form.get("certifications", "")
        job.job_type = form.get("job_type", "")
        job.description = form.get("description", "")

        db.session.commit()
        flash("Job updated.", "success")
        return redirect(url_for("jobs"))

    return render_template("job_edit.html", job=job)


@app.route("/jobs/delete/<int:job_id>")
@login_required
def delete_job(job_id):
    job = Job.query.filter_by(
        id=job_id,
        user_id=session["user_id"]
    ).first_or_404()

    db.session.delete(job)
    db.session.commit()
    flash("Job deleted.", "success")
    return redirect(url_for("jobs"))


# ============================================================
# CANDIDATES
# ============================================================

@app.route("/candidates", methods=["GET", "POST"])
@login_required
def candidates():
    if request.method == "POST":
        if any(not request.form.get(field, "").strip() for field in ["full_name", "job_title", "education", "degree", "salary_range"]) or not request.form.getlist("technical_skills"):
            flash("Please fill all required fields, including at least one skill.", "danger")
            return redirect(url_for("candidates"))
        form = request.form

        candidate = Candidate(
            user_id=session["user_id"],
            full_name=form.get("full_name", "").strip(),
            email=form.get("email", ""),
            job_title=form.get("job_title", ""),
            education=form.get("education", ""),
            degree=form.get("degree", ""),
            experience=float(form.get("experience") or 0),
            salary_range=form.get("salary_range", ""),
            technical_skills=", ".join(form.getlist("technical_skills")),
            soft_skills="",
            certifications=form.get("certifications", ""),
            projects=form.get("projects", ""),
            previous_job_titles=form.get("previous_job_titles", ""),
            resume_text=form.get("resume_text", "")
        )

        db.session.add(candidate)
        db.session.commit()
        flash("Candidate saved.", "success")
        return redirect(url_for("candidates"))

    candidates_list = Candidate.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Candidate.created_at.desc()).all()

    return render_template(
        "candidates.html",
        candidates=candidates_list
    )


@app.route("/candidates/edit/<int:candidate_id>", methods=["GET", "POST"])
@login_required
def edit_candidate(candidate_id):
    candidate = Candidate.query.filter_by(
        id=candidate_id,
        user_id=session["user_id"]
    ).first_or_404()

    if request.method == "POST":
        if any(not request.form.get(field, "").strip() for field in ["full_name", "job_title", "education", "degree", "salary_range"]) or not request.form.getlist("technical_skills"):
            flash("Please fill all required fields, including at least one skill.", "danger")
            return redirect(url_for("candidates"))
        form = request.form

        candidate.full_name = form.get("full_name", "").strip()
        candidate.email = form.get("email", "")
        candidate.job_title = form.get("job_title", "")
        candidate.education = form.get("education", "")
        candidate.degree = form.get("degree", "")
        candidate.experience = float(form.get("experience") or 0)
        candidate.salary_range = form.get("salary_range", "")
        candidate.technical_skills = ", ".join(form.getlist("technical_skills"))
        candidate.soft_skills = ""
        candidate.certifications = form.get("certifications", "")
        candidate.projects = form.get("projects", "")
        candidate.previous_job_titles = form.get("previous_job_titles", "")
        candidate.resume_text = form.get("resume_text", "")

        db.session.commit()
        flash("Candidate updated.", "success")
        return redirect(url_for("candidates"))

    return render_template(
        "candidate_edit.html",
        candidate=candidate
    )


@app.route("/candidates/delete/<int:candidate_id>")
@login_required
def delete_candidate(candidate_id):
    candidate = Candidate.query.filter_by(
        id=candidate_id,
        user_id=session["user_id"]
    ).first_or_404()

    db.session.delete(candidate)
    db.session.commit()
    flash("Candidate deleted.", "success")
    return redirect(url_for("candidates"))


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    result = None

    if request.method == "POST":
        choice = request.form.get("dataset_choice", "1")
        if choice == "custom":
            file = request.files.get("dataset")
            if not file or not file.filename:
                flash("Please choose a CSV file to upload.", "danger")
                return redirect(url_for("train"))
            if not file.filename.lower().endswith(".csv"):
                flash("Please upload a CSV dataset.", "danger")
                return redirect(url_for("train"))
            from werkzeug.utils import secure_filename
            import uuid
            upload_dir = os.path.join(BASE_DIR, "uploads", "custom")
            os.makedirs(upload_dir, exist_ok=True)
            safe_name = secure_filename(file.filename)
            file_path = os.path.join(upload_dir, uuid.uuid4().hex + "_" + safe_name)
            file.save(file_path)
            dataset_name = "Custom — " + safe_name
        elif choice in DATASETS:
            dataset_name, file_path = DATASETS[choice]
            if not os.path.isfile(file_path):
                flash("The selected dataset could not be found.", "danger")
                return redirect(url_for("train"))
        else:
            flash("Please select a valid dataset.", "danger")
            return redirect(url_for("train"))

        if not TRAINING_LOCK.acquire(blocking=False):
            flash("A training job is already running. Please try again later.", "warning")
            return redirect(url_for("train"))

        try:
            result = train_models(file_path)
            best = result["results"][result["best_model"]]

            db.session.add(Training(
                user_id=session["user_id"],
                dataset_name=dataset_name,
                rows=result["rows"],
                columns=result["columns"],
                target=result["target"],
                best_model=result["best_model"],
                accuracy=best["accuracy"],
                precision=best["precision"],
                recall=best["recall"],
                f1=best["f1"],
                confusion=json.dumps(best["confusion"])
            ))
            db.session.commit()

            session["training_result"] = result
            flash("Training successful.", "success")
        except Exception as error:
            app.logger.exception("Model training failed")
            flash("Training failed. Check the Render logs for the detailed error.", "danger")
        finally:
            TRAINING_LOCK.release()
            if choice == "custom" and os.path.exists(file_path):
                os.remove(file_path)

        return redirect(url_for("train"))

    result = session.pop("training_result", None)

    history = Training.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Training.created_at.desc()).limit(10).all()

    return render_template("train.html", result=result, history=history)


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    user_id = session["user_id"]
    jobs_list = Job.query.filter_by(user_id=user_id).order_by(Job.created_at.desc()).all()
    candidates_list = Candidate.query.filter_by(user_id=user_id).order_by(Candidate.created_at.desc()).all()
    result = None

    if request.method == "POST":
        if not os.path.exists(MODEL_PATH):
            flash("Prediction failed. Please train the model first.", "danger")
            return redirect(url_for("train"))

        job = Job.query.filter_by(
            id=request.form.get("job_id"),
            user_id=user_id
        ).first_or_404()

        candidate = Candidate.query.filter_by(
            id=request.form.get("candidate_id"),
            user_id=user_id
        ).first_or_404()

        bundle = joblib.load(MODEL_PATH)
        breakdown = matching_breakdown(job, candidate)
        try:
            probability = match_probability(job, candidate, bundle)
            nearest_jobs = nearest_jobs_for(candidate, job, user_id, bundle)
            prediction_model_name = bundle.get("best_model", "Trained ML Model")
        except Exception:
            flash("Prediction failed. Please retrain the model using the updated dataset.", "danger")
            return redirect(url_for("train"))

        label = (
            "Strong Match" if probability >= 75
            else "Good Match" if probability >= 50
            else "Low Match"
        )

        prediction = Prediction(
            user_id=user_id,
            job_id=job.id,
            candidate_id=candidate.id,
            probability=probability,
            label=label,
            model_name=prediction_model_name,
            breakdown=json.dumps(breakdown)
        )

        db.session.add(prediction)
        db.session.commit()

        result = {
            "probability": probability,
            "label": label,
            "model": prediction_model_name,
            "breakdown": breakdown,
            "job": job,
            "candidate": candidate,
            "nearest_jobs": nearest_jobs
        }

    return render_template(
        "predict.html",
        jobs=jobs_list,
        candidates=candidates_list,
        result=result
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():
    predictions = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Prediction.created_at.desc()).all()

    return render_template("history.html", predictions=predictions)


# ============================================================
# START APPLICATION
# ============================================================

@app.route("/health")
def health():
    """Simple server availability check; does not require a database connection."""
    return {"status": "running", "service": "JobMatch"}


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
    initialize_database()
    # Disable the debug reloader to avoid restarting while training large models.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
