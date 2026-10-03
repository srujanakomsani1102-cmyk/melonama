from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'cefm.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
from pymongo import MongoClient
from app.core.config import MONGODB_URL, MONGODB_DATABASE


# ============================================================
# MONGODB CONNECTION
# ============================================================

client = MongoClient(
    MONGODB_URL,
    serverSelectionTimeoutMS=5000,
)

db = client[MONGODB_DATABASE]


# ============================================================
# COLLECTIONS
# ============================================================

cases_collection = db["cases"]
analyses_collection = db["analyses"]
rag_documents_collection = db["rag_documents"]


# ============================================================
# CONNECTION CHECK
# ============================================================

def check_mongodb():
    """
    Check whether MongoDB is reachable.
    """
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False