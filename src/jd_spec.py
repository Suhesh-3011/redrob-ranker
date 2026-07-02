"""
jd_spec.py

A structured, hand-authored encoding of the Redrob JD (job_description.docx).

Why this exists: the JD is written specifically to punish naive keyword
matching ("A candidate who has all the AI keywords listed as skills but
whose title is 'Marketing Manager' is not a fit"). Rather than let an LLM
re-derive this at ranking time (which would violate the no-network-during-
ranking constraint anyway), we encode the JD's actual requirements once,
by hand, as structured rules + a rich reference text for semantic matching.
This is the "JD understanding" layer -- done offline, deterministically,
and transparently (every number here is traceable to a sentence in the JD).
"""

# ---------------------------------------------------------------------------
# Rich free-text description of the IDEAL candidate, used only for the
# semantic-embedding side of the score (captures nuance regex can't).
# Pulled directly from the JD's "how to read between the lines" section +
# the "must have" skills inventory, in the JD's own language.
# ---------------------------------------------------------------------------
IDEAL_CANDIDATE_TEXT = """
Senior AI/ML engineer with 6-8 years of total experience, of which 4-5 years
are in applied ML/AI roles at product companies, not pure IT services or
consulting firms. Has shipped at least one end-to-end embeddings-based
retrieval system, ranking system, search system, or recommendation system
to real users at meaningful production scale. Has hands-on production
experience with vector databases or hybrid search infrastructure such as
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, or FAISS --
has personally handled embedding drift, index refresh, and retrieval-quality
regressions in production, not just used these tools in a tutorial. Strong,
production-grade Python. Has designed evaluation frameworks for ranking
systems: NDCG, MRR, MAP, offline-to-online correlation, A/B test
interpretation. Understood retrieval and ranking before it was fashionable --
has substantive pre-LLM-era search, recommendation, or ranking production
experience, not just recent LangChain-to-OpenAI wrapper projects. Writes
production code personally, is not purely in an architecture or tech-lead
role. Has a stable career trajectory, staying 2-3+ years per company rather
than title-hopping every 12-18 months. Thinks in terms of systems and
tradeoffs (hybrid vs dense retrieval, fine-tune vs prompt) rather than
chasing frameworks. Some external validation of thinking -- publications,
talks, open source -- rather than five-plus years entirely on closed
proprietary systems. Comfortable being scrappy and shipping an imperfect v1
in a startup context, not just a polished-process big-company engineer.
"""

# ---------------------------------------------------------------------------
# Must-have hard requirements (JD: "Things you absolutely need")
# Each is a list of regex-safe substrings (lowercase) that count as evidence
# when found in career_history descriptions / titles / skill names.
# ---------------------------------------------------------------------------
MUST_HAVE_EMBEDDINGS = [
    "embedding", "sentence-transformer", "sentence transformer", "bge",
    "e5 embedding", "text embedding", "semantic search", "dense retrieval",
    "vector search", "similarity search",
]

MUST_HAVE_VECTOR_DB = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "elastic search", "faiss", "vector database",
    "vector db", "hybrid search", "hybrid retrieval",
]

MUST_HAVE_PYTHON = ["python"]

MUST_HAVE_EVAL_FRAMEWORK = [
    "ndcg", "mrr", "map@", "mean average precision", "a/b test", "ab test",
    "a/b testing", "offline evaluation", "online evaluation",
    "evaluation framework", "precision@", "recall@", "click-through",
    "ctr uplift", "offline-to-online",
]

MUST_HAVE_PRODUCTION_SYSTEM = [
    "ranking system", "recommendation system", "recommender system",
    "search system", "retrieval system", "search engine",
    "recommendation engine", "ranking model", "search ranking",
    "personalization", "matching system", "relevance",
]

# ---------------------------------------------------------------------------
# Nice-to-have (JD: "won't reject you for")
# ---------------------------------------------------------------------------
NICE_TO_HAVE = {
    "fine_tuning": ["lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetun"],
    "learning_to_rank": ["learning to rank", "learning-to-rank", "ltr", "xgboost", "lambdamart"],
    "hr_tech": ["hr-tech", "hr tech", "recruiting tech", "ats", "talent", "marketplace"],
    "distributed_systems": ["distributed system", "large-scale inference", "kubernetes", "kafka at scale", "sharding"],
    "open_source": ["open-source", "open source", "github stars", "published", "paper", "conference talk"],
}

# ---------------------------------------------------------------------------
# Hard disqualifiers (JD: "Things we explicitly do NOT want" + experience section)
# ---------------------------------------------------------------------------
BAD_CORE_TITLES = [
    "marketing manager", "marketing executive", "marketing specialist",
    "human resources", "hr manager", "hr executive", "hr generalist",
    "talent acquisition", "recruiter",
    "accountant", "accounting", "finance manager", "auditor",
    "civil engineer", "structural engineer",
    "graphic designer", "graphic design",
    "sales executive", "sales manager", "account executive", "business development",
    "customer support", "customer service", "support engineer", "technical support",
    "mechanical engineer",
    "content writer", "copywriter",
    "administrative", "office manager",
]

# "Operations" and "Business Analyst" are ambiguous (could be ops for an ML
# platform, or a legit analytics BA) -- we do NOT hard-disqualify on these,
# we just don't give them credit unless production ML evidence shows up
# elsewhere in career_history.
SOFT_CAUTION_TITLES = ["operations manager", "operations executive", "business analyst", "project manager"]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
]

CV_ROBOTICS_SPEECH_TERMS = [
    "computer vision", "image classification", "object detection",
    "robotics", "robot", "slam", "autonomous", "speech recognition",
    "speech-to-text", "asr ", "audio processing",
]

NLP_IR_TERMS = [
    "nlp", "natural language", "information retrieval", "text classification",
    "language model", "llm", "retrieval", "search", "ranking", "embedding",
    "transformer", "bert", "gpt",
]

PURE_RESEARCH_INDUSTRY_TERMS = ["academia", "research institute", "research lab", "university research"]
PRODUCTION_EVIDENCE_TERMS = [
    "deployed", "production", "shipped", "real users", "live traffic",
    "scaled to", "in production", "rolled out", "launched",
]

NON_CODING_SENIOR_TITLES = [
    "engineering manager", "director of engineering", "vp of engineering",
    "vp engineering", "head of engineering", "chief technology officer", "cto",
    "engineering director", "principal architect", "solutions architect",
]

# ---------------------------------------------------------------------------
# Location (JD: Pune/Noida preferred; Hyderabad/Pune/Mumbai/Delhi NCR welcome;
# outside India is case-by-case with NO visa sponsorship)
# ---------------------------------------------------------------------------
PREFERRED_CITIES = ["noida", "pune"]
WELCOME_CITIES = ["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "new delhi", "ncr"]

IDEAL_EXPERIENCE_MIN = 6.0
IDEAL_EXPERIENCE_MAX = 8.0
JD_EXPERIENCE_MIN = 5.0
JD_EXPERIENCE_MAX = 9.0

IDEAL_NOTICE_DAYS = 30

# Phrases that mark a claim as hobbyist/coursework rather than production
# experience. JD: "if your AI experience consists primarily of recent
# (under 12 months) projects using LangChain to call OpenAI ... unless you
# can demonstrate substantial pre-LLM-era ML production experience."
# A sentence containing one of these should NOT count as production
# evidence, even if it name-drops the right keywords.
HEDGE_TERMS = [
    "online course", "taking a course", "taking online courses", "certification",
    "side project", "personal project", "hobby project", "self-directed",
    "kaggle competition", "kaggle", "bootcamp", "learning about", "curious about",
    "exploring how", "experimenting with", "exploring", "in my spare time",
    "for fun", "toy project", "tutorial",
]
