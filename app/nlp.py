"""
NLP analysis module for ARC AI — 9 techniques, lazy-loaded models.

1. intent_classify     — zero-shot query intent (BART-MNLI)
2. extract_entities    — NER: dates, money, orgs, statutes (spaCy + regex)
3. topic_model         — LDA topic discovery (scikit-learn)
4. extractive_qa       — exact-span answer (RoBERTa-SQuAD2)
5. sentiment_analysis  — VADER lexicon + RoBERTa transformer
6. summarize_text      — abstractive summary (BART-large-CNN)
7. extract_keywords    — keyword extraction (KeyBERT)
8. readability_score   — Flesch-Kincaid grade level + reading ease
9. emotion_detect      — emotion classification (j-hartmann RoBERTa)
"""
from __future__ import annotations

import re
import math
from collections import Counter
from typing import Any

# Lazy singletons — loaded on first call
_intent_pipe = None
_qa_pipe = None
_sentiment_pipe = None
_summarize_pipe = None
_emotion_pipe = None
_keyword_model = None
_nlp_spacy = None


# ---------- helpers ----------

def _get_intent_pipe():
    global _intent_pipe
    if _intent_pipe is None:
        from transformers import pipeline
        _intent_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    return _intent_pipe


def _get_qa_pipe():
    global _qa_pipe
    if _qa_pipe is None:
        from transformers import pipeline
        _qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
    return _qa_pipe


def _get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        from transformers import pipeline
        _sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)
    return _sentiment_pipe


def _get_summarize_pipe():
    global _summarize_pipe
    if _summarize_pipe is None:
        from transformers import pipeline
        _summarize_pipe = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    return _summarize_pipe


def _get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is None:
        from transformers import pipeline
        _emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1, top_k=None)
    return _emotion_pipe


def _get_keyword_model():
    global _keyword_model
    if _keyword_model is None:
        from keybert import KeyBERT
        _keyword_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _keyword_model


def _get_spacy():
    global _nlp_spacy
    if _nlp_spacy is None:
        import spacy
        try:
            _nlp_spacy = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp_spacy = spacy.load("en_core_web_sm")
    return _nlp_spacy


# ---------- 1. Intent classification ----------

INTENT_LABELS = [
    "security deposit",
    "eviction",
    "rent increase",
    "repairs and habitability",
    "lease termination",
    "discrimination",
    "rental application",
    "landlord entry and privacy",
    "tenant rights general",
]


def intent_classify(query: str) -> dict[str, Any]:
    """Classify user query into intent categories (zero-shot)."""
    pipe = _get_intent_pipe()
    result = pipe(query, INTENT_LABELS)
    return {
        "technique": "intent_classification",
        "model": "facebook/bart-large-mnli",
        "query": query,
        "top_intent": result["labels"][0],
        "confidence": round(result["scores"][0], 4),
        "all_intents": [
            {"label": l, "score": round(s, 4)}
            for l, s in zip(result["labels"][:5], result["scores"][:5])
        ],
    }


# ---------- 2. NER ----------

_STATUTE_RE = re.compile(r"(?:Section|§|Sec\.?)\s*\d+[-–.]\d+(?:\.\d+)?", re.I)


def extract_entities(text: str) -> dict[str, Any]:
    """Extract legal entities: dates, money, orgs, statute references."""
    nlp = _get_spacy()
    doc = nlp(text[:5000])
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    money = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    statutes = _STATUTE_RE.findall(text)
    return {
        "technique": "named_entity_recognition",
        "model": "spacy/en_core_web_sm + regex",
        "dates": Counter(dates).most_common(10),
        "money": Counter(money).most_common(10),
        "organizations": Counter(orgs).most_common(10),
        "statute_references": Counter(statutes).most_common(10),
        "total_entities": len(dates) + len(money) + len(orgs) + len(statutes),
    }


# ---------- 3. Topic modeling ----------

def topic_model(texts: list[str], n_topics: int = 6) -> dict[str, Any]:
    """Run LDA topic modeling over a list of text chunks."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    vectorizer = CountVectorizer(
        max_df=0.85,
        min_df=2,
        stop_words="english",
        token_pattern=r"\b[a-z]{3,}\b",
    )
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=15, n_jobs=-1
    )
    lda.fit(X)

    topics = []
    for i, topic_weights in enumerate(lda.components_):
        top_words = [vocab[j] for j in topic_weights.argsort()[:-11:-1]]
        topics.append({"topic_id": i + 1, "top_words": top_words})

    return {
        "technique": "topic_modeling_lda",
        "model": "sklearn/LatentDirichletAllocation",
        "n_topics": n_topics,
        "n_documents": len(texts),
        "topics": topics,
    }


# ---------- 4. Extractive QA ----------

def extractive_qa(question: str, context: str) -> dict[str, Any]:
    """Extract exact answer span from context."""
    pipe = _get_qa_pipe()
    result = pipe(question=question, context=context[:3000])
    return {
        "technique": "extractive_qa",
        "model": "deepset/roberta-base-squad2",
        "question": question,
        "answer": result["answer"],
        "confidence": round(result["score"], 4),
        "start": result["start"],
        "end": result["end"],
    }


# ---------- 5. Sentiment analysis ----------

def sentiment_analysis(text: str) -> dict[str, Any]:
    """Dual sentiment: VADER (lexicon) + RoBERTa (transformer)."""
    # VADER
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(text[:5000])

    # RoBERTa
    pipe = _get_sentiment_pipe()
    # RoBERTa has a 512 token limit, truncate
    roberta_result = pipe(text[:1500])[0]

    return {
        "technique": "sentiment_analysis",
        "models": ["nltk/vader", "cardiffnlp/twitter-roberta-base-sentiment-latest"],
        "vader": {
            "compound": round(vader_scores["compound"], 4),
            "positive": round(vader_scores["pos"], 4),
            "negative": round(vader_scores["neg"], 4),
            "neutral": round(vader_scores["neu"], 4),
        },
        "roberta": {
            "label": roberta_result["label"],
            "score": round(roberta_result["score"], 4),
        },
    }


# ---------- 6. Summarization ----------

def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> dict[str, Any]:
    """Abstractive summary using BART-large-CNN."""
    pipe = _get_summarize_pipe()
    # BART-large-CNN handles up to 1024 tokens; truncate input
    result = pipe(text[:3000], max_length=max_length, min_length=min_length, do_sample=False)
    return {
        "technique": "text_summarization",
        "model": "facebook/bart-large-cnn",
        "summary": result[0]["summary_text"],
        "input_chars": len(text),
        "summary_chars": len(result[0]["summary_text"]),
    }


# ---------- 7. Keyword extraction ----------

def extract_keywords(text: str, top_n: int = 10) -> dict[str, Any]:
    """Extract keywords using KeyBERT (MiniLM embeddings)."""
    model = _get_keyword_model()
    keywords = model.extract_keywords(
        text[:5000],
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return {
        "technique": "keyword_extraction",
        "model": "keybert/all-MiniLM-L6-v2",
        "keywords": [{"keyword": kw, "score": round(sc, 4)} for kw, sc in keywords],
    }


# ---------- 8. Readability scoring ----------

def readability_score(text: str) -> dict[str, Any]:
    """Flesch-Kincaid grade level + Flesch Reading Ease. Pure Python, no deps."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    if not sentences or not words:
        return {
            "technique": "readability_scoring",
            "error": "not enough text",
        }

    n_sentences = len(sentences)
    n_words = len(words)

    # Count syllables (rough heuristic)
    def count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) <= 3:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_v = ch in vowels
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    n_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease
    fre = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)

    # Flesch-Kincaid Grade Level
    fkgl = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59

    # Interpret
    if fre >= 80:
        difficulty = "Easy (6th grade)"
    elif fre >= 60:
        difficulty = "Standard (8th-9th grade)"
    elif fre >= 40:
        difficulty = "Difficult (college level)"
    elif fre >= 20:
        difficulty = "Very difficult (graduate level)"
    else:
        difficulty = "Extremely difficult (professional/legal)"

    return {
        "technique": "readability_scoring",
        "model": "flesch-kincaid (pure python)",
        "flesch_reading_ease": round(fre, 2),
        "flesch_kincaid_grade": round(fkgl, 1),
        "difficulty": difficulty,
        "stats": {
            "sentences": n_sentences,
            "words": n_words,
            "syllables": n_syllables,
        },
    }


# ---------- 9. Emotion detection ----------

def emotion_detect(text: str) -> dict[str, Any]:
    """Emotion classification using j-hartmann distilroberta."""
    pipe = _get_emotion_pipe()
    results = pipe(text[:1500])
    # results is a list of lists for top_k=None
    emotions = results[0] if results else []
    sorted_emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)
    return {
        "technique": "emotion_detection",
        "model": "j-hartmann/emotion-english-distilroberta-base",
        "top_emotion": sorted_emotions[0]["label"] if sorted_emotions else "unknown",
        "confidence": round(sorted_emotions[0]["score"], 4) if sorted_emotions else 0,
        "all_emotions": [
            {"emotion": e["label"], "score": round(e["score"], 4)}
            for e in sorted_emotions
        ],
    }


# ---------- Run all (convenience) ----------

def analyze_all(query: str, context: str, corpus_chunks: list[str] | None = None) -> dict[str, Any]:
    """Run all 9 analyses and return combined results."""
    results = {}

    results["intent"] = intent_classify(query)
    results["ner"] = extract_entities(context)
    results["extractive_qa"] = extractive_qa(query, context)
    results["sentiment"] = sentiment_analysis(context)
    results["summary"] = summarize_text(context)
    results["keywords"] = extract_keywords(context)
    results["readability"] = readability_score(context)
    results["emotion"] = emotion_detect(context)

    if corpus_chunks and len(corpus_chunks) >= 5:
        results["topics"] = topic_model(corpus_chunks)
    else:
        results["topics"] = {"technique": "topic_modeling_lda", "note": "requires full corpus (pass corpus_chunks)"}

    return results