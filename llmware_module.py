from llmware.models import ModelCatalog
from llmware.prompts import Prompt

def classify_sentiment(text):
    sentiment_model = ModelCatalog().load_model("slim-sentiment-tool")
    response_sentiment = sentiment_model.function_call(text, get_logits=False)
    return response_sentiment

def detect_emotions(text):
    emotions_model = ModelCatalog().load_model("slim-emotions-tool")
    response_emotions = emotions_model.function_call(text, get_logits=False)
    return response_emotions

def generate_tags(text):
    tags_model = ModelCatalog().load_model("slim-tags-tool")
    response_tags = tags_model.function_call(text, get_logits=False)
    return response_tags

def identify_topics(text):
    topics_model = ModelCatalog().load_model("slim-topics-tool")
    response_topics = topics_model.function_call(text, get_logits=False)
    return response_topics

def perform_intent(text):
    intent_model = ModelCatalog().load_model("slim-intent-tool")
    response_intent = intent_model.function_call(text)
    return response_intent

def get_ratings(text):
    ratings_model = ModelCatalog().load_model("slim-ratings-tool")
    response_ratings = ratings_model.function_call(text)
    return response_ratings

def get_category(text):
    category_model = ModelCatalog().load_model("slim-category-tool")
    response_category = category_model.function_call(text)
    return response_category

def perform_ner(text):
    ner_model = ModelCatalog().load_model("slim-ner-tool")
    response_ner = ner_model.function_call(text)
    return response_ner

def perform_nli(text):
    nli_model = ModelCatalog().load_model("slim-nli-tool")
    response_nli = nli_model.function_call(text)
    return response_nli
