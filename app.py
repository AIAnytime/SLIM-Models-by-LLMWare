import streamlit as st

from llmware_module import (
    classify_sentiment,
    detect_emotions,
    generate_tags,
    identify_topics,
    perform_intent,
    get_ratings,
    get_category,
    perform_ner,
    perform_nli,
)

# Streamlit app layout
st.title("Perform NLP Tasks on CPU")

# Text input
text = st.text_area("Enter text here:")

# Analysis tools selection
analysis_tools = st.multiselect(
    "Select the analysis tools to use:",
    ["Sentiment Analysis", "Emotion Detection", "Generate Tags", "Identify Topics",
     "Perform Intent", "Get Ratings", "Get Category",
     "Perform NER", "Perform NLI"],
    ["Sentiment Analysis"]  # Default selection
)

# Execute analysis and display results
if st.button("Analyze"):
    results = {}
    
    if "Sentiment Analysis" in analysis_tools:
        results["Sentiment Analysis"] = classify_sentiment(text)
    if "Emotion Detection" in analysis_tools:
        results["Emotion Detection"] = detect_emotions(text)
    if "Generate Tags" in analysis_tools:
        results["Generate Tags"] = generate_tags(text)
    if "Identify Topics" in analysis_tools:
        results["Identify Topics"] = identify_topics(text)
    if "Perform Intent" in analysis_tools:
        results["Perform Intent"] = perform_intent(text)
    if "Get Ratings" in analysis_tools:
        results["Get Ratings"] = get_ratings(text)
    if "Get Category" in analysis_tools:
        results["Get Category"] = get_category(text)
    if "Perform NER" in analysis_tools:
        results["Perform NER"] = perform_ner(text)
    if "Perform NLI" in analysis_tools:
        results["Perform NLI"] = perform_nli(text)
    
    for tool, response in results.items():
        st.subheader(tool)
        st.json(response)
