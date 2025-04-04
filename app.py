import streamlit as st
from transformers import pipeline

# Load emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Emoji mapping for emotions
emotion_emojis = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲',
    'disgust': '🤢',
    'neutral': '😐'
}

def analyze_sentiment(text):
    results = emotion_classifier(text[:512])
    emotion_scores = results[0]
    emotion_scores.sort(key=lambda x: x['score'], reverse=True)
    return emotion_scores

def summarize_emotions(emotion_scores):
    dominant_emotion = emotion_scores[0]['label']
    explanation = {
        'joy': "This message expresses happiness or positivity. The person may feel connected, warm, or excited about you.",
        'sadness': "The tone feels heavy. The person may be going through something emotionally difficult or feeling disconnected.",
        'anger': "There may be frustration, defensiveness, or hurt feelings. The message could be a sign of conflict.",
        'fear': "The person may be anxious, insecure, or worried—possibly about your relationship or their emotions.",
        'surprise': "Something unexpected is being processed—could be good or bad, depending on context.",
        'disgust': "There could be avoidance, rejection, or emotional discomfort behind the words.",
        'neutral': "The message has a balanced, non-emotional tone. The person may be calm or unclear about their feelings."
    }
    return explanation.get(dominant_emotion, "This message has a mixed or unclear emotional tone."), dominant_emotion

def suggested_response(emotion):
    suggestions = {
        'joy': "Respond with warmth or shared excitement—keep the good vibes flowing!",
        'sadness': "Show empathy and ask if they’re okay or want to talk.",
        'anger': "Stay calm and ask questions to better understand their perspective.",
        'fear': "Offer reassurance and encourage open communication.",
        'surprise': "Ask for clarification to better understand the unexpected news.",
        'disgust': "Respect boundaries and gently shift the topic if needed.",
        'neutral': "It’s a good time to ask an open-ended question to invite deeper conversation."
    }
    return suggestions.get(emotion, "Respond mindfully and ask how they’re feeling.")

def main():
    st.set_page_config(page_title="📱 Emotion Decoder for Text Messages", layout="centered")
    st.title("💬 Emotion Decoder")
    st.write("Paste a message below to understand how the other person may be feeling.")

    text_input = st.text_area("Paste message here:", height=200)

    if st.button("Analyze Emotion") and text_input.strip():
        with st.spinner("Analyzing emotion..."):
            sentiment_scores = analyze_sentiment(text_input)

            st.subheader("🎭 Detected Emotions:")
            for score in sentiment_scores:
                emoji = emotion_emojis.get(score['label'], '')
                st.write(f"{emoji} **{score['label']}**: {round(score['score'] * 100, 2)}%")

            summary, dominant_emotion = summarize_emotions(sentiment_scores)
            st.subheader("🧠 What This Might Mean:")
            st.write(summary)

            st.subheader("💡 Suggested Response:")
            st.write(suggested_response(dominant_emotion))

if __name__ == "__main__":
    main()
