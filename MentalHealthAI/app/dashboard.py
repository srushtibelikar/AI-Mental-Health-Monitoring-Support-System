import streamlit as st
import sys
import os
import plotly.graph_objects as go
from nltk.sentiment import SentimentIntensityAnalyzer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_mental_health
from src.preprocessing import clean_text

sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="AI Mental Health Monitor", page_icon="🧠")

st.title("🧠 AI Mental Health Monitoring & Support System")
st.write("Your personal emotional wellness assistant 💙")

st.markdown("---")

user_input = st.text_area("💬 Express your thoughts here:")

if st.button("Analyze My Mood"):

    if user_input.strip() != "":

        result = predict_mental_health(user_input)

        cleaned = clean_text(user_input)
        sentiment_score = sia.polarity_scores(cleaned)["compound"]

        # 📊 Sentiment Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={'text': "Sentiment Score (-1 to +1)"},
            gauge={
                'axis': {'range': [-1, 1]},
            }
        ))

        st.plotly_chart(fig)

        st.markdown("---")

        # 🎯 Risk Output Section
        if result == "High":
            st.error("🚨 High Stress / Risk Detected")
            st.write("Please consider talking to someone you trust or a professional.")

            st.markdown("### 🌿 Immediate Stress Relief Tips:")
            st.write("• Take 5 deep breaths slowly.")
            st.write("• Go for a short walk.")
            st.write("• Drink some water.")
            st.write("• Call a close friend or family member.")

            st.markdown("### 💙 Remember:")
            st.info("You are not alone. Tough times don’t last, but strong people do.")

        elif result == "Medium":
            st.warning("⚠ Moderate Stress Level Detected")

            st.markdown("### 🌿 Suggested Actions:")
            st.write("• Try 10 minutes of meditation.")
            st.write("• Listen to calming music.")
            st.write("• Organize your tasks step by step.")
            st.write("• Take a small break from work.")

            st.markdown("### 💡 Motivation:")
            st.info("Progress, not perfection. You are doing better than you think.")

        else:
            st.success("😊 Low Stress Level – You’re Doing Great!")

            st.markdown("### 🌸 Keep It Up:")
            st.write("• Maintain your healthy habits.")
            st.write("• Practice gratitude.")
            st.write("• Keep connecting with positive people.")

            st.markdown("### 🌟 Positive Affirmation:")
            st.info("Happiness looks good on you. Keep shining!")

    else:
        st.warning("Please enter some text to analyze.")
