import streamlit as st
import os
from PIL import Image
import tempfile
import numpy as np

st.set_page_config(page_title="Emotion Recognition", layout="wide")

st.title("Multi-Modal Emotion Recognition")
st.write("Upload text, audio, or image to predict emotions")
tab1, tab2, tab3 = st.tabs(["Text Emotion", "Audio Emotion", "Image Emotion"])

# Text Emotion Tab
with tab1:
    
    from text.text_prediction import predict_emotions
    st.header("Text Emotion Prediction")
    
    emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    
    with st.expander("ℹ️ Emotions this model can detect", expanded=False):
        cols = st.columns(4)  
        for i, emotion in enumerate(emotion_labels):
            cols[i % 4].write(f"• {emotion.capitalize()}")
    
    text_input = st.text_area("Enter your text here:", "I'm feeling great today!")
    
    if st.button("Predict Text Emotion"):
        
        with st.spinner('Analyzing emotions...'):
            predicted_emotions, probabilities = predict_emotions(text_input)
        
        st.subheader("Results:")
        if predicted_emotions:

            tags = ""
            for emotion in predicted_emotions:
                color = "#4CAF50" if emotion in ["joy", "love", "amusement"] else "#F44336" if emotion in ["anger", "disgust"] else "#2196F3"
                tags += f"<span style='background-color:{color}; color:white; padding:0.2em 0.4em; border-radius:0.5em; margin-right:0.5em; display:inline-block;'>{emotion.capitalize()}</span>"
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No strong emotions detected (likely neutral)")
        
    
        st.subheader("Emotion Probabilities:")
        
       
        tab_top, tab_all = st.tabs(["Top Emotions", "All Emotions"])
        
        with tab_top:
            
            top_indices = np.argsort(probabilities)[::-1][:5]
            for idx in top_indices:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"{emotion_labels[idx].capitalize()}:")
                with col2:
                    st.progress(float(probabilities[idx]))
                    st.write(f"{probabilities[idx]:.3f}")
        
        with tab_all:
           
            emotion_data = []
            for i, emotion in enumerate(emotion_labels):
                emotion_data.append({
                    "Emotion": emotion.capitalize(),
                    "Probability": f"{probabilities[i]:.3f}",
                    "Detected": "✅" if probabilities[i] > 0.5 else "❌"
                })
            
            st.dataframe(
                emotion_data,
                column_config={
                    "Emotion": "Emotion",
                    "Probability": st.column_config.ProgressColumn(
                        "Probability",
                        min_value=0,
                        max_value=1,
                        format="%.3f"
                    ),
                    "Detected": "Detected?"
                },
                hide_index=True,
                use_container_width=True
            )


# Audio Emotion Tab
with tab2:
    from audio.audio_prediction import predict_emotion
    
    st.header("🎙️ Audio Emotion Recognition")
    
    
    with st.expander("ℹ️ Emotion Classes This Model Recognizes"):
        st.write("""
        - **ANG**: Anger 😠  
        - **DIS**: Disgust 🤢  
        - **FEA**: Fear 😨  
        - **HAP**: Happiness 😄  
        - **SAD**: Sadness 😢  
        - **NEU**: Neutral 😐  
        """)
    
   
    audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
    
    if audio_file is not None:
        
        st.audio(audio_file, format='audio/wav')        
        if st.button("Analyze Audio Emotion"):
            with st.spinner('Processing audio...'):
                try:
                   
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_file.read())
                        audio_path = tmp_file.name
                  
                    result = predict_emotion(audio_path)
                    
                   
                    st.subheader("Results")
                    
                    emotion_emoji = {
                        "ANG": "😠", "DIS": "🤢", "FEA": "😨", 
                        "HAP": "😄", "SAD": "😢", "NEU": "😐"
                    }.get(result['prediction'], "")
                    
                    st.success(f"""
                    **Predicted Emotion**: {result['prediction']} {emotion_emoji}  
                    **Confidence**: {result['confidence']:.1%}
                    """)
                    
                    
                    st.subheader("Detailed Probabilities")
                    
                    sorted_probs = sorted(
                        result['all_probs'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    for emotion, prob in sorted_probs:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.write(f"{emotion}:")
                        with col2:
                            st.progress(prob)
                            st.write(f"{prob:.1%}")
                    
                    
                    os.unlink(audio_path)
                    
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    if 'audio_path' in locals() and os.path.exists(audio_path):
                        os.unlink(audio_path)



# Image Emotion Tab
with tab3:
    from image.image_prediction import predict_emotion 
    from PIL import Image
    import tempfile
    import os
    
    st.header("Image Emotion Prediction")
    
    emotion_map = {
        '0': 'amusement',
        '1': 'awe',
        '2': 'contentment', 
        '3': 'excitement',
        '4': 'anger',
        '5': 'disgust',
        '6': 'fear',
        '7': 'sadness'
    }
    
    with st.expander("ℹ️ Emotion Classes This Model Recognizes"):
        st.write("The model can detect these emotions from images:")
        cols = st.columns(3)
        for emotion in emotion_map.values():
            cols[list(emotion_map.values()).index(emotion) % 3].write(f"• {emotion.capitalize()}")
    
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
       
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image Emotion"):
            with st.spinner('Processing image...'):
                try:
                   
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        image.save(tmp_file, format="JPEG")
                        image_path = tmp_file.name
                    
                    
                    emotion = predict_emotion(image_path)
                    
                    st.subheader("Results")
                    
                    emotion_emoji = {
                        "anger": "😠",
                        "disgust": "🤢",
                        "fear": "😨",
                        "amusement": "😄",
                        "sadness": "😢",
                        "awe": "😲",
                        "contentment": "😌",
                        "excitement": "🤩"
                    }.get(emotion.lower(), "😐")
                    
                    st.success(f"**Predicted Emotion**: {emotion.capitalize()} {emotion_emoji}")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                finally:
                   
                    if 'image_path' in locals() and os.path.exists(image_path):
                        os.unlink(image_path)
                    
                
    



 
