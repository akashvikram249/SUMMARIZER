import streamlit as st
import requests
import pandas as pd
import speech_recognition as sr
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
import time
import googletrans
from googletrans import Translator

stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')
translator = Translator()

auth_token = "046f137297734d48a2594adb074e6baa"

headers = {
    "authorization": auth_token,
    "content-type": "application/json"
}


def upload_to_AssemblyAI(audio_file):
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    upload_endpoint = "https://api.assemblyai.com/v2/upload"

    print("Uploading")
    upload_response = requests.post(
        upload_endpoint,
        headers=headers, data=audio_file
    )

    audio_url = upload_response.json()["upload_url"]
    print("Done")

    json = {
        "audio_url": audio_url,
        "iab_categories": True,
        "auto_chapters": True
    }

    response = requests.post(transcript_endpoint, json=json, headers=headers)
    print(response.json())

    polling_endpoint = transcript_endpoint+"/"+response.json()["id"]
    return polling_endpoint


def convertMillis(start_ms):
    seconds = int((start_ms/1000) % 60)
    minutes = int((start_ms/(1000*60)) % 60)
    hours = int((start_ms/(1000*60*60)) % 24)
    btn_txt = ''
    if hours > 0:
        btn_txt += f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    else:
        btn_txt += f'{minutes:02d}:{seconds:02d}'
    return btn_txt


if 'start_point' not in st.session_state:
    st.session_state["start_point"] = 0


def update_start(start_t):
    st.session_state["start_point"] = int(start_t/1000)


uploaded_file = st.file_uploader("Upload the file")

if uploaded_file is not None:
    st.audio(uploaded_file, start_time=st.session_state['start_point'])
    polling_endpoint = upload_to_AssemblyAI(uploaded_file)

    status = 'submitted'
    while status != 'completed':
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()['status']

        if status == 'completed':
            st.subheader('Main themes')
            with st.expander('Themes'):
                categories = polling_response.json(
                )['iab_categories_result']['summary']
                for cat in categories:
                    st.markdown("* "+cat)

            st.subheader('Summary notes of this meeting')
            chapters = polling_response.json()['chapters']
            chapters_df = pd.DataFrame(chapters)
            chapters_df['start_str'] = chapters_df['start'].apply(
                convertMillis)
            chapters_df['end_str'] = chapters_df['end'].apply(convertMillis)

            # Translate the summary for each chapter
            for index, row in chapters_df.iterrows():
                summary_en = row['summary']
                summary_ta = translator.translate(summary_en, dest='ta').text
                row['summary_ta'] = summary_ta

            # Display the summary and its translation for each chapter
            for index, row in chapters_df.iterrows():
                with st.expander(row['gist']):
                    st.write(row['summary'])
                    st.write(translator.translate(
                        row['summary'], dest='ta').text)
                    st.button(row['start_str'],
                              on_click=update_start, args=(row['start']))

# Load English and Spanish models
nlp = spacy.load('en_core_web_md')
translator = Translator()

# Set up stop words
stopwords = list(STOP_WORDS)

# Create empty Streamlit components for displaying recorded text and buttons
recorded_text = st.empty()
start_button = st.empty()

# Create a recognizer object and use the microphone as the source for recording
r = sr.Recognizer()

# If "Record audio" button is clicked
if start_button.button("Record audio"):
    recorded_text.text("Speak now...")
    with sr.Microphone() as source:
        # Adjust microphone for ambient noise
        r.adjust_for_ambient_noise(source)

        start_time = time.time()

        # Record audio from the microphone
        audio = r.record(source, duration=15)

    try:
        # Use Google Speech Recognition to convert the recorded audio to text
        text = r.recognize_google(audio)
        recorded_text.text(text)

        # Add period to text automatically if there's a pause of more than 2 seconds
        paused_text = ""
        last_pause = time.time()
        for i, word in enumerate(text.split()):
            # Add the word to the paused_text string
            paused_text += word + " "

            # Check if there has been a pause of more than 2 seconds
            if time.time() - last_pause > 1 and i != len(text.split()) - 1:
                paused_text += ". "
                last_pause = time.time()

        # Translate the text to Spanish

        # Summarize the text
        word_list = paused_text.split()
        chunks = [word_list[i:i + 7] for i in range(0, len(word_list), 7)]
        chunk_scores = {}
        for chunk in chunks:
            chunk_text = ' '.join(chunk)
            sentence_scores = {}
            for sent in chunk_text.split('.'):
                for word in nlp(sent):
                    if word.text.lower() not in stopwords:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word.similarity(nlp(sent))
                        else:
                            sentence_scores[sent] += word.similarity(nlp(sent))
            # Check if sentence_scores is not empty before getting the max
            if sentence_scores:
                chunk_scores[chunk_text] = max(sentence_scores.values())

        summarized_chunks = nlargest(3, chunk_scores, key=chunk_scores.get)
        summary = '. '.join(summarized_chunks)

        st.subheader("Summary")
        st.write(summary)

        # Check if the summary is not empty before translating
        if summary:
            # Translate the summary to Spanish
            translated_text = translator.translate(summary, dest='ta').text
            st.subheader("Translated Text")
            st.write(translated_text)
        else:
            st.warning("No summary available for translation.")
        translated_text = translator.translate(summary, dest='ta').text
        st.write("Translated Text:", translated_text)
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand what you said")
    except sr.RequestError as e:
        st.write("Could not request results; {0}".format(e))

    # Calculate recording duration and display it
    duration = time.time() - start_time
    st.write(f"Recording duration: {duration:.2f} seconds")

    # Clear Streamlit components
    start_button.empty()
    recorded_text.empty()

    # Display original and translated text along with summary
    st.subheader("Original Text")
    st.write(paused_text)
    st.subheader("Summary")
    st.write(summary)
    st.subheader("Translated Text")
    st.write(translated_text)