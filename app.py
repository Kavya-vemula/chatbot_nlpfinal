import streamlit as st
from chatbot import train_model, get_response

# Train the model and get necessary components
@st.cache_resource
def load_model():
    return train_model()

vectorizer, best_rf_model, dataset = load_model()

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit App
st.title("Chatbot Using NLP")
st.write("Enter your message below and get a response from the chatbot.")

# Create a form for user input
with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("You: ", key="user_input")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    response = get_response(best_rf_model, vectorizer, user_input, dataset)
    # Append user input and bot response to conversation history
    st.session_state.conversation_history.append(("You", user_input))
    st.session_state.conversation_history.append(("Bot", response))

# Display conversation history
for sender, message in st.session_state.conversation_history:
    st.write(f"**{sender}:** {message}")

# Optional: Display some model training metrics or summary
st.write("The model has been successfully trained and is ready to respond to your queries!")
