from flask import Flask, render_template, request
from transformers import pipeline # type: ignore

# Initialize the Flask application
app = Flask(__name__)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']  # Get text from the form
    summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
    return render_template('index.html', summary=summary[0]['summary_text'], original_text=text)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
