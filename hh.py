from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_name = "muhamaddkhaledd/skin-diseases-chatbot-s3"  # Change to your fine-tuned model or local path
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Encode input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate long response
    output = model.generate(
        input_ids,
        max_length=500,  # You can adjust this based on the response length you want
        num_return_sequences=1,
        do_sample=True,
        top_k=50,  # You can increase or decrease this value to control randomness
        top_p=0.95,  # Nucleus sampling, controlling randomness
        temperature=0.7,  # Lower temperature makes the model more deterministic
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2  # This helps prevent repetition of n-grams
    )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
