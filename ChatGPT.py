import openai

# Set your API key
api_key = "sk-proj-YTIlKRvyvEL8n3t2ASKXT3BlbkFJG6lVGa4bqv7lDOCnVRCc"
openai.api_key = api_key

# Create a simple prompt using the latest API
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
)
# Print the response
print(response['choices'][0]['message']['content'])
