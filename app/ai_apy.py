import openai
import config

openai.api_key = config.DevelopmentConfig.OPENAI_API_KEY


def generate_response(prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    question = {'role': "user", 'content': prompt}
    messages.append(question)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    try:
        answer = response['choices'][0]['message']['content']
    except:
        answer = 'Failure! Try with a different question.\nIf the problem persists, please contact us.'

    return answer

