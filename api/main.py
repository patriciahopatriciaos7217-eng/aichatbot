import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chat.chat import chatter


while True:
    result = []
    answer = []
    question = input("user: ")
    if question.lower() == "exit":
        break
    
    answer = chatter(question);
    
    for item in answer:
        result.append(item["name"])
    
    print("Bot: ", result)
    
    