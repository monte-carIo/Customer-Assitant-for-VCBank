
from bs4 import BeautifulSoup
import requests
import re
import csv

qa_list = []
for i in range(1,20):
    url = f'https://portal.vietcombank.com.vn/FAQs/Pages/cau-hoi-thuong-gap.aspx?Page={i}&devicechannel=default'
    page = requests.get(url)
    soup_p = BeautifulSoup(page.text, 'html')
    html_content = str(soup_p).split('Câu hỏi thường gặp')[1]
    soup = BeautifulSoup(html_content, 'html.parser')
    
    items = soup.find_all('div', class_='item')
    for item in items:
        question_split = item.find('p', class_='title').get_text(strip=True).split('(')
        question = question_split[0]
        for i in range(1,len(question_split)-1):
            question += '(' + question_split[i]
        answer = item.find('div', class_='des').get_text(strip=True)
        qa_list.append((question, answer))

csv_file = 'questions_answers.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['question', 'answer'])
    writer.writerows(qa_list)

print(f"Data has been saved to {csv_file}")