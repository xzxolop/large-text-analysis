# large-text-analysis

## Идея приложения

Необходимо провести что-то на подобии кластерного анализа текстов. На вход программе передается слово на английском языке. Например слово data. Далее я получаю слова в контексте с которыми это слово встречается чаще всего:

data
> big 1000 совпадений
> python 200 совпадений

далее я могу раскрыть один из пунктов и получить следующие слова с которыми чаще всего идут совпадения. Например нажав на big: 

data
> big 1000
    > job 500
    > ML 200
    > ...
> python 200

Мой текст разбивается на sentance (предложения) используя sentance tokenization, world tokenization. Каждое предложение будем независимо обрабатывать с помощью TF-IDF индекса, а также используем ElasticSearch. 

Демонстрацию планируется написать с использованием streamlit.

## Сборка проекта

### С использованием uv

uv sync

uv run ./backend/run.py

### С использованием pip

pip install kagglehub pandas nltk uvicorn fastapi pydantic

Если версии python и pip не совпадают то используйте

python -m pip install kagglehub

### Структура проекта

large-text-analysis/
├── backend/
│   ├── the-reddit-dataset-dataset-comments.csv  # ← ПЕРЕНЕСИТЕ ФАЙЛ СЮДА
│   ├── app/
│   │   ├── main.py
│   │   ├── core.py
│   │   └── invertedindex.py
│   └── run.py
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js

### Запуск проекта

pip install -r ./backend/requirements.txt
python ./backend/run.py
python -m http.server 8080 --directory frontend

Backend:

bash
```
cd backend
pip install -r requirements.txt
python run.py

```

Frontend: 

bash
cd frontend
python -m http.server 8080

Перейти в папку /frontend/index.html


### Запуск демонстрации в консоли

```
python .\backend\console-demo\main.py
```