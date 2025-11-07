const API_BASE_URL = 'http://localhost:8000';

async function search() {
    const searchWord = document.getElementById('searchWord').value.trim();
    const maxWords = document.getElementById('maxWords').value;
    
    if (!searchWord) {
        alert('Пожалуйста, введите слово для поиска');
        return;
    }

    const loadingElement = document.getElementById('loading');
    const wordsResults = document.getElementById('wordsResults');
    const sentencesResults = document.getElementById('sentencesResults');

    // Показываем загрузку
    loadingElement.classList.remove('hidden');
    wordsResults.innerHTML = '<p>Поиск...</p>';
    sentencesResults.innerHTML = '<p>Поиск...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                search_word: searchWord,
                max_words: parseInt(maxWords) || 10
            })
        });

        if (!response.ok) {
            throw new Error(`Ошибка: ${response.status}`);
        }

        const data = await response.json();

        // Отображаем слова
        if (data.words && data.words.length > 0) {
            wordsResults.innerHTML = data.words.map(word => `
                <div class="word-item">
                    <span class="word-text">${word.word}</span>
                    <span class="word-count">${word.count}</span>
                </div>
            `).join('');
        } else {
            wordsResults.innerHTML = '<p>Слова не найдены</p>';
        }

        // Отображаем предложения
        if (data.sentences && data.sentences.length > 0) {
            sentencesResults.innerHTML = data.sentences.map(sentence => `
                <div class="sentence-item">${sentence}</div>
            `).join('');
        } else {
            sentencesResults.innerHTML = '<p>Предложения не найдены</p>';
        }

    } catch (error) {
        console.error('Ошибка:', error);
        wordsResults.innerHTML = `<p style="color: red;">Ошибка: ${error.message}</p>`;
        sentencesResults.innerHTML = `<p style="color: red;">Ошибка при поиске</p>`;
    } finally {
        loadingElement.classList.add('hidden');
    }
}

// Поиск при нажатии Enter
document.getElementById('searchWord').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        search();
    }
});

// Проверка здоровья API при загрузке
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('API подключен успешно');
        }
    } catch (error) {
        console.warn('Не удалось подключиться к API:', error);
    }
}

// Проверяем здоровье при загрузке страницы
document.addEventListener('DOMContentLoaded', checkHealth);