const API_BASE_URL = 'http://localhost:8000';
let currentSearchWords = [];

async function search() {
    const searchWord = document.getElementById('searchWord').value.trim();
    const maxWords = document.getElementById('maxWords').value;
    
    if (!searchWord) {
        alert('Пожалуйста, введите слово для поиска');
        return;
    }

    currentSearchWords = [searchWord.toLowerCase()];
    await performSearch();
}

async function performSearch() {
    const maxWords = document.getElementById('maxWords').value;
    const loadingElement = document.getElementById('loading');
    const wordsTree = document.getElementById('wordsTree');
    const sentencesResults = document.getElementById('sentencesResults');

    loadingElement.classList.remove('hidden');
    wordsTree.innerHTML = '<p>Поиск...</p>';
    sentencesResults.innerHTML = '<p>Поиск...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                search_words: currentSearchWords,
                max_words: parseInt(maxWords) || 10
            })
        });

        if (!response.ok) {
            throw new Error(`Ошибка: ${response.status}`);
        }

        const data = await response.json();

        // Обновляем путь поиска
        updateCurrentPath();

        // Отображаем дерево слов
        displayWordTree(data.word_tree);

        // Отображаем предложения
        displaySentences(data.sentences);

    } catch (error) {
        console.error('Ошибка:', error);
        wordsTree.innerHTML = `<p style="color: red;">Ошибка: ${error.message}</p>`;
        sentencesResults.innerHTML = `<p style="color: red;">Ошибка при поиске</p>`;
    } finally {
        loadingElement.classList.add('hidden');
    }
}

function displayWordTree(wordTree) {
    const wordsTree = document.getElementById('wordsTree');
    
    if (!wordTree || wordTree.length === 0) {
        wordsTree.innerHTML = '<p>Слова не найдены</p>';
        return;
    }

    wordsTree.innerHTML = '';
    wordTree.forEach(node => {
        wordsTree.appendChild(createWordNode(node, 0));
    });
}

function createWordNode(node, level) {
    const nodeElement = document.createElement('div');
    nodeElement.className = 'word-node';
    
    const itemElement = document.createElement('div');
    itemElement.className = 'word-item';
    itemElement.style.paddingLeft = (level * 15) + 'px';
    
    itemElement.innerHTML = `
        <span class="word-text">${node.word}</span>
        <span class="word-count">${node.count}</span>
    `;
    
    // Обработчик клика для раскрытия/закрытия
    itemElement.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleWordNode(node, itemElement, nodeElement, level);
    });
    
    nodeElement.appendChild(itemElement);
    
    // Если есть дети, создаем контейнер для них
    if (node.children && node.children.length > 0) {
        const childrenContainer = document.createElement('div');
        childrenContainer.className = 'word-children';
        childrenContainer.style.display = 'none';
        
        node.children.forEach(child => {
            childrenContainer.appendChild(createWordNode(child, level + 1));
        });
        
        nodeElement.appendChild(childrenContainer);
    }
    
    return nodeElement;
}

async function toggleWordNode(node, itemElement, nodeElement, level) {
    const childrenContainer = nodeElement.querySelector('.word-children');
    
    // Если узел уже раскрыт - закрываем
    if (itemElement.classList.contains('expanded')) {
        itemElement.classList.remove('expanded');
        if (childrenContainer) {
            childrenContainer.style.display = 'none';
        }
        return;
    }
    
    // Добавляем слово в текущий поиск
    const newSearchWords = [...currentSearchWords, node.word];
    
    try {
        const maxWords = document.getElementById('maxWords').value;
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                search_words: newSearchWords,
                max_words: parseInt(maxWords) || 10
            })
        });

        if (!response.ok) {
            throw new Error(`Ошибка: ${response.status}`);
        }

        const data = await response.json();
        
        // Раскрываем узел
        itemElement.classList.add('expanded');
        
        // Создаем или обновляем контейнер детей
        let childrenContainer = nodeElement.querySelector('.word-children');
        if (!childrenContainer) {
            childrenContainer = document.createElement('div');
            childrenContainer.className = 'word-children';
            childrenContainer.style.display = 'block';
            nodeElement.appendChild(childrenContainer);
        } else {
            childrenContainer.style.display = 'block';
        }
        
        // Очищаем и добавляем детей
        childrenContainer.innerHTML = '';
        data.word_tree.forEach(child => {
            childrenContainer.appendChild(createWordNode(child, level + 1));
        });
        
        // Обновляем предложения для нового набора слов
        displaySentences(data.sentences);
        
    } catch (error) {
        console.error('Ошибка при раскрытии узла:', error);
    }
}

function displaySentences(sentences) {
    const sentencesResults = document.getElementById('sentencesResults');
    
    if (!sentences || sentences.length === 0) {
        sentencesResults.innerHTML = '<p>Предложения не найдены</p>';
        return;
    }

    sentencesResults.innerHTML = sentences.map(sentence => `
        <div class="sentence-item">
            <div class="sentence-original">${sentence.original}</div>
        </div>
    `).join('');
}

function updateCurrentPath() {
    const pathText = document.getElementById('pathText');
    
    if (currentSearchWords.length === 0) {
        pathText.innerHTML = '-';
        return;
    }
    
    // Создаем хлебные крошки
    const breadcrumbs = currentSearchWords.map((word, index) => {
        const wordsUpToHere = currentSearchWords.slice(0, index + 1);
        return `<span class="breadcrumb" onclick="navigateToPath(${index})">${word}</span>`;
    }).join('<span class="breadcrumb-separator">→</span>');
    
    pathText.innerHTML = breadcrumbs;
}

function navigateToPath(index) {
    // Обрезаем путь до выбранного элемента
    currentSearchWords = currentSearchWords.slice(0, index + 1);
    performSearch();
}

function clearSearch() {
    currentSearchWords = [];
    document.getElementById('searchWord').value = '';
    document.getElementById('wordsTree').innerHTML = '<p>Введите слово для поиска...</p>';
    document.getElementById('sentencesResults').innerHTML = '<p>Предложения появятся здесь...</p>';
    document.getElementById('pathText').innerHTML = '-';
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

document.addEventListener('DOMContentLoaded', checkHealth);