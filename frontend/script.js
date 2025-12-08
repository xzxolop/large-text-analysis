const API_BASE_URL = 'http://localhost:8000';
let currentBaseWords = [];
let wordTreeState = {};

async function search() {
    const searchWord = document.getElementById('searchWord').value.trim();
    const maxWords = document.getElementById('maxWords').value;
    
    if (!searchWord) {
        alert('Пожалуйста, введите слово для поиска');
        return;
    }

    currentBaseWords = [searchWord.toLowerCase()];
    wordTreeState = {};
    await performSearch();
}

async function performSearch(expandWord = null) {
    const maxWords = document.getElementById('maxWords').value;
    const loadingElement = document.getElementById('loading');
    const wordsTree = document.getElementById('wordsTree');
    const sentencesResults = document.getElementById('sentencesResults');

    loadingElement.classList.remove('hidden');
    
    if (!expandWord) {
        wordsTree.innerHTML = '<p>Поиск...</p>';
    }
    sentencesResults.innerHTML = '<p>Поиск...</p>';

    try {
        // Формируем корректный запрос согласно новой структуре API
        const requestBody = {
            base_words: currentBaseWords,
            max_words: parseInt(maxWords) || 10
        };
        
        // Добавляем expand_word только если он есть
        if (expandWord) {
            requestBody.expand_word = expandWord;
        }

        console.log('Отправка запроса:', JSON.stringify(requestBody, null, 2));

        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            let errorMessage = `Ошибка: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMessage = `Ошибка ${response.status}: ${errorData.detail || JSON.stringify(errorData)}`;
            } catch (e) {
                const errorText = await response.text();
                errorMessage = `Ошибка ${response.status}: ${errorText}`;
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        console.log('Получен ответ:', data);

        // Обновляем путь поиска
        updateCurrentPath(expandWord);

        // Отображаем дерево слов
        if (expandWord) {
            updateWordChildren(expandWord, data.word_tree);
        } else {
            displayWordTree(data.word_tree);
        }

        // Отображаем предложения
        displaySentences(data.sentences);

    } catch (error) {
        console.error('Полная ошибка:', error);
        const errorMessage = error.message.includes('422') ? 
            'Ошибка валидации данных. Проверьте структуру запроса.' : 
            error.message;
            
        document.getElementById('wordsTree').innerHTML = `<p style="color: red;">${errorMessage}</p>`;
        document.getElementById('sentencesResults').innerHTML = `<p style="color: red;">Ошибка при поиске</p>`;
    } finally {
        loadingElement.classList.add('hidden');
    }
}

// Остальные функции остаются без изменений...
function displayWordTree(wordTree) {
    const wordsTree = document.getElementById('wordsTree');
    
    if (!wordTree || wordTree.length === 0) {
        wordsTree.innerHTML = '<p>Слова не найдены</p>';
        return;
    }

    wordsTree.innerHTML = '';
    wordTree.forEach(node => {
        const nodeElement = createWordNode(node, 0);
        wordsTree.appendChild(nodeElement);
        
        const nodePath = getNodePath(node);
        wordTreeState[nodePath] = {
            element: nodeElement,
            children: [],
            expanded: false
        };
    });
}

function createWordNode(node, level) {
    const nodeElement = document.createElement('div');
    nodeElement.className = 'word-node';
    nodeElement.dataset.level = level;
    nodeElement.dataset.word = node.word;
    
    const itemElement = document.createElement('div');
    itemElement.className = 'word-item';
    itemElement.style.paddingLeft = (level * 20) + 'px';
    
    const arrow = node.has_children ? 
        '<span class="arrow">▶</span>' : 
        '<span class="arrow" style="visibility: hidden;">▶</span>';
    
    itemElement.innerHTML = `
        ${arrow}
        <span class="word-text">${node.word}</span>
        <span class="word-count">${node.count}</span>
    `;
    
    if (node.has_children) {
        itemElement.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleWordNode(node, itemElement, nodeElement, level);
        });
        itemElement.style.cursor = 'pointer';
    } else {
        itemElement.style.cursor = 'default';
    }
    
    nodeElement.appendChild(itemElement);
    
    const childrenContainer = document.createElement('div');
    childrenContainer.className = 'word-children';
    childrenContainer.style.display = 'none';
    nodeElement.appendChild(childrenContainer);
    
    return nodeElement;
}

async function toggleWordNode(node, itemElement, nodeElement, level) {
    const childrenContainer = nodeElement.querySelector('.word-children');
    const arrow = itemElement.querySelector('.arrow');
    const nodePath = getNodePath(node);
    
    if (wordTreeState[nodePath]?.expanded) {
        wordTreeState[nodePath].expanded = false;
        itemElement.classList.remove('expanded');
        childrenContainer.style.display = 'none';
        arrow.textContent = '▶';
        return;
    }
    
    wordTreeState[nodePath] = {
        element: nodeElement,
        children: [],
        expanded: true
    };
    
    itemElement.classList.add('expanded');
    arrow.textContent = '▼';
    
    childrenContainer.innerHTML = '<div style="padding: 10px; color: #666;">Загрузка...</div>';
    childrenContainer.style.display = 'block';
    
    try {
        await performSearch(node.word);
    } catch (error) {
        console.error('Ошибка при раскрытии узла:', error);
        childrenContainer.innerHTML = '<div style="padding: 10px; color: red;">Ошибка загрузки</div>';
        wordTreeState[nodePath].expanded = false;
        itemElement.classList.remove('expanded');
        arrow.textContent = '▶';
    }
}

function updateWordChildren(parentWord, children) {
    const nodePath = getNodePath({ word: parentWord });
    const parentNode = wordTreeState[nodePath];
    
    if (!parentNode) {
        console.error('Родительский узел не найден:', nodePath);
        return;
    }
    
    const childrenContainer = parentNode.element.querySelector('.word-children');
    
    if (!children || children.length === 0) {
        childrenContainer.innerHTML = '<div style="padding: 10px; color: #666;">Нет дополнительных слов</div>';
        return;
    }
    
    childrenContainer.innerHTML = '';
    children.forEach(child => {
        const childElement = createWordNode(child, parseInt(parentNode.element.dataset.level) + 1);
        childrenContainer.appendChild(childElement);
        
        const childPath = nodePath + '+' + child.word;
        wordTreeState[childPath] = {
            element: childElement,
            children: [],
            expanded: false
        };
    });
}

function getNodePath(node) {
    return [...currentBaseWords, node.word].join('+');
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

function updateCurrentPath(expandWord = null) {
    const pathText = document.getElementById('pathText');
    
    if (currentBaseWords.length === 0) {
        pathText.innerHTML = '-';
        return;
    }
    
    let path = currentBaseWords.join(' + ');
    if (expandWord) {
        path += ` + ${expandWord}`;
    }
    
    pathText.innerHTML = path;
}

function clearSearch() {
    currentBaseWords = [];
    wordTreeState = {};
    document.getElementById('searchWord').value = '';
    document.getElementById('wordsTree').innerHTML = '<p>Введите слово для поиска...</p>';
    document.getElementById('sentencesResults').innerHTML = '<p>Предложения появятся здесь...</p>';
    document.getElementById('pathText').innerHTML = '-';
}

document.getElementById('searchWord').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        search();
    }
});

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