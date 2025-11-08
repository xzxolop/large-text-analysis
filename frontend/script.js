const API_BASE_URL = 'http://localhost:8000';
let currentSearchWords = [];
let wordTreeState = {}; // Храним состояние дерева

async function search() {
    const searchWord = document.getElementById('searchWord').value.trim();
    const maxWords = document.getElementById('maxWords').value;
    
    if (!searchWord) {
        alert('Пожалуйста, введите слово для поиска');
        return;
    }

    currentSearchWords = [searchWord.toLowerCase()];
    wordTreeState = {}; // Сбрасываем состояние дерева
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
        const requestBody = {
            search_words: currentSearchWords,
            max_words: parseInt(maxWords) || 10
        };
        
        if (expandWord) {
            requestBody.expand_word = expandWord;
        }

        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Ошибка: ${response.status}`);
        }

        const data = await response.json();

        // Обновляем путь поиска
        updateCurrentPath();

        // Отображаем дерево слов
        if (expandWord) {
            // Обновляем только детей для раскрытого слова
            updateWordChildren(expandWord, data.word_tree);
        } else {
            // Отображаем полное дерево
            displayWordTree(data.word_tree);
        }

        // Отображаем предложения
        displaySentences(data.sentences);

    } catch (error) {
        console.error('Ошибка:', error);
        document.getElementById('wordsTree').innerHTML = `<p style="color: red;">Ошибка: ${error.message}</p>`;
        document.getElementById('sentencesResults').innerHTML = `<p style="color: red;">Ошибка при поиске</p>`;
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
        const nodeElement = createWordNode(node, 0);
        wordsTree.appendChild(nodeElement);
        
        // Сохраняем состояние узла
        const nodePath = getNodePath(node);
        wordTreeState[nodePath] = {
            element: nodeElement,
            children: node.children || [],
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
    
    // Добавляем стрелку для узлов, у которых могут быть дети
    const arrow = node.has_children ? 
        '<span class="arrow">▶</span>' : 
        '<span class="arrow" style="visibility: hidden;">▶</span>';
    
    itemElement.innerHTML = `
        ${arrow}
        <span class="word-text">${node.word}</span>
        <span class="word-count">${node.count}</span>
    `;
    
    // Обработчик клика для раскрытия/закрытия
    if (node.has_children) {
        itemElement.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleWordNode(node, itemElement, nodeElement, level);
        });
        itemElement.style.cursor = 'pointer';
    }
    
    nodeElement.appendChild(itemElement);
    
    // Контейнер для детей
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
    
    // Если узел уже раскрыт - закрываем
    if (wordTreeState[nodePath]?.expanded) {
        wordTreeState[nodePath].expanded = false;
        itemElement.classList.remove('expanded');
        childrenContainer.style.display = 'none';
        arrow.textContent = '▶';
        return;
    }
    
    // Раскрываем узел
    wordTreeState[nodePath] = {
        element: nodeElement,
        children: [],
        expanded: true
    };
    
    itemElement.classList.add('expanded');
    arrow.textContent = '▼';
    
    // Показываем загрузку в контейнере детей
    childrenContainer.innerHTML = '<div style="padding: 10px; color: #666;">Загрузка...</div>';
    childrenContainer.style.display = 'block';
    
    try {
        // Запрашиваем детей для этого узла
        await performSearch(node.word);
        
    } catch (error) {
        console.error('Ошибка при раскрытии узла:', error);
        childrenContainer.innerHTML = '<div style="padding: 10px; color: red;">Ошибка загрузки</div>';
    }
}

function updateWordChildren(parentWord, children) {
    const nodePath = currentSearchWords.join('+') + '+' + parentWord;
    const parentNode = wordTreeState[nodePath];
    
    if (!parentNode) return;
    
    const childrenContainer = parentNode.element.querySelector('.word-children');
    
    if (!children || children.length === 0) {
        childrenContainer.innerHTML = '<div style="padding: 10px; color: #666;">Нет дополнительных слов</div>';
        return;
    }
    
    // Очищаем и добавляем детей
    childrenContainer.innerHTML = '';
    children.forEach(child => {
        const childElement = createWordNode(child, parseInt(parentNode.element.dataset.level) + 1);
        childrenContainer.appendChild(childElement);
        
        // Сохраняем состояние ребенка
        const childPath = nodePath + '+' + child.word;
        wordTreeState[childPath] = {
            element: childElement,
            children: child.children || [],
            expanded: false
        };
    });
}

function getNodePath(node) {
    return [...currentSearchWords, node.word].join('+');
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
    
    pathText.innerHTML = currentSearchWords.join(' + ');
}

function clearSearch() {
    currentSearchWords = [];
    wordTreeState = {};
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