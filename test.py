import invertedindex as ii

index = ii.InvertedIndex()
#index = ii.AdvancedInvertedIndex() #ii.InvertedIndex()

documents = {
    1: "Python is a programming language",
    2: "Java is another programming language",
    3: "Python and Java are both popular",
    4: "Programming languages are important for developers"
}

index.add_documents(documents=documents)

# Поиск
print("Документы содержащие 'python':", index.search("python"))
print("Документы содержащие 'programming':", index.search("programming"))
print("Документы содержащие 'python programming':", index.search("python programming"))

# Вывод всего индекса
print("\nПолный индекс:")
index.print_index()