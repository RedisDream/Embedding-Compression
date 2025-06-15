#!pip install -q transformers datasets scikit-learn umap-learn matplotlib
#я пытался проводить обучение локально

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from umap import UMAP
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import time
import warnings

#хотелось чтобы код запустился, были проблемы с дебагом
warnings.filterwarnings('ignore')
#нашел в интернете фрагмент кода для проверки работаем мы на cpu или gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

print("Загрузка данных...")#подгружаем датасет, убираем ненужное
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data
labels = newsgroups.target

#рамзер датасета который мы используем (можно увеличить, с тем увеличится и время обучения)
n_samples = 2000  #примерно 18 тысяч писем, классифицированных по теме
texts = texts[:n_samples]
labels = labels[:n_samples]

#датасет предусматривает внутреннее деление на train и test, но используем
#train split из sklearn тк отладка была не на всём датасете, а на его части
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

#BERT
print("\nЗагрузка модели BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

#функция для получения эмбеддингов
def get_embeddings(text_list, batch_size=16):
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        #получаем вектор из берта
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)

print("\nСоздание эмбеддингов для тренировочных данных...")
start_time = time.time()
train_embeddings = get_embeddings(X_train)
print(f"Затраченное время: {time.time()-start_time:.1f} сек")

print("\nСоздание эмбеддингов для тестовых данных...")
test_embeddings = get_embeddings(X_test)

original_dim = train_embeddings.shape[1]
print(f"\nИсходная размерность эмбеддингов: {original_dim}")

#понижаем размерность в 2,4,8..... раз
reduction_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
results = {'factors': [], 'dims': [], 'accuracies': [], 'times': []}

#функция для оценки классификации 
def evaluate_classification(train_emb, test_emb, y_train, y_test):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, y_train)
    preds = clf.predict(test_emb)
    return accuracy_score(y_test, preds)

print("\nОценка исходной точности...")
acc_original = evaluate_classification(train_embeddings, test_embeddings, y_train, y_test)
print(f"Исходная точность: {acc_original:.4f}")

# Сохранение результатов для исходной размерности
results['factors'].append(1)
results['dims'].append(original_dim)
results['accuracies'].append(acc_original)
results['times'].append(0)

#прогоняем по циклу 
for factor in reduction_factors[1:]:
    n_components = max(2, original_dim // factor)     
    print(f"\nКоэффициент понижения: {factor}x -> осталось {n_components} компонент...")    
    #понижение размерности с помощью UMAP
    start_time = time.time()
    reducer = UMAP(n_components=n_components, random_state=42)
    train_reduced = reducer.fit_transform(train_embeddings)
    test_reduced = reducer.transform(test_embeddings)
    reduction_time = time.time() - start_time    
    #сама оценка точности
    acc = evaluate_classification(train_reduced, test_reduced, y_train, y_test)
    #записываем результаты
    results['factors'].append(factor)
    results['dims'].append(n_components)
    results['accuracies'].append(acc)
    results['times'].append(reduction_time)

    print(f"  Точность: {acc:.4f}")
    print(f"  Время понижения: {reduction_time:.1f} сек")

#графики
plt.figure(figsize=(15, 10))

# График зависимости точности от конечной размерности
plt.subplot(2, 2, 1)
plt.plot(results['dims'], results['accuracies'], 'o-', color='darkorange')
plt.xscale('log')
plt.xlabel('Конечная размерность (log scale)')
plt.ylabel('Точность классификации')
plt.title('Зависимость точности от размерности')
plt.grid(True)

# График времени выполнения
plt.subplot(2, 2, 2)
plt.plot(results['dims'], results['times'], 'o-', color='forestgreen')
plt.xscale('log')
plt.xlabel('Конечная размерность (log scale)')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения от размерности')
plt.grid(True)

plt.tight_layout()
plt.savefig('umap_reduction_analysis.png', dpi=300)
plt.show()

#таблица для красоты
print("\nРезультаты применения UMAP:")
print(f"{'Коэф.':>8} | {'Размерность':>12} | {'Точность':>10} | {'Время (сек)':>10}")
print("-" * 50)
for i in range(len(results['factors'])):
    f = results['factors'][i]
    d = results['dims'][i]
    a = results['accuracies'][i]
    t = results['times'][i]
    print(f"{f:>8} | {d:>12} | {a:>10.4f} | {t:>10.2f}")
