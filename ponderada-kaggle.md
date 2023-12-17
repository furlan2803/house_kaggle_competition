# Ponderada Kaggle

# Compreensão do desafio - House Kaggle

Ao examinar o código, entende-se que o desafio consiste em prever os preços de residências em Ames, Iowa, com base em 79 variáveis que abrangem diversos aspectos relacionados às próprias habitações e seus arredores. Essas variáveis englobam desde o número de quartos e o tamanho do quintal até a localização da casa, considerando elementos como a proximidade de uma ferrovia ou a existência de um parque nas proximidades. Dada a complexidade desses fatores e a não evidência de todos os seus impactos, o objetivo é criar um modelo capaz de fornecer uma faixa de valores para essas residências com base em suas características específicas.

# Métodos de treinamento - Essemble 

No código fornecido no arquivo "houses_kaggle_competition.ipynb", estão disponíveis oito métodos de treinamento. Abaixo, são listados juntamente com breve descrição de seu funcionamentos.

## Árvore de Decisão

Árvore de decisão é um modelo que toma decisões com base em regras de decisão hierárquicas. Cada nó da árvore representa uma condição sobre um atributo e cada ramo representa o resultado dessa condição. As folhas da árvore contêm a predição. Os parâmetros mais relevantes para o método são, max_depth (profundidade máxima da árvore), min_samples_leaf (número mínimo de amostras em uma folha).

## KNN (K-Nearest Neighbors)

KNN é um algoritmo de aprendizado supervisionado que classifica os pontos de dados com base na proximidade com os vizinhos mais próximos. A predição é feita pela maioria das classes dos vizinhos mais próximos. O método é sensível à escala dos recursos, computacionalmente caro para grandes conjuntos de dados.

## Ridge (Regressão Linear com Regularização L2)

Regressão Ridge é uma extensão da regressão linear que adiciona um termo de regularização L2 à função de custo. Isso ajuda a evitar overfitting, especialmente quando há multicolinearidade entre os recursos. Porém, é sensível à escala dos recursos.

## SVM (Support Vector Machine)

Máquinas de Vetores de Suporte são utilizadas para problemas de classificação e regressão. Buscam encontrar o hiperplano que melhor separa as classes ou se ajusta aos dados de regressão. Método sensível à escala dos recursos, escolha do kernel influencia o desempenho.

## Floresta Aleatória

Floresta Aleatória é um conjunto de árvores de decisão treinadas de forma independente. A predição é a média ou a moda das previsões das árvores individuais. Uma das vantagens do método é que reduz overfitting e captura relações não lineares.

## Boosted Trees (AdaBoost e Gradient Boosting)

Os métodos de Boosting combinam modelos fracos para formar um modelo forte. AdaBoost foca em exemplos mal classificados, enquanto Gradient Boosting ajusta os erros residuais de modelos anteriores. Uma das vantagens do método é ser menos propenso a overfitting.

## Stacking

Stacking combina as previsões de vários modelos de base usando um meta-modelo. Pode melhorar o desempenho combinando diferentes perspectivas dos modelos individuais, porém, pode ser computacionalmente intensivo.

## XGBoost

Implementação otimizada de Gradient Boosting que oferece eficiência computacional, regularização, tratamento de dados ausentes e paralelismo. Eficiente para grandes conjuntos de dados e suporte a early stopping.

# Diferença de treinamento Teórica - Essemble 

Apresenta-se abaixo as principais diferenças de treinamento entre os métodos de Ensemble:

## Diversidade dos modelos

- **Árvores de decisão, KNN, Ridge, SVM:** Modelos individuais que não se baseiam em outros para fazer previsões.
- **Random Forest e Boosted Trees:** Constroem uma coleção de modelos fracos (Árvores de decisão) sequencialmente, onde cada modelo aprende a corrigir os erros do anterior.
- **Stacking:** Combina as previsões de vários modelos individuais (como gboost, adaboost, ridge, svm) para gerar uma previsão final.
- **XGBoost:** Utiliza árvores de decisão como modelos base.

## Natureza do aprendizado:

- **Árvores de decisão, KNN, Ridge, SVM:** Aprendem independentemente a partir do conjunto de dados completo.
- **Random Forest e Boosted Trees:** Cada modelo base é treinado em uma subamostra aleatória do conjunto de dados (bagging) ou a partir da distribuição de erros do modelo anterior (boosting).
- **Stacking:** Os modelos base são treinados individualmente no conjunto de dados completo, e então um meta-modelo (como Regressão Linear) é treinado para combinar suas previsões.
- **XGBoost:** É um método de boosting.

## Configuração dos hiperparâmetros:

- **Árvores de decisão, KNN, Ridge, SVM:** Cada modelo tem seus próprios hiperparâmetros que precisam ser ajustados (por exemplo, profundidade máxima da árvore, número de vizinhos, constante de regularização).
- **Random Forest e Boosted Trees:** Além dos hiperparâmetros dos modelos base, também há hiperparâmetros que controlam a construção do ensemble (por exemplo, número de árvores, taxa de aprendizagem).
- **Stacking:** Cada modelo base tem seus próprios hiperparâmetros, e o meta-modelo também pode ter seus próprios (por exemplo, pesos atribuídos aos modelos base).
- **XGBoost:** Possui uma variedade de hiperparâmetros que podem ser ajustados, incluindo o número de árvores, a taxa de aprendizagem, o tamanho do passo e o coeficiente de regularização.

## Complexidade de treinamento:

- **Árvores de decisão, KNN, Ridge, SVM:** Relativamente simples e rápidos de treinar.
- **Random Forest e Boosted Trees:** Podem ser mais lentos de treinar devido à construção sequencial de vários modelos.
- **Stacking:** Exige o treinamento de vários modelos individuais e depois o meta-modelo, podendo ser o mais lento.
- **XGBoost:** Pode ser mais lento de treinar do que outros métodos de ensemble, pois usa um algoritmo de otimização gradiente.

## Interpretabilidade:

- **Árvores de decisão, KNN, Ridge, SVM:** Fáceis de interpretar individualmente.
- **Random Forest e Boosted Trees:** Difíceis de interpretar devido à natureza combinada das previsões.
- **Stacking:** A interpretabilidade depende do meta-modelo escolhido.
- **XGBoost:** Difícil de interpretar, pois é um método de boosting.


# Diferença de treinamento Prática - Essemble 

## Árvore de Decisão:

- Utiliza DecisionTreeRegressor do scikit-learn.
- Hiperparâmetros ajustados: max_depth e min_samples_leaf.

## KNN (K-Nearest Neighbors):

- Utiliza KNeighborsRegressor do scikit-learn.
- Não especifica hiperparâmetros diretamente.

## Ridge (Regressão Linear com Regularização L2):

- Utiliza Ridge do scikit-learn.
- Pode ser treinado tanto com o target original quanto com o log do target.

## SVM (Support Vector Machine):

- Utiliza SVR do scikit-learn com um kernel linear.

## Floresta Aleatória:

- Utiliza RandomForestRegressor do scikit-learn.
- Hiperparâmetros ajustados: max_depth e min_samples_leaf.

## Boosted Trees (AdaBoost e Gradient Boosting):

- Para AdaBoost, utiliza AdaBoostRegressor com árvores de decisão.
- Para Gradient Boosting, utiliza GradientBoostingRegressor.

## Stacking:

- Utiliza VotingRegressor ou StackingRegressor do scikit-learn.
- Combina os modelos individuais (Gradient Boosting, AdaBoost, Ridge, SVM) com pesos iguais no primeiro caso e usando um meta-modelo Linear Regression no segundo.

## XGBoost:

- Pode ser integrado a um pipeline do scikit-learn ou treinado diretamente com a biblioteca XGBoost.
- Na segunda opção, utiliza conjuntos de treino/validação e critério de parada antecipada.

**Observação:** Todos os métodos utilizam Cross-Validation para avaliação.

# Extra

Com o objetivo de aprimorar o código, propõe-se um novo método de pré-processamento que incorpora uma descrição mais abrangente do conjunto de dados. Além disso, demonstra-se a abordagem específica para lidar com dados nulos em cada coluna do conjunto.

A visualização pode ser feita no seguinte documento <a href="./entendimento_e_preprocessamento.ipynb">entendimento_e_preprocessamento.ipynb</a>

# Sugestões de mudança no código 

São fornecidos exemplos de como aplicar os códigos com base nas sugestões apresentadas.

## Árvores de decisão

Aplicar técnica de seleção de atributos 'SelectKBest' para reduzir a dimensionalidade dos dados e utilizar algoritmo de busca por espaço de hiperparâmetros GridSearchCV. O objetivo é melhorar a performance e a interpretabilidade do modelo além de encontrar os melhores valores de hiperparâmetros.

```Python
k_best = [5, 10, 15, 20]

param_grid = {
    'decisiontreeregressor__max_depth': [10, 20, 30, 40],
    'decisiontreeregressor__min_samples_leaf': [5, 10, 15, 20]
}

selector = SelectKBest(score_func=f_regression)

model = DecisionTreeRegressor()
pipe = Pipeline([
    ('preproc', preproc),
    ('selector', selector),
    ('model', model)
])

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X, y_log)

selected_features = selector.get_support(indices=True)

best_model = grid_search.best_estimator_
score = cross_val_score(best_model, X, y_log, cv=5, scoring='neg_mean_squared_error')

print("Melhores parâmetros:", grid_search.best_params_)
print("Atributos selecionados:", selected_features)
print("RMSE médio:", abs(score.mean())**0.5)
```

## KNN

Utilizar algoritmo de vizinhança BallTree e técnica de seleção de atributos para reduzir a dimensionalidade dos dados. O objetivo é melhorar a performance e a interpretabilidade do modelo.

```Python
k_best = [5, 10, 15, 20]

param_grid = {'kneighborsregressor__n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]}

selector = SelectKBest(score_func=f_regression)

ball_tree = BallTree()
model = KNeighborsRegressor()
pipe_knn = make_pipeline(preproc, selector, ball_tree, model)

scores = cross_val_score(pipe_knn, X, y_log, cv=5, scoring=make_scorer(rmse, greater_is_better=False))

param_grid = {
    'kneighborsregressor__n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30],
    'selectkbest__k': k_best
}

search_knn = GridSearchCV(
    pipe_knn,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=make_scorer(rmse, greater_is_better=False)
)

search_knn.fit(X, y_log)

print("RMSE médio: \n", abs(scores.mean()))
print(f'Melhores parâmetros {search_knn.best_params_} \n')
print(f'Melhor RMSE {abs(search_knn.best_score_)}')
```

## Ridge

Utilizar algoritmo de otimização como L-BFGS e técnica de regularização para evitar o overfitting. O objetivo é melhorar a performance do modelo no conjunto de dados de teste.

```Python

def objective_function(params, X, y, alpha):
    beta = params.reshape(-1, 1)
    loss = np.mean((y - X @ beta)**2) + alpha * np.sum(beta**2)
    return loss

def gradient_function(params, X, y, alpha):
    beta = params.reshape(-1, 1)
    gradient = -2 * X.T @ (y - X @ beta) + 2 * alpha * beta
    return gradient.flatten()

class RidgeWithLBFGS(Ridge):
    def __init__(self, alpha=1.0, max_iter=100, tol=1e-3):
        super().__init__(alpha=alpha)
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        initial_params = self.coef_.flatten()

        result = minimize(
            fun=objective_function,
            jac=gradient_function,
            x0=initial_params,
            args=(X_std, y, self.alpha),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'disp': True, 'ftol': self.tol}
        )

        self.coef_ = result.x.reshape(-1, 1)
        return self

cachedir = mkdtemp()
preproc = StandardScaler()
model = RidgeWithLBFGS()

pipe_ridge = make_pipeline(preproc, model, memory=cachedir)
cross_val_score(pipe_ridge, X, y, cv=5, scoring=rmsle).mean()

param_grid = {'ridgewithlbfgs__alpha': np.linspace(0.5, 2, num=20)}

search_ridge = GridSearchCV(
    pipe_ridge,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring=rmse_neg
)

search_ridge.fit(X, y_log);

print(f'Melhores parâmetros {search_ridge.best_params_}')
print(f'Melhor Score {search_ridge.best_score_}')
```

## SVM

Utilizar kernel com Polynomial e técnica de regularização para evitar o overfitting. O objetivo é melhorar a performance do modelo no conjunto de dados de teste.

```Python

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_neg = make_scorer(rmse, greater_is_better=False)

model_poly = SVR(kernel='poly')
preproc = make_pipeline(StandardScaler(), PCA())
regressor = TransformedTargetRegressor(regressor=model_poly, transformer=StandardScaler())

pipe_svm_poly = make_pipeline(preproc, regressor)

if allow_grid_searching:
    param_grid_poly = {
        'transformedtargetregressor__regressor__C': [0.5, 0.7, 1, 2, 5, 10],
        'transformedtargetregressor__regressor__epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
        'transformedtargetregressor__regressor__degree': [2, 3, 4], 
    }

    search_svm_poly = GridSearchCV(
        pipe_svm_poly,
        param_grid=param_grid_poly,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring=rmse_neg
    )

    search_svm_poly.fit(X, y_log)

    svm_poly_best = search_svm_poly.best_estimator_

    print(f'Melhores parâmetros {search_svm_poly.best_params_}')
    print(f'Melhor Score {search_svm_poly.best_score_}')

```

## Random Forest

Utilizar um maior número de árvores para melhorar a precisão do modelo e um algoritmo de busca por espaço de hiperparâmetros como RandomizedSearchCV. O objetivo é encontrar os melhores valores para os hiperparâmetros do modelo.

```Python 

preproc = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), slice(None)), 
        ('imp', SimpleImputer(strategy='mean'), slice(None))
    ])

model = RandomForestRegressor(max_depth=50, min_samples_leaf=20)
pipe = make_pipeline(preproc, model)

param_dist = {
    'randomforestregressor__n_estimators': [50, 100, 200, 300],  
    'randomforestregressor__max_depth': [10, 20, 30, 40, 50], 
    'randomforestregressor__min_samples_leaf': [5, 10, 20, 30]  
}

random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

random_search.fit(X, y_log)

print("Melhores hiperparâmetros:", random_search.best_params_)
score = cross_val_score(random_search.best_estimator_, X, y_log, cv=5, scoring='neg_mean_squared_error')
print("RMSE médio:", (-score.mean())**0.5)
```

## Boosted Trees

Utilizar um número maior de árvores para melhorar a precisão do modelo, uma taxa de aprendizagem menor para evitar o overfitting e um algoritmo de busca por espaço de hiperparâmetros, GridSearchCV. O objetivo é ajudar a melhorar a performance do modelo no conjunto de dados de teste e encontrar os melhores valores para os hiperparâmetros.

```Python

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

cachedir = mkdtemp()

model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None), n_estimators=100, learning_rate=0.01)
pipe = make_pipeline(FunctionTransformer(func=preproc), model, memory=cachedir)
score = cross_val_score(pipe, X, y_log, cv=5, scoring=rmse)

print("Cross-validation RMSE:")
print(score.std())
print(score.mean())

grid = {
    'adaboostregressor__n_estimators': [50, 100, 150],
    'adaboostregressor__learning_rate': [0.01, 0.1, 0.5]
}

search_ab = GridSearchCV(pipe, grid, scoring='neg_mean_squared_error', cv=5, n_jobs=1, verbose=2)
search_ab.fit(X, y_log)

print(f'Melhores parâmetros {search_ab.best_params_}')
print(f'Melhor score 👉 {search_ab.best_score_}')

df_cv_results_ = pd.DataFrame(search_ab.cv_results_)
sns.scatterplot(x="param_adaboostregressor__n_estimators", y='mean_test_score', data=df_cv_results_)
sns.scatterplot(x="param_adaboostregressor__learning_rate", y='mean_test_score', data=df_cv_results_)
```

## Stacking

Utilizar um meta-modelo com uma regressão linear, usar um número maior de modelos base e hiperparâmetros de RandomizedSearchCV. O objetivo é melhorar a precisão do modelo e encontrar os melhores valores para os hiperparâmetros.

```Python

gboost = GradientBoostingRegressor(n_estimators=100)
ridge = Ridge()
svm = SVR(C=1, epsilon=0.05)
adaboost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None))

model = StackingRegressor(
    estimators=[("gboost", gboost), ("adaboost", adaboost), ("ridge", ridge), ("svm_rbf", svm)],
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

pipe_stacking = make_pipeline(preproc, model, memory=cachedir)

param_dist = {
    'stackingregressor__gboost__n_estimators': randint(50, 200),
    'stackingregressor__ridge__alpha': uniform(0.1, 10),
    'stackingregressor__svm_rbf__C': [0.1, 1, 10],
}

random_search = RandomizedSearchCV(
    estimator=pipe_stacking,
    param_distributions=param_dist,
    n_iter=10,  
    scoring='neg_mean_squared_error',  
    cv=5,
    n_jobs=-1
)

random_search.fit(X, y_log)
score = cross_val_score(random_search.best_estimator_, X, y_log, cv=5, scoring=rmse, n_jobs=-1)

print("Melhores Hiperparâmetros:", random_search.best_params_)
print("Pontuação Média:", score.mean())
print("Desvio Padrão:", score.std())
```

## XGBoost

Utilizar um número maior de árvores para melhorar a precisão do modelo, taxa de aprendizagem menor para evitar o overfitting e um algoritmo de early stopping para interromper o treinamento quando o desempenho no conjunto de dados de validação parar de melhorar. O objetivo é melhorar a performance do modelo no conjunto de dados de teste e evitar o overfitting.

```Python
X_train, X_eval, y_train_log, y_eval_log = train_test_split(X, y_log, random_state=42)

model_xgb = XGBRegressor(
    max_depth=10,
    n_estimators=3000,  
    learning_rate=0.01,  
    early_stopping_rounds=50, 
    eval_metric="rmse", 
    eval_set=[(X_eval, y_eval_log)], 
    verbose=False  
)

pipe_xgb = make_pipeline(YourPreprocessor(), model_xgb)  

param_grid = {
    "xgbregressor__max_depth": [8, 10, 12],
}

grid_search = GridSearchCV(pipe_xgb, param_grid, cv=5, scoring=make_scorer(rmse), n_jobs=-1)
grid_search.fit(X, y_log)

print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor pontuação RMSE:", grid_search.best_score_)
```

