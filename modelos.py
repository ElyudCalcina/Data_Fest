"""Un pipeline nuevo por ventana; CatBoost utiliza categorías nativas."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

def crear_modelo(nombre, x_train, parametros, semilla, hilos):
    categoricas = x_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numericas = [c for c in x_train.columns if c not in categoricas]
    p = dict(parametros)
    if nombre == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**p, cat_features=categoricas, random_seed=semilla, thread_count=hilos)
    if nombre == 'logistica':
        modelo = LogisticRegression(**p, random_state=semilla)
    elif nombre == 'hist_gradient_boosting':
        modelo = HistGradientBoostingClassifier(**p, random_state=semilla)
    elif nombre == 'random_forest':
        modelo = RandomForestClassifier(**p, random_state=semilla, n_jobs=hilos)
    elif nombre == 'xgboost':
        from xgboost import XGBClassifier
        modelo = XGBClassifier(**p, random_state=semilla, n_jobs=hilos)
    elif nombre == 'lightgbm':
        from lightgbm import LGBMClassifier
        modelo = LGBMClassifier(**p, random_state=semilla, n_jobs=hilos)
    else:
        raise ValueError(f'Modelo desconocido: {nombre}')
    transformacion = ColumnTransformer([
        ('categorias', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricas),
        ('numericas', StandardScaler() if nombre == 'logistica' else 'passthrough', numericas),
    ])
    return Pipeline([('preparacion', transformacion), ('modelo', modelo)])

def probabilidad_positiva(modelo, x):
    clases = list(modelo.classes_)
    return modelo.predict_proba(x)[:, clases.index(1)]
