from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Set random seed for reproducibility
np.random.seed(0)

# SPARQL endpoint for DBpedia
sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# Function to fetch triples with pagination
def fetch_triples(limit=10000, max_results=5000000):
    triples = []
    offset = 0
    while offset < max_results:
        query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?movie ?title ?director ?releaseDate ?genre ?actor
            WHERE {{
                ?movie a dbo:Film .
                ?movie rdfs:label ?title .
                FILTER (lang(?title) = 'en')
                OPTIONAL {{ ?movie dbo:director ?director . }}
                OPTIONAL {{ ?movie dbo:releaseDate ?releaseDate . }}
                OPTIONAL {{ ?movie dbo:genre ?genre . }}
                OPTIONAL {{ ?movie dbo:starring ?actor . }}
            }}
            LIMIT {limit} OFFSET {offset}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            new_triples = [
                (
                    result["movie"]["value"],
                    result["title"]["value"],
                    result.get("director", {}).get("value", ""),
                    result.get("releaseDate", {}).get("value", ""),
                    result.get("genre", {}).get("value", ""),
                    result.get("actor", {}).get("value", "")
                )
                for result in results["results"]["bindings"]
            ]
            if not new_triples:
                break
            triples.extend(new_triples)
            print(f"Retrieved {len(new_triples)} triples at offset {offset}. Total: {len(triples)}")
            offset += limit
            if len(new_triples) < limit:
                break
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break
    return triples

# Fetch triples
triples = fetch_triples(limit=10000, max_results=5000000)
print(f"Total retrieved triples: {len(triples)}")
print("First 5 triples:", triples[:5])

# Convert to dictionary
data = {}
for movie, title, director, release_date, genre, actor in triples:
    if movie not in data:
        data[movie] = {
            "title": title,
            "director": director,
            "release_date": release_date,
            "genre": genre,
            "actor": actor
        }
    else:
        if genre and genre not in data[movie]["genre"]:
            data[movie]["genre"] += f";{genre}" if data[movie]["genre"] else genre
        if actor and actor not in data[movie]["actor"]:
            data[movie]["actor"] += f";{actor}" if data[movie]["actor"] else actor

print(f"Data dictionary created with {len(data)} unique movies.")
print("Sample data from the dictionary:")
for key, value in list(data.items())[:5]:
    print(f"{key}: {value}")

# Convert to DataFrame
df = pd.DataFrame.from_dict(data, orient='index')
df = df.fillna('')
print("DataFrame created from DBpedia movie data:")
print(df.head())

# Simplify column names and values
df.index = df.index.str.split("/").str[-1]
for col in df.columns:
    df[col] = df[col].str.split("/").str[-1]
    df[col] = df[col].str.split("#").str[-1]

# Extract year from release_date
df['release_date'] = df['release_date'].str.extract(r'(\d{4})')

# Print missing values
print("\nMissing values before filtering:")
print(df[['director', 'release_date', 'genre', 'actor']].isna().sum())

# Normalize genre labels
df['genre'] = df['genre'].str.lower().replace({
    'comedy_film': 'comedy',
    'drama_(film_and_television)': 'drama',
    'drama_film': 'drama',
    'romantic_comedy': 'comedy',
    'comedy-drama': 'comedy',
    'dramedy': 'comedy',
    'comedy': 'comedy',
    'drama': 'drama',
    'romance_film': 'drama',
    'romantic_drama': 'drama',
    'romantic_comedy_film': 'comedy'
}, regex=True)

# Debug: Print unique genres
print("\nUnique genres before filtering:")
print(df['genre'].str.split(';').explode().value_counts())

# Filter for comedy or drama
def filter_genres(genre_string):
    if not genre_string:
        return False
    genres = genre_string.split(';')
    for genre in genres:
        if 'comedy' in genre or 'drama' in genre:
            return True
    return False

df = df[df['genre'].apply(filter_genres)]

# Assign labels
df['label'] = df['genre'].apply(lambda x: 'comedy' if 'comedy' in x else 'drama')

# Fill missing values
df['director'] = df['director'].replace('', 'Unknown')
df['release_date'] = df['release_date'].fillna('Unknown')
df['actor'] = df['actor'].replace('', 'Unknown')

# Drop rows with empty genres
df = df[df['genre'] != '']

# Drop title and genre columns
df = df.drop(columns=['title', 'genre'])

# Print label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())

print("\nSimplified and filtered DataFrame:")
print(df.head())
print(f"Number of movies after filtering: {len(df)}")

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'label':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Prepare features and labels
X = df.drop(columns=['label'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = GradientBoostingClassifier(n_estimators=100, random_state=0, max_depth=3)
model.fit(X_train, y_train)

# Evaluate model
accuracy_train = model.score(X_train, y_train)
accuracy_test = model.score(X_test, y_test)
precision = precision_score(y_test, model.predict(X_test), average='binary', pos_label='comedy', zero_division=0)
recall = recall_score(y_test, model.predict(X_test), average='binary', pos_label='comedy', zero_division=0)
f1 = f1_score(y_test, model.predict(X_test), average='binary', pos_label='comedy', zero_division=0)

print(f"\nModel accuracy (Training): {accuracy_train:.4f}")
print(f"Model accuracy (Test): {accuracy_test:.4f}")
print(f"Precision (Test): {precision:.4f}")
print(f"Recall (Test): {recall:.4f}")
print(f"F1-Score (Test): {f1:.4f}")

# Gini Based Feature importance
importances_mean = model.feature_importances_
importances_std = np.std([tree.feature_importances_ for tree in model.estimators_[:, 0]], axis=0)

# Sort features
top_n = len(X.columns)
indices = np.argsort(importances_mean)[::-1][:top_n]
top_importances_mean = importances_mean[indices]
top_importances_std = importances_std[indices]
top_feature_names = X.columns[indices]

# Display feature importance
print("\nGini Feature Importance:")
feature_importance_df = pd.DataFrame({
    'top_importances_mean': top_importances_mean,
    'top_importances_std': top_importances_std
}, index=top_feature_names)
print(feature_importance_df)

# Permutation feature importance
from sklearn.inspection import permutation_importance

r = permutation_importance(model, X_test, y_test,
    n_repeats=5,
    random_state=0)

importances_mean = r.importances_mean
importances_std = r.importances_std

# Sort features for permutation importance
top_n = len(X.columns)
indices = np.argsort(importances_mean)[::-1][:top_n]
top_importances_mean = importances_mean[indices]
top_importances_std = importances_std[indices]


# Display permutation feature importance
print("\nPermutation Feature Importance:")
perm_feature_importance_df = pd.DataFrame({
    'top_importances_mean': top_importances_mean,
    'top_importances_std': top_importances_std
}, index=top_feature_names)
print(perm_feature_importance_df)
