import pandas as pd

# Récupération des données CSV via DataFrame
selected_cols = [
  "code",
  "product_name",
  "quantity",
  "brands",
  "categories",
  "ingredients_text",
  "nutriscore_score",
  "nutriscore_grade",
  "energy-kj_100g",
  "energy-kcal_100g",
  "fat_100g",
  "saturated-fat_100g",
  "omega-3-fat_100g",
  "omega-6-fat_100g",
  "sugars_100g",
  "added-sugars_100g",
  "fiber_100g",
  "proteins_100g",
  "salt_100g",
  "fruits-vegetables-nuts-estimate-from-ingredients_100g"
    ]
cols_stat = [
  "nutriscore_score",
  "nutriscore_grade",
  "energy-kj_100g",
  "energy-kcal_100g",
  "fat_100g",
  "saturated-fat_100g",
  "omega-3-fat_100g",
  "omega-6-fat_100g",
  "sugars_100g",
  "added-sugars_100g",
  "fiber_100g",
  "proteins_100g",
  "salt_100g",
  "fruits-vegetables-nuts-estimate-from-ingredients_100g",
]
directory_path = "../static/"
file_name = "en.openfoodfacts.org.products.csv"
output_name = 'openfoodfact_clean.csv'
chunksize = 10000

chunks = pd.read_csv(directory_path+file_name, sep="\t", low_memory=False, header=0, chunksize=chunksize, on_bad_lines="skip")
list_df = []
for chunk in chunks:
  list_df.append(chunk[selected_cols])

# Enlever les lignes indésirables (sans nutriscore_score et nutriscore_grade)
list_df_notna = []
for df in list_df:
  if len(df) > 0:
    df_append = df[df[['nutriscore_score', 'nutriscore_grade']].notna().all(axis=1)][selected_cols]
    list_df_notna.append(df_append)
df_notna = pd.concat(list_df_notna, ignore_index=True)

# Enlever les lignes or nutriscore-grade
nutri_ok = ["a", "b", "c", "d", "e"]
df_notna = df_notna[df_notna["nutriscore_grade"].isin(nutri_ok)]

# Remplacement des valeurs pour les colonnes qui servira de calcul
df_notna[cols_stat] = df_notna[cols_stat].fillna(0)

# Exporter en CSV avec un séparateur de tabulation (\t)
df_notna.to_csv(directory_path+output_name, sep='\t', index=False)