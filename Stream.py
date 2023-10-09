# Importing necessary libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
import joblib

# Titre en grand, en vert et avec une police moderne
st.markdown("<h1 style='text-align: center; color: #4CAF50; font-family:Arial;'>Interface de Prédiction IA</h1>", unsafe_allow_html=True)

# Streamlit code for file upload
uploaded_file = st.file_uploader("Choisissez votre fichier de données", type=['xlsx'])

# Streamlit code for specifying date range
st.write('Paramétrez les dates de début et de fin des prédictions')
start_date = st.date_input('Date de début', value=pd.to_datetime('2023-01-01'))
end_date = st.date_input('Date de fin', value=pd.to_datetime('2023-12-31'))

# Function to convert and clean date columns
def convert_and_clean_date(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name], errors='coerce', utc=True).dt.tz_convert(None)
    df.dropna(subset=[col_name], inplace=True)

# If a file is uploaded and date range is selected
if uploaded_file is not None and start_date and end_date:
    df = pd.read_excel(uploaded_file)
    st.write('Dataframe chargé avec succès !')
    st.write(df.head())



    # Create future_dates based on user input
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    st.write('Plage de dates enregistrée!')
    #st.write(future_dates)

    # Chargement des fichiers
    excel_paths = {'vacances': 'fr-en-calendrier-scolaire.csv', 
                'feries': 'Jours_feries.xlsx', 'inflation': 'Inflation.xlsx'}
    df_vacances = pd.read_csv(excel_paths['vacances'], sep=';').query('Zones == "Corse"')
    df_feries = pd.read_excel(excel_paths['feries'])
    df_inflation = pd.read_excel(excel_paths['inflation'])

    # Conversion et nettoyage des colonnes de date
    for df_temp, col in zip([df_vacances, df_feries], ['Date de début', 'Date']):
        convert_and_clean_date(df_temp, col)

    # Normalisation des noms de colonnes et types de données
    df_inflation.rename(columns={'Année': 'Annee'}, inplace=True)
    df_inflation['Annee'] = df_inflation['Annee'].astype(int)

    # CREATION D UN DATAFRAME D ENTRAINEMENT

    # Grouper les données par 'Catégorie de Pneu' et créer des DataFrames
    dfs_by_category = {}
    tire_categories = df['Catégorie de Pneu'].unique()

    for category in tire_categories:
        sub_df = df[df['Catégorie de Pneu'] == category]
        sub_df = sub_df.groupby('Date de la réception')['Poids Net Collecté'].sum().reset_index()
        sub_df['Date de la réception'] = pd.to_datetime(sub_df['Date de la réception'])
        sub_df.set_index('Date de la réception', inplace=True)
        sub_df = sub_df.resample('D').asfreq().fillna(0).reset_index()
        
        # Ajouter des colonnes temporelles et d'autres métadonnées
        sub_df['Jours_Sem'] = sub_df['Date de la réception'].dt.dayofweek
        sub_df['Jours_Mois'] = sub_df['Date de la réception'].dt.day
        sub_df['Mois'] = sub_df['Date de la réception'].dt.month
        sub_df['Annee'] = sub_df['Date de la réception'].dt.year
        
        # Fusion avec df_inflation pour chaque année décalée
        for shift_year in range(6):
            df_temp = df_inflation.copy()
            column_name = f'Inflation_Annee_N-{shift_year}'
            
            # Ici, je crée une nouvelle colonne qui représente les années décalées
            sub_df[f'Annee_N-{shift_year}'] = sub_df['Annee'] - shift_year
            
            # Fusion avec le DataFrame principal
            sub_df = pd.merge(sub_df, df_temp, left_on=f'Annee_N-{shift_year}', right_on='Annee', how='left')
            
            # Renommer la colonne nouvellement ajoutée
            sub_df.rename(columns={'Taux inflation': column_name}, inplace=True)
            
            # Supprimer les colonnes temporaires et inutiles
            sub_df.drop(columns=[f'Annee_N-{shift_year}', 'Annee_y'], inplace=True)
            sub_df.rename(columns={'Annee_x': 'Annee'}, inplace=True)
            
        # Marquage des jours fériés et vacances
        sub_df['Vacances'] = 0
        sub_df['Feries'] = 0
        sub_df['Description_Vacances'] = 'NC'
        
        # Conversion des dates en datetime.date pour une comparaison valide
        for i, row in df_vacances.iterrows():
            date_debut = pd.Timestamp(row['Date de début']).date()
            date_fin = pd.Timestamp(row['Date de fin']).date()
            mask = (sub_df['Date de la réception'].dt.date >= date_debut) & (sub_df['Date de la réception'].dt.date <= date_fin)
            sub_df.loc[mask, 'Vacances'] = 1
            sub_df.loc[mask, 'Description_Vacances'] = row['Description']
            
        for i, row in df_feries.iterrows():
            mask = sub_df['Date de la réception'].dt.date == pd.Timestamp(row['Date']).date()

            sub_df.loc[mask, 'Feries'] = 1

        # Ordre des colonnes
        col_order = ['Date de la réception', 'Jours_Sem', 'Jours_Mois', 'Mois', 'Annee', 'Poids Net Collecté'] + \
                    [f'Inflation_Annee_N-{i}' for i in range(6)] + ['Vacances', 'Feries', 'Description_Vacances']
        sub_df = sub_df[col_order]
        
        # Ajouter à la collection de DataFrames
        dfs_by_category[category] = sub_df

    # Visualisation d'un exemple
    print(dfs_by_category[tire_categories[0]].head())

    # CREATION d'un DATAFRAME DE PREDICTION

    # 1. Plage de Dates
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame({'Date de la réception': future_dates})

    # 2. Colonnes Temporelles
    future_df['Jours_Sem'] = future_df['Date de la réception'].dt.dayofweek
    future_df['Jours_Mois'] = future_df['Date de la réception'].dt.day
    future_df['Mois'] = future_df['Date de la réception'].dt.month
    future_df['Annee'] = future_df['Date de la réception'].dt.year

    # 3. Données Externes
    # Inflation
    # Supprimer les anciennes colonnes d'inflation
    cols_to_drop = [col for col in future_df.columns if 'Inflation' in col]
    future_df.drop(columns=cols_to_drop, inplace=True)

    # Effectuer la jointure
    for i in range(0, 6):
        temp_df = df_inflation.copy()
        temp_df['Annee'] = temp_df['Annee'] + i
        temp_df.rename(columns={'Taux inflation': f'Inflation_Annee_N-{i}'}, inplace=True)
        future_df = pd.merge(future_df, temp_df, how='left', on='Annee', suffixes=('', f'_N-{i}'))


    # Vacances et Feries
    future_df['Vacances'] = 0
    future_df['Feries'] = 0
    future_df['Description_Vacances'] = 'NC'

    for i, row in df_vacances.iterrows():
        date_debut = pd.Timestamp(row['Date de début']).date()
        date_fin = pd.Timestamp(row['Date de fin']).date()
        mask = (future_df['Date de la réception'].dt.date >= date_debut) & (future_df['Date de la réception'].dt.date <= date_fin)
        future_df.loc[mask, 'Vacances'] = 1
        future_df.loc[mask, 'Description_Vacances'] = row['Description']

    for i, row in df_feries.iterrows():
        mask = future_df['Date de la réception'].dt.date == pd.Timestamp(row['Date']).date()
        future_df.loc[mask, 'Feries'] = 1

    # Ordre des colonnes
    col_order = ['Date de la réception', 'Jours_Sem', 'Jours_Mois', 'Mois', 'Annee'] + \
                [f'Inflation_Annee_N-{i}' for i in range(6)] + ['Vacances', 'Feries', 'Description_Vacances']

    future_df = future_df[col_order]
    future_df.head()

    # Dictionnaire pour stocker les modèles entraînés
    trained_models = {}




    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib


 

    # Supposons que dfs_by_category est votre dictionnaire contenant les DataFrames
    # dfs_by_category = {'A|E': df_AE, 'B': df_B, 'C': df_C, 'D': df_D}
    def escape_special_chars(text):
        return text.replace("|", "PIPE")


    def predict_future(df, category):
        
        # Préparer les variables
        feature_cols = ['Jours_Sem', 'Jours_Mois', 'Mois', 'Annee', 'Vacances', 'Feries'] + [f'Inflation_Annee_N-{i}' for i in range(6)]

        df_2023 = df

        # Filtrer les données pour les années spécifiée
        df_filtered = df[(df['Date de la réception'].dt.year >= start_year) & (df['Date de la réception'].dt.year <= end_year)]
        
        # Préparation des données pour l'entraînement
        X_train = df[feature_cols]
        y_train = df['Poids Net Collecté']
        X_future = future_df[feature_cols]
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_future_scaled = scaler.transform(X_future)
        
        # Entraînement
        max_iter_value = 3000 if category == 'A|E' else 500

        # Entraînement
        escaped_category = escape_special_chars(category)
        # Charger le modèle existant
        model = joblib.load(f'model_{escaped_category}.pkl')

        # Prédiction
        y_pred = model.predict(X_future_scaled)
        
        # Ajout des prédictions au DataFrame
        future_df['Predicted'] = y_pred
        
        # Préparer le DataFrame pour la fusion
        df_2023_reduced = df_2023[['Date de la réception', 'Poids Net Collecté']].rename(columns={'Poids Net Collecté': 'Real'})
        
        # Fusionner avec future_df pour obtenir les valeurs réelles
        merged_df = pd.merge(future_df, df_2023_reduced, on='Date de la réception', how='left')
        
        # Groupement par mois et somme des colonnes numériques
        grouped = merged_df.groupby(merged_df['Date de la réception'].dt.to_period("M"))[['Predicted', 'Real']].sum()

        # Filtrer les mois où les valeurs réelles sont différentes de 0
        filtered_grouped = grouped[grouped['Real'] != 0]

        # Calculer les sommes totales pour les valeurs prédites et réelles
        total_predicted = filtered_grouped['Predicted'].sum()
        total_real = filtered_grouped['Real'].sum()

        # Calculer le taux d'erreur
        error_rate = np.abs(filtered_grouped['Predicted'] - filtered_grouped['Real']) / filtered_grouped['Real']
        mean_error_rate = error_rate.mean() * 100  # en pourcentage
        
        # Calculer le taux d'erreur sur les sommes totales
        total_error_rate = np.abs(total_predicted - total_real) / total_real * 100  # en pourcentage

        # Diagramme à barres mensuels
        plt.figure(figsize=(10, 6))
        grouped.plot(kind='bar')
        plt.title(f'Catégorie de pneu : {category} (Mensuel)')
        plt.xlabel('Mois')
        plt.ylabel('Poids Net Total')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        
        # Groupement par mois et somme des colonnes numériques
        grouped = merged_df.groupby(merged_df['Date de la réception'].dt.to_period("M"))[['Predicted', 'Real']].sum()
        
        # Ajout de la colonne de date
        grouped['Mois_Annee'] = grouped.index.strftime('%Y-%m')

        # Réorganiser les colonnes
        grouped = grouped[['Mois_Annee', 'Predicted', 'Real']]

        # Renommer les colonnes en français
        grouped.rename(columns={'Mois_Annee': 'Mois et Année', 'Predicted': 'Prédit', 'Real': 'Réel'}, inplace=True)

        # Calcul et affichage du tableau récapitulatif dans Streamlit
        st.write(f"Tableau récapitulatif pour la catégorie {category}")
        html = grouped.to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

        # Calculer les sommes totales pour les valeurs prédites et réelles
        total_predicted = round(filtered_grouped['Predicted'].sum(), 2)
        total_real = round(filtered_grouped['Real'].sum(), 2)

        # Calculer le taux d'erreur sur les sommes totales
        total_error_rate = round(np.abs(total_predicted - total_real) / total_real * 100, 2)  # en pourcentage

        # Affichage des totaux dans Streamlit
        st.write(f"Total Prédit  : {total_predicted}, Total Réel: {total_real}")

        # Afficher le taux d'erreur sur les sommes totales dans Streamlit
        st.write(f"Taux d'erreur total : {total_error_rate}%")


    # Exécution pour chaque catégorie avec les années spécifiées
    start_year = 2018  # Spécifier l'année de début
    end_year = 2023  # Spécifier l'année de fin
    # Liste des 3 premières catégories que vous souhaitez prédire
    first_three_categories = list(dfs_by_category.keys())[:2] #le nb ajuste le nb de cat

    # Exécution pour chaque catégorie dans la liste des 3 premières
    for category in first_three_categories:
        predict_future(dfs_by_category[category], category)

