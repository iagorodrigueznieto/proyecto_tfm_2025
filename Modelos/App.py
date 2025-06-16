import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from functools import partial
from multiprocessing import Pool
from joblib import Parallel, delayed

# Configuración de la página
st.set_page_config(
    page_title="⚽ Predictor de Fútbol",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #2a5298;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; text-align: center; margin: 0;">
        ⚽ Predictor de Fútbol
    </h1>
    <p style="color: white; text-align: center; margin: 0; font-size: 1.2em;">
        Predice el potencial de jugadores jóvenes y resultados de partidos usando Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def cargar_datos(data_path):
    try:
        datasets = {
            'appearances': pd.read_csv(data_path + '/appearances.csv'),
            'clubs': pd.read_csv(data_path + '/clubs.csv'),
            'competitions': pd.read_csv(data_path + '/competitions.csv'),
            'game_events': pd.read_csv(data_path + '/game_events.csv'),
            'game_lineups': pd.read_csv(data_path + '/game_lineups.csv'),
            'games': pd.read_csv(data_path + '/games.csv'),
            'player_valuations': pd.read_csv(data_path + '/player_valuations.csv'),
            'players': pd.read_csv(data_path + '/players.csv'),
            'transfers': pd.read_csv(data_path + '/transfers.csv')
        }
        return datasets
    except Exception as e:
        st.error(f"Error cargando los datos: {str(e)}")
        return None


# Función para preparar datos de jugadores
@st.cache_data
def preparar_datos(datasets, max_age, min_matches):
    """Prepara datos para entrenamiento de predicción de talento"""
    df_players = datasets['players']
    df_appearances = datasets['appearances']
    df_player_valuations = datasets['player_valuations']
    df_transfers = datasets['transfers']

    current_date = pd.to_datetime('2025-06-12 13:39:00').tz_localize(None)
    df_players['date_of_birth'] = pd.to_datetime(df_players['date_of_birth'], errors='coerce').dt.tz_localize(None)
    df_players['age'] = (current_date - df_players['date_of_birth']).dt.days / 365.25

    if 'name' not in df_players.columns:
        df_players['name'] = df_players['first_name'] + ' ' + df_players['last_name']

    df_young = df_players[df_players['age'] <= max_age].copy()
    matches_played = df_appearances.groupby('player_id')['game_id'].nunique().reset_index(name='matches_played')
    df_young = df_young.merge(matches_played, on='player_id', how='left').fillna({'matches_played': 0})
    df_young = df_young[df_young['matches_played'] >= min_matches].copy()

    recent_appearances = df_appearances[pd.to_datetime(df_appearances['date'], errors='coerce') >= '2024-01-01']
    if recent_appearances.empty:
        recent_appearances = df_appearances

    df_agg = recent_appearances.groupby('player_id').agg({
        'goals': 'mean',
        'assists': 'mean',
        'minutes_played': 'mean',
        'yellow_cards': 'mean',
        'red_cards': 'mean'
    }).reset_index()

    transfer_fees = df_transfers.groupby('player_id')['transfer_fee'].max().reset_index(name='max_transfer_fee')
    df_agg = df_agg.merge(transfer_fees, on='player_id', how='left').fillna({'max_transfer_fee': 0})
    df_merged = df_young.merge(df_agg, on='player_id', how='left')

    df_player_valuations['date'] = pd.to_datetime(df_player_valuations['date'], errors='coerce').dt.tz_localize(None)
    future_date = current_date + pd.Timedelta(days=3 * 365)
    df_future_vals = df_player_valuations[df_player_valuations['date'] <= future_date]
    df_future_vals = df_future_vals.sort_values('date').groupby(
        'player_id').last().reset_index() if df_future_vals.empty else df_future_vals.sort_values('date').groupby(
        'player_id').last().reset_index()

    mean_market_value = df_future_vals['market_value_in_eur'].mean()
    df_future_vals['target'] = (df_future_vals['market_value_in_eur'] > mean_market_value).astype(int)
    df_final = df_merged.merge(df_future_vals[['player_id', 'target', 'market_value_in_eur']], on='player_id',
                               how='inner')

    for col in ['goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'max_transfer_fee']:
        df_final[col] = df_final[col].fillna(0)

    return df_final


# Función para preparar datos para modelos de talento
def prepare_model_data(df_final):
    """Prepara los datos procesados para entrenamiento"""
    le = LabelEncoder()
    df_final['position_encoded'] = le.fit_transform(
        df_final['position'].fillna('Unknown')) if 'position' in df_final.columns else 0

    features = ['age', 'goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'position_encoded',
                'max_transfer_fee']
    for feature in features:
        if feature not in df_final.columns:
            df_final[feature] = 0

    X = df_final[features].fillna(0)
    y = df_final['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, df_final, le


# Función para crear red neuronal
def create_neural_network(input_shape, layers_config, learning_rate=0.001, regularization=0.01):
    """Crea una red neuronal personalizable"""
    model = Sequential()
    model.add(Dense(layers_config[0]['neurons'], activation=layers_config[0]['activation'], input_shape=(input_shape,),
                    kernel_regularizer=l2(regularization)))
    if layers_config[0]['batch_norm']:
        model.add(BatchNormalization())
    if layers_config[0]['dropout'] > 0:
        model.add(Dropout(layers_config[0]['dropout']))

    for layer_config in layers_config[1:]:
        model.add(Dense(layer_config['neurons'], activation=layer_config['activation'],
                        kernel_regularizer=l2(regularization)))
        if layer_config['batch_norm']:
            model.add(BatchNormalization())
        if layer_config['dropout'] > 0:
            model.add(Dropout(layer_config['dropout']))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Función para graficar comparación de métricas
def plot_metrics_comparison(models_results):
    """Crea gráfico comparativo de métricas"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig = go.Figure()

    for model_name, results in models_results.items():
        fig.add_trace(go.Scatterpolar(
            r=[results[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparación de Métricas por Modelo"
    )
    return fig


# Función para graficar curvas ROC
def plot_roc_curves(models_results, y_test):
    """Crea gráfico de curvas ROC"""
    fig = go.Figure()

    for model_name, results in models_results.items():
        if 'y_prob' in results:
            fpr, tpr, _ = roc_curve(y_test, results['y_prob'])
            auc_score = results['ROC-AUC']
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {auc_score:.3f})',
                                     line=dict(width=2)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='Curvas ROC - Comparación de Modelos', xaxis_title='Tasa de Falsos Positivos',
                      yaxis_title='Tasa de Verdaderos Positivos', width=700, height=500)
    return fig


# Funciones para predicción de partidos
def get_match_result(df):
    """Vectorized match result calculation"""
    conditions = [
        df['home_club_goals'] > df['away_club_goals'],
        df['home_club_goals'] < df['away_club_goals']
    ]
    choices = ['home_win', 'away_win']
    return np.select(conditions, choices, default='draw')


def get_team_stats_vectorized(df_games, team_id, game_date, n_matches=5):
    """Vectorized team stats calculation"""
    past_games = df_games[(df_games['date'] < game_date) &
                          ((df_games['home_club_id'] == team_id) |
                           (df_games['away_club_id'] == team_id))].tail(n_matches)
    if past_games.empty:
        league_avg = df_games[['home_club_goals', 'away_club_goals']].mean()
        return pd.Series({
            'avg_goals_scored': league_avg['home_club_goals'],
            'avg_goals_conceded': league_avg['away_club_goals']
        })

    is_home = past_games['home_club_id'] == team_id
    goals_scored = np.where(is_home, past_games['home_club_goals'], past_games['away_club_goals']).sum()
    goals_conceded = np.where(is_home, past_games['away_club_goals'], past_games['home_club_goals']).sum()

    return pd.Series({
        'avg_goals_scored': goals_scored / len(past_games),
        'avg_goals_conceded': goals_conceded / len(past_games)
    })


def parallel_get_team_stats(df_games, team_ids, game_dates, n_matches=5):
    """Parallelized team stats calculation"""
    df_games = df_games.sort_values('date')
    team_date_pairs = [(team_id, game_date) for team_id, game_date in zip(team_ids, game_dates)]
    team_date_pairs = list(set(team_date_pairs))

    with Pool() as pool:
        results = pool.starmap(
            partial(get_team_stats_vectorized, df_games, n_matches=n_matches),
            team_date_pairs
        )

    stats_df = pd.DataFrame(results, index=[f"{tid}_{gd}" for tid, gd in team_date_pairs])
    return stats_df


def get_team_player_stats_vectorized(df_appearances, game_ids, club_ids):
    """Vectorized player stats calculation"""
    results = []
    for game_id, club_id in zip(game_ids, club_ids):
        team_appearances = df_appearances[(df_appearances['game_id'] == game_id) &
                                          (df_appearances['player_club_id'] == club_id)]
        stats = {
            'team_avg_goals': team_appearances['goals'].mean() if not team_appearances.empty else 0,
            'team_avg_assists': team_appearances['assists'].mean() if not team_appearances.empty else 0
        }
        results.append(stats)
    return pd.DataFrame(results)


def prepare_match_data(home_club_id, away_club_id, game_id, date, df_games, df_clubs, df_appearances):
    # Create a dataframe with the match data
    match_data = pd.DataFrame({
        'game_id': [game_id],
        'home_club_id': [home_club_id],
        'away_club_id': [away_club_id],
        'date': [pd.to_datetime(date)]
    })

    # Merge club features from df_clubs (including club names for display)
    df_clubs_features = df_clubs[
        ['club_id', 'name', 'squad_size', 'average_age', 'foreigners_percentage',
         'national_team_players', 'stadium_seats']]

    match_data = match_data.merge(df_clubs_features, left_on='home_club_id', right_on='club_id',
                                  suffixes=('', '_home'), how='left').drop(columns=['club_id'],
                                                                           errors='ignore')
    match_data = match_data.merge(df_clubs_features, left_on='away_club_id', right_on='club_id',
                                  suffixes=('_home', '_away'), how='left').drop(columns=['club_id'],
                                                                                errors='ignore')

    # Calculate stats for the last 5 matches for each team
    def get_team_stats(df_games, team_id, game_date, n_matches=5):
        past_games = df_games[(df_games['date'] < game_date) &
                              ((df_games['home_club_id'] == team_id) |
                               (df_games['away_club_id'] == team_id))].tail(n_matches)
        if past_games.empty:
            league_avg = df_games[['home_club_goals', 'away_club_goals']].mean()
            return pd.Series({
                'avg_goals_scored': league_avg['home_club_goals'],
                'avg_goals_conceded': league_avg['away_club_goals']
            })

        goals_scored = 0
        goals_conceded = 0
        for _, row in past_games.iterrows():
            if row['home_club_id'] == team_id:
                goals_scored += row['home_club_goals']
                goals_conceded += row['away_club_goals']
            else:
                goals_scored += row['away_club_goals']
                goals_conceded += row['home_club_goals']

        return pd.Series({
            'avg_goals_scored': goals_scored / len(past_games),
            'avg_goals_conceded': goals_conceded / len(past_games)
        })

    home_stats = match_data.apply(
        lambda row: get_team_stats(df_games, row['home_club_id'], row['date']), axis=1)
    away_stats = match_data.apply(
        lambda row: get_team_stats(df_games, row['away_club_id'], row['date']), axis=1)
    match_data[['home_avg_goals_scored', 'home_avg_goals_conceded']] = home_stats
    match_data[['away_avg_goals_scored', 'away_avg_goals_conceded']] = away_stats

    # Calculate player stats (average goals and assists per team)
    team_stats = df_appearances[df_appearances['player_club_id'].isin([home_club_id, away_club_id])]
    team_stats = team_stats.groupby('player_club_id')[['goals', 'assists']].mean().reset_index()
    team_stats = team_stats.rename(columns={'goals': 'team_avg_goals', 'assists': 'team_avg_assists'})

    match_data = match_data.merge(
        team_stats[team_stats['player_club_id'] == home_club_id],
        left_on='home_club_id',
        right_on='player_club_id',
        how='left'
    ).rename(columns={
        'team_avg_goals': 'home_team_avg_goals',
        'team_avg_assists': 'home_team_avg_assists'
    }).drop(columns=['player_club_id'], errors='ignore')

    match_data = match_data.merge(
        team_stats[team_stats['player_club_id'] == away_club_id],
        left_on='away_club_id',
        right_on='player_club_id',
        how='left'
    ).rename(columns={
        'team_avg_goals': 'away_team_avg_goals',
        'team_avg_assists': 'away_team_avg_assists'
    }).drop(columns=['player_club_id'], errors='ignore')

    # Ensure all features are present
    features = [
        'squad_size_home', 'average_age_home', 'foreigners_percentage_home',
        'national_team_players_home',
        'stadium_seats_home', 'squad_size_away', 'average_age_away', 'foreigners_percentage_away',
        'national_team_players_away', 'stadium_seats_away', 'home_avg_goals_scored',
        'home_avg_goals_conceded', 'away_avg_goals_scored', 'away_avg_goals_conceded',
        'home_team_avg_goals', 'home_team_avg_assists', 'away_team_avg_goals', 'away_team_avg_assists'
    ]

    for feature in features:
        if feature not in match_data.columns:
            match_data[feature] = 0

    match_data[features] = match_data[features].fillna(0)

    # Keep club names for display
    match_data['home_club_name'] = match_data['name_home']
    match_data['away_club_name'] = match_data['name_away']

    return match_data[features + ['home_club_name', 'away_club_name']]


def predict_match_result(model, le, home_club_id, away_club_id, game_id, date, df_games, df_clubs,
                         df_appearances):
    # Prepare match data
    match_data = prepare_match_data(home_club_id, away_club_id, game_id, date, df_games, df_clubs,
                                    df_appearances)

    # Extract club names
    home_club_name = match_data['home_club_name'].iloc[
        0] if 'home_club_name' in match_data.columns else 'Unknown Home Club'
    away_club_name = match_data['away_club_name'].iloc[
        0] if 'away_club_name' in match_data.columns else 'Unknown Away Club'

    # Select features for prediction
    features = [
        'squad_size_home', 'average_age_home', 'foreigners_percentage_home',
        'national_team_players_home',
        'stadium_seats_home', 'squad_size_away', 'average_age_away', 'foreigners_percentage_away',
        'national_team_players_away', 'stadium_seats_away', 'home_avg_goals_scored',
        'home_avg_goals_conceded', 'away_avg_goals_scored', 'away_avg_goals_conceded',
        'home_team_avg_goals', 'home_team_avg_assists', 'away_team_avg_goals', 'away_team_avg_assists'
    ]
    X_new = match_data[features]

    # Make prediction
    prediction = model.predict(X_new)
    prediction_proba = model.predict_proba(X_new)
    predicted_result = le.inverse_transform(prediction)[0]
    proba_dict = dict(zip(le.classes_, prediction_proba[0]))

    # Format the result
    result_text = f"Partido: {home_club_name} vs {away_club_name}\n"
    result_text += f"Resultado predicho: {predicted_result}\n"
    result_text += "Probabilidades:\n"
    for outcome, prob in proba_dict.items():
        result_text += f"  {outcome}: {prob * 100:.2f}%\n"

    return result_text


# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")
st.sidebar.subheader("📁 Datos")
default_path = "./dataset"
data_path = st.sidebar.text_input("Ruta de los datasets:", value=default_path)

# Crear pestañas
tab1, tab2 = st.tabs(["🌟 Predictor de Talento Joven", "⚽ Predictor de Resultados de Partidos"])

# Pestaña 1: Predictor de Talento Joven
with tab1:
    st.sidebar.subheader("Filtros de Talento")
    edad_maxima = st.sidebar.slider("Edad máxima", 18, 25, 22, key="max_age_talent")
    minimo_de_partidos = st.sidebar.slider("Partidos mínimos", 10, 50, 30, key="min_matches_talent")
    use_smote = st.sidebar.checkbox("Usar SMOTE", True, key="smote_talent")

    st.sidebar.subheader("Modelos de Talento")
    models_to_train = st.sidebar.multiselect(
        "Selecciona modelos:",
        ["Random Forest", "Logistic Regression", "SVM", "Naive Bayes", "Neural Network"],
        default=["Random Forest", "Neural Network"],
        key="models_talent"
    )

    if "Random Forest" in models_to_train:
        st.sidebar.subheader("Random Forest")
        rf_n_estimators = st.sidebar.slider("Número de árboles", 50, 500, 100, key="rf_n_estimators_talent")
        rf_max_depth = st.sidebar.slider("Profundidad máxima", 5, 20, 10, key="rf_max_depth_talent")

    if "Neural Network" in models_to_train:
        st.sidebar.subheader("Red Neuronal")
        nn_layers = st.sidebar.slider("Capas ocultas", 1, 5, 3, key="nn_layers_talent")
        layers_config = []
        for i in range(nn_layers):
            st.sidebar.write(f"**Capa {i + 1}:**")
            neurons = st.sidebar.slider(f"Neuronas", 16, 256, 64, key=f"neurons_{i}_talent")
            activation = st.sidebar.selectbox(f"Activación", ["relu", "tanh", "sigmoid"], key=f"activation_{i}_talent")
            dropout = st.sidebar.slider(f"Dropout", 0.0, 0.8, 0.3, key=f"dropout_{i}_talent")
            batch_norm = st.sidebar.checkbox(f"Batch Normalization", True, key=f"bn_{i}_talent")
            layers_config.append(
                {'neurons': neurons, 'activation': activation, 'dropout': dropout, 'batch_norm': batch_norm})
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f", key="lr_talent")
        epochs = st.sidebar.slider("Épocas", 10, 200, 50, key="epochs_talent")
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, key="batch_size_talent")

    train_talent_button = st.sidebar.button("🚀 Entrenar Modelos de Talento", key="train_talent")

    if train_talent_button:
        try:
            with st.spinner("Cargando datasets..."):
                datasets = cargar_datos(data_path)
                if datasets is None:
                    st.error("❌ No se pudieron cargar los datasets.")
                    st.stop()
                st.success("✅ Datasets cargados correctamente")

                with st.expander("📊 Información de Datasets"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Jugadores", f"{len(datasets['players']):,}")
                        st.metric("Apariciones", f"{len(datasets['appearances']):,}")
                        st.metric("Clubs", f"{len(datasets['clubs']):,}")
                    with col2:
                        st.metric("Juegos", f"{len(datasets['games']):,}")
                        st.metric("Eventos", f"{len(datasets['game_events']):,}")
                        st.metric("Competiciones", f"{len(datasets['competitions']):,}")
                    with col3:
                        st.metric("Valoraciones", f"{len(datasets['player_valuations']):,}")
                        st.metric("Transferencias", f"{len(datasets['transfers']):,}")
                        st.metric("Alineaciones", f"{len(datasets['game_lineups']):,}")

            with st.spinner("Preparando datos..."):
                df_processed = preparar_datos(datasets, edad_maxima, minimo_de_partidos)
                if len(df_processed) == 0:
                    st.error("❌ No se encontraron jugadores.")
                    st.stop()

                X, y, scaler, df_final, le = prepare_model_data(df_processed)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Jugadores Filtrados", len(df_final))
                with col2:
                    st.metric("Características", X.shape[1])
                with col3:
                    st.metric("Alto Potencial", int(y.sum()))
                with col4:
                    st.metric("Balance de Clases", f"{y.mean():.1%}")

                with st.expander("📈 Estadísticas"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Distribución por Edad")
                        age_hist = px.histogram(df_final, x='age', nbins=20, title="Distribución de Edades")
                        st.plotly_chart(age_hist, use_container_width=True)
                    with col2:
                        st.subheader("Top 10 Jugadores")
                        if 'market_value_in_eur' in df_final.columns:
                            top_players = df_final.nlargest(10, 'market_value_in_eur')[
                                ['name', 'age', 'market_value_in_eur', 'goals', 'assists']]
                            top_players['market_value_in_eur'] = top_players['market_value_in_eur'].apply(
                                lambda x: f"€{x:,.0f}")
                            st.dataframe(top_players, hide_index=True)

            if use_smote and len(np.unique(y)) > 1:
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2,
                                                                    random_state=42)
                st.info("✓ Datos balanceados con SMOTE")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                if len(np.unique(y)) <= 1:
                    st.warning("⚠️ Solo una clase encontrada.")

        except Exception as e:
            st.error(f"❌ Error procesando datos: {str(e)}")
            st.stop()

        models_results = {}
        trained_models = {}
        progress_bar = st.progress(0)
        total_models = len(models_to_train)

        for idx, model_name in enumerate(models_to_train):
            with st.spinner(f"Entrenando {model_name}..."):
                if model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                                   random_state=42)
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif model_name == "SVM":
                    model = SVC(probability=True, random_state=42)
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif model_name == "Naive Bayes":
                    model = GaussianNB()
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif model_name == "Neural Network":
                    model = create_neural_network(X_train.shape[1], layers_config, learning_rate)
                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                                        verbose=0)
                    y_prob = model.predict(X_test, verbose=0).flatten()
                    y_pred = (y_prob > 0.5).astype(int)

                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'ROC-AUC': roc_auc_score(y_test, y_prob),
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
                models_results[model_name] = metrics
            progress_bar.progress((idx + 1) / total_models)

        st.subheader("📊 Resultados")
        metrics_df = pd.DataFrame(models_results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        st.subheader("Métricas de Rendimiento")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen').format("{:.3f}"))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Comparación de Métricas")
            radar_fig = plot_metrics_comparison(models_results)
            st.plotly_chart(radar_fig, use_container_width=True)
        with col2:
            st.subheader("Curvas ROC")
            roc_fig = plot_roc_curves(models_results, y_test)
            st.plotly_chart(roc_fig, use_container_width=True)

        st.subheader("🌟 Jugadores con Mayor Potencial")
        features = ['age', 'goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'position_encoded',
                    'max_transfer_fee']
        df_final['avg_potential'] = 0.0

        for idx, row in df_final.iterrows():
            player_features = [row.get(feature, 0) if not pd.isna(row.get(feature)) else 0 for feature in features]
            player_scaled = scaler.transform([player_features])
            avg_prob = 0
            model_count = 0

            for model_name in models_to_train:
                try:
                    if model_name == "Neural Network":
                        prob = model.predict(player_scaled, verbose=0)[0][0]
                    elif model_name in trained_models:
                        prob = trained_models[model_name].predict_proba(player_scaled)[0][1]
                    avg_prob += prob
                    model_count += 1
                except:
                    continue

            if model_count > 0:
                df_final.at[idx, 'avg_potential'] = avg_prob / model_count

        top_players = df_final.nlargest(5, 'avg_potential').to_dict('records')
        famous_players = ['Lamine Yamal', 'Florian Wirtz', 'Jamal Musiala', 'Pedri', 'Gavi', 'Bellingham', 'Haaland']
        famous_players_found = df_final[df_final['name'].str.contains('|'.join(famous_players), case=False, na=False)]
        if not famous_players_found.empty:
            for famous_player in famous_players_found.to_dict('records'):
                if famous_player not in top_players and famous_player['avg_potential'] > 0.5:
                    top_players.append(famous_player)
                    if len(top_players) > 5:
                        top_players = top_players[:5]

        if top_players:
            for player in top_players[:5]:
                player_name = player.get('name', 'Jugador Desconocido')
                player_age = player.get('age', 0)
                market_value = player.get('market_value_in_eur', 0)

                with st.expander(f"⚽ {player_name} (Edad: {player_age:.1f}, Valor: €{market_value:,.0f})"):
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                        st.metric("Goles (promedio)", f"{player.get('goals', 0):.1f}")
                        st.metric("Asistencias (promedio)", f"{player.get('assists', 0):.1f}")
                    with col_stats2:
                        st.metric("Minutos jugados (promedio)", f"{player.get('minutes_played', 0):.0f}")
                        st.metric("Partidos jugados", f"{player.get('matches_played', 0):.0f}")

                    st.subheader("Predicciones de Potencial:")
                    cols = st.columns(len(models_to_train))
                    for idx, model_name in enumerate(models_to_train):
                        with cols[idx]:
                            player_features = [player.get(feature, 0) if not pd.isna(player.get(feature)) else 0 for
                                               feature in features]
                            player_scaled = scaler.transform([player_features])
                            try:
                                if model_name == "Neural Network":
                                    prob = model.predict(player_scaled, verbose=0)[0][0]
                                else:
                                    prob = trained_models[model_name].predict_proba(player_scaled)[0][1]
                                potential = "Alto" if prob > 0.5 else "Medio" if prob > 0.3 else "Bajo"
                                color = "green" if prob > 0.7 else "orange" if prob > 0.4 else "red"
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{model_name}</h4>
                                    <p style="color: {color}; font-size: 1.2em; font-weight: bold;">
                                        {potential} ({prob:.1%})
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

        else:
            st.info("No se encontraron jugadores destacados.")

        if "Neural Network" in models_to_train:
            st.subheader("📈 Historial de Entrenamiento - Red Neuronal")
            fig_history = make_subplots(rows=1, cols=2, subplot_titles=('Pérdida', 'Precisión'))
            fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Entrenamiento', line=dict(color='blue')),
                                  row=1, col=1)
            fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validación', line=dict(color='red')),
                                  row=1, col=1)
            fig_history.add_trace(
                go.Scatter(y=history.history['accuracy'], name='Entrenamiento', line=dict(color='blue'),
                           showlegend=False), row=1, col=2)
            fig_history.add_trace(
                go.Scatter(y=history.history['val_accuracy'], name='Validación', line=dict(color='red'),
                           showlegend=False), row=1, col=2)
            fig_history.update_layout(height=400, title_text="Historial de Entrenamiento")
            st.plotly_chart(fig_history, use_container_width=True)

    else:
        st.markdown("""
        ## 🎯 Predictor de Talento Joven
        Predice el potencial de jugadores jóvenes usando Machine Learning.

        ### 🚀 Para comenzar:
        1. Configura la ruta de datos
        2. Ajusta filtros de edad y partidos
        3. Selecciona modelos
        4. Personaliza la red neuronal
        5. Haz clic en "Entrenar Modelos de Talento"

        ### 📊 Datasets:
        - appearances.csv
        - clubs.csv
        - competitions.csv
        - game_events.csv
        - games.csv
        - player_valuations.csv
        - players.csv
        - transfers.csv

        ### 🎯 Modelos:
        - Random Forest
        - Logistic Regression
        - SVM
        - Naive Bayes
        - Neural Network

        ### 📈 Características:
        - Edad
        - Goles y asistencias
        - Minutos jugados
        - Tarjetas
        - Posición
        - Valor de transferencia
        """)
        st.subheader("📊 Vista previa de datos")
        sample_data = cargar_datos(default_path).get('players').head(10) if cargar_datos(
            default_path) else pd.DataFrame()
        st.dataframe(sample_data)

# Pestaña 2: Predictor de Resultados de Partidos
with tab2:
    st.header("⚽ Predictor de Resultados de Partidos")

    st.sidebar.subheader("Configuración de Partidos")
    n_estimators_match = st.sidebar.slider("Número de árboles", 50, 500, 100, key="n_estimators_match")
    max_depth_match = st.sidebar.slider("Profundidad máxima", 5, 20, 10, key="max_depth_match")
    n_matches_stats = st.sidebar.slider("Partidos previos para estadísticas", 3, 10, 5, key="n_matches_stats")

    train_match_button = st.sidebar.button("🚀 Entrenar Modelo de Partidos", key="train_match")

    if train_match_button:
        try:
            with st.spinner("Cargando datasets..."):
                datasets = cargar_datos(data_path)
                if datasets is None:
                    st.error("❌ No se pudieron cargar los datasets.")
                    st.stop()
                st.success("✅ Datasets cargados correctamente")

            with st.spinner("Preparando datos de partidos..."):
                df_games = datasets['games'].copy()
                df_clubs = datasets['clubs'].copy()
                df_appearances = datasets['appearances'].copy()


                def get_match_result(row):
                    if row['home_club_goals'] > row['away_club_goals']:
                        return 'home_win'
                    elif row['home_club_goals'] < row['away_club_goals']:
                        return 'away_win'
                    else:
                        return 'draw'


                df_games['result'] = df_games.apply(get_match_result, axis=1)

                df_clubs_features = df_clubs[['club_id', 'squad_size', 'average_age', 'foreigners_percentage',
                                              'national_team_players', 'stadium_seats']]

                df_games = df_games.merge(df_clubs_features, left_on='home_club_id', right_on='club_id',
                                          suffixes=('', '_home')).drop(columns=['club_id'])
                df_games = df_games.merge(df_clubs_features, left_on='away_club_id', right_on='club_id',
                                          suffixes=('_home', '_away')).drop(columns=['club_id'])


                def get_team_stats(df_games, team_id, game_date, n_matches=n_matches_stats):
                    past_games = df_games[(df_games['date'] < game_date) &
                                          ((df_games['home_club_id'] == team_id) |
                                           (df_games['away_club_id'] == team_id))].tail(n_matches)
                    if past_games.empty:
                        return pd.Series({'avg_goals_scored': 0, 'avg_goals_conceded': 0})

                    goals_scored = 0
                    goals_conceded = 0
                    for _, row in past_games.iterrows():
                        if row['home_club_id'] == team_id:
                            goals_scored += row['home_club_goals']
                            goals_conceded += row['away_club_goals']
                        else:
                            goals_scored += row['away_club_goals']
                            goals_conceded += row['home_club_goals']

                    return pd.Series({
                        'avg_goals_scored': goals_scored / len(past_games),
                        'avg_goals_conceded': goals_conceded / len(past_games)
                    })


                def parallel_get_team_stats(df_games, team_ids, game_date, n_matches=n_matches_stats, num_processes=None):
                    """
                    Parallelize get_team_stats across multiple team_ids.

                    Parameters:
                    - df_games: DataFrame containing game data
                    - team_ids: List of team IDs to process
                    - game_date: Date for filtering past games
                    - n_matches: Number of past matches to consider (default: 5)
                    - num_processes: Number of processes to use (default: None, uses CPU count)

                    Returns:
                    - DataFrame with team stats indexed by team_id
                    """
                    # Create a partial function with fixed df_games, game_date, and n_matches
                    func = partial(get_team_stats, df_games, game_date=game_date, n_matches=n_matches)

                    # Initialize process pool
                    with Pool(processes=num_processes) as pool:
                        # Map the function to team_ids and collect results
                        results = pool.map(func, team_ids)

                    # Combine results into a DataFrame
                    return pd.DataFrame(results, index=team_ids)


                # Convertir la columna 'date' a datetime
                df_games['date'] = pd.to_datetime(df_games['date'])

                # Aplicar estadísticas para equipos locales y visitantes
                home_stats = df_games.apply(lambda row: get_team_stats(df_games, row['home_club_id'], row['date']),
                                            axis=1)
                away_stats = df_games.apply(lambda row: get_team_stats(df_games, row['away_club_id'], row['date']),
                                            axis=1)

                # Añadir estadísticas al dataframe
                df_games[['home_avg_goals_scored', 'home_avg_goals_conceded']] = home_stats
                df_games[['away_avg_goals_scored', 'away_avg_goals_conceded']] = away_stats


                def get_team_player_stats(df_appearances, game_id, club_id):
                    team_appearances = df_appearances[
                        (df_appearances['game_id'] == game_id) &
                        (df_appearances['player_club_id'] == club_id)
                        ]

                    if team_appearances.empty:
                        # Return a Series to maintain consistent structure
                        return pd.Series({'team_avg_goals': 0, 'team_avg_assists': 0})

                    return pd.Series({
                        'team_avg_goals': team_appearances['goals'].mean(),
                        'team_avg_assists': team_appearances['assists'].mean()
                    })


                def get_team_stats_for_games(df_games, df_appearances, club_type='home'):
                    """
                    Obtiene estadísticas de equipos para todos los partidos.

                    Args:
                        df_games: Información de partidos
                        df_appearances: Apariciones de jugadores
                        club_type: 'home' o 'away'
                    """
                    club_col = f'{club_type}_club_id'

                    def process_row(row):
                        # Call the helper function for each row
                        return get_team_player_stats(df_appearances, row['game_id'], row[club_col])

                    # Use apply with axis=1; if process_row returns a Series, apply will return a DataFrame
                    stats = df_games.apply(process_row, axis=1)

                    # The result of apply is already a DataFrame, so just return it
                    return stats


                # Uso simplificado
                home_player_stats = get_team_stats_for_games(df_games, df_appearances, 'home')
                away_player_stats = get_team_stats_for_games(df_games, df_appearances, 'away')


                # Alternativa aún más eficiente usando operaciones vectorizadas
                def get_all_team_stats_vectorized(df_games, df_appearances):
                    """
                    Versión vectorizada más eficiente para procesar todas las estadísticas de una vez.
                    """
                    # Crear merge keys para home y away
                    home_stats = df_appearances.groupby(['game_id', 'player_club_id']).agg({
                        'goals': 'mean',
                        'assists': 'mean'
                    }).reset_index()

                    # Merge para equipos locales
                    home_merged = df_games.merge(
                        home_stats,
                        left_on=['game_id', 'home_club_id'],
                        right_on=['game_id', 'player_club_id'],
                        how='left'
                    )[['goals', 'assists']].fillna(0)
                    home_merged.columns = ['team_avg_goals', 'team_avg_assists']

                    # Merge para equipos visitantes
                    away_merged = df_games.merge(
                        home_stats,
                        left_on=['game_id', 'away_club_id'],
                        right_on=['game_id', 'player_club_id'],
                        how='left'
                    )[['goals', 'assists']].fillna(0)
                    away_merged.columns = ['team_avg_goals', 'team_avg_assists']

                    return home_merged, away_merged


                # Añadir al dataframe
                df_games[['home_team_avg_goals', 'home_team_avg_assists']] = home_player_stats
                df_games[['away_team_avg_goals', 'away_team_avg_assists']] = away_player_stats

                features = [
                    'squad_size_home', 'average_age_home', 'foreigners_percentage_home', 'national_team_players_home',
                    'stadium_seats_home', 'squad_size_away', 'average_age_away', 'foreigners_percentage_away',
                    'national_team_players_away', 'stadium_seats_away', 'home_avg_goals_scored',
                    'home_avg_goals_conceded', 'away_avg_goals_scored', 'away_avg_goals_conceded',
                    'home_team_avg_goals', 'home_team_avg_assists', 'away_team_avg_goals', 'away_team_avg_assists'
                ]

                df_games[features] = df_games[features].fillna(0)

                le = LabelEncoder()
                df_games['result_encoded'] = le.fit_transform(df_games['result'])

                X = df_games[features]
                y = df_games['result_encoded']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier(n_estimators=n_estimators_match, random_state=42,max_depth=max_depth_match)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                print("Accuracy:", accuracy_score(y_test, y_pred))
                print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

                st.metric('Accuracy',accuracy_score(y_test,y_pred))
                st.metric('Classification Report',classification_report(y_test, y_pred, target_names=le.classes_))


                def prepare_match_data(home_club_id, away_club_id, game_id, date, df_games, df_clubs, df_appearances):
                    # Create a dataframe with the match data
                    match_data = pd.DataFrame({
                        'game_id': [game_id],
                        'home_club_id': [home_club_id],
                        'away_club_id': [away_club_id],
                        'date': [pd.to_datetime(date)]
                    })

                    df_clubs_features = df_clubs[
                        ['club_id', 'name', 'squad_size', 'average_age', 'foreigners_percentage',
                         'national_team_players', 'stadium_seats']]

                    match_data = match_data.merge(df_clubs_features, left_on='home_club_id', right_on='club_id',
                                                  suffixes=('', '_home'), how='left').drop(columns=['club_id'],
                                                                                           errors='ignore')
                    match_data = match_data.merge(df_clubs_features, left_on='away_club_id', right_on='club_id',
                                                  suffixes=('_home', '_away'), how='left').drop(columns=['club_id'],
                                                                                                errors='ignore')

                    # Calculate stats for the last 5 matches for each team
                    def get_team_stats(df_games, team_id, game_date, n_matches=5):
                        past_games = df_games[(df_games['date'] < game_date) &
                                              ((df_games['home_club_id'] == team_id) |
                                               (df_games['away_club_id'] == team_id))].tail(n_matches)
                        if past_games.empty:
                            league_avg = df_games[['home_club_goals', 'away_club_goals']].mean()
                            return pd.Series({
                                'avg_goals_scored': league_avg['home_club_goals'],
                                'avg_goals_conceded': league_avg['away_club_goals']
                            })

                        goals_scored = 0
                        goals_conceded = 0
                        for _, row in past_games.iterrows():
                            if row['home_club_id'] == team_id:
                                goals_scored += row['home_club_goals']
                                goals_conceded += row['away_club_goals']
                            else:
                                goals_scored += row['away_club_goals']
                                goals_conceded += row['home_club_goals']

                        return pd.Series({
                            'avg_goals_scored': goals_scored / len(past_games),
                            'avg_goals_conceded': goals_conceded / len(past_games)
                        })

                    home_stats = match_data.apply(
                        lambda row: get_team_stats(df_games, row['home_club_id'], row['date']), axis=1)
                    away_stats = match_data.apply(
                        lambda row: get_team_stats(df_games, row['away_club_id'], row['date']), axis=1)
                    match_data[['home_avg_goals_scored', 'home_avg_goals_conceded']] = home_stats
                    match_data[['away_avg_goals_scored', 'away_avg_goals_conceded']] = away_stats

                    # Calculate player stats (average goals and assists per team)
                    team_stats = df_appearances[df_appearances['player_club_id'].isin([home_club_id, away_club_id])]
                    team_stats = team_stats.groupby('player_club_id')[['goals', 'assists']].mean().reset_index()
                    team_stats = team_stats.rename(columns={'goals': 'team_avg_goals', 'assists': 'team_avg_assists'})

                    match_data = match_data.merge(
                        team_stats[team_stats['player_club_id'] == home_club_id],
                        left_on='home_club_id',
                        right_on='player_club_id',
                        how='left'
                    ).rename(columns={
                        'team_avg_goals': 'home_team_avg_goals',
                        'team_avg_assists': 'home_team_avg_assists'
                    }).drop(columns=['player_club_id'], errors='ignore')

                    match_data = match_data.merge(
                        team_stats[team_stats['player_club_id'] == away_club_id],
                        left_on='away_club_id',
                        right_on='player_club_id',
                        how='left'
                    ).rename(columns={
                        'team_avg_goals': 'away_team_avg_goals',
                        'team_avg_assists': 'away_team_avg_assists'
                    }).drop(columns=['player_club_id'], errors='ignore')

                    # Ensure all features are present
                    features = [
                        'squad_size_home', 'average_age_home', 'foreigners_percentage_home',
                        'national_team_players_home',
                        'stadium_seats_home', 'squad_size_away', 'average_age_away', 'foreigners_percentage_away',
                        'national_team_players_away', 'stadium_seats_away', 'home_avg_goals_scored',
                        'home_avg_goals_conceded', 'away_avg_goals_scored', 'away_avg_goals_conceded',
                        'home_team_avg_goals', 'home_team_avg_assists', 'away_team_avg_goals', 'away_team_avg_assists'
                    ]

                    for feature in features:
                        if feature not in match_data.columns:
                            match_data[feature] = 0

                    match_data[features] = match_data[features].fillna(0)

                    # Keep club names for display
                    match_data['home_club_name'] = match_data['name_home']
                    match_data['away_club_name'] = match_data['name_away']

                    return match_data[features + ['home_club_name', 'away_club_name']]


                def predict_match_result(model, le, home_club_id, away_club_id, game_id, date, df_games, df_clubs,
                                         df_appearances):
                    # Prepare match data
                    match_data = prepare_match_data(home_club_id, away_club_id, game_id, date, df_games, df_clubs,
                                                    df_appearances)

                    # Extract club names
                    home_club_name = match_data['home_club_name'].iloc[
                        0] if 'home_club_name' in match_data.columns else 'Unknown Home Club'
                    away_club_name = match_data['away_club_name'].iloc[
                        0] if 'away_club_name' in match_data.columns else 'Unknown Away Club'

                    # Select features for prediction
                    features = [
                        'squad_size_home', 'average_age_home', 'foreigners_percentage_home',
                        'national_team_players_home',
                        'stadium_seats_home', 'squad_size_away', 'average_age_away', 'foreigners_percentage_away',
                        'national_team_players_away', 'stadium_seats_away', 'home_avg_goals_scored',
                        'home_avg_goals_conceded', 'away_avg_goals_scored', 'away_avg_goals_conceded',
                        'home_team_avg_goals', 'home_team_avg_assists', 'away_team_avg_goals', 'away_team_avg_assists'
                    ]
                    X_new = match_data[features]

                    # Make prediction
                    prediction = model.predict(X_new)
                    prediction_proba = model.predict_proba(X_new)
                    predicted_result = le.inverse_transform(prediction)[0]
                    proba_dict = dict(zip(le.classes_, prediction_proba[0]))

                    result_text = f"Partido: {home_club_name} vs {away_club_name}\n"
                    result_text += f"Resultado predicho: {predicted_result}\n"
                    result_text += "Probabilidades:\n"
                    for outcome, prob in proba_dict.items():
                        result_text += f"  {outcome}: {prob * 100:.2f}%\n"

                    return result_text



                st.session_state['match_model'] = model
                st.session_state['match_le'] = le
                st.session_state['df_games'] = df_games
                st.session_state['df_clubs'] = df_clubs
                st.session_state['df_appearances'] = df_appearances

        except Exception as e:
            st.error(f"❌ Error procesando datos: {str(e)}")
            st.stop()

    if 'df_clubs' not in st.session_state:
        datasets = cargar_datos(data_path)
        if datasets:
            st.session_state['df_clubs'] = datasets['clubs'].copy()

    if 'df_clubs' in st.session_state:
        nombres = st.session_state['df_clubs'][['club_id', 'name']].set_index('club_id')['name'].to_dict()
        opciones = list(nombres.values())

        col1, col2 = st.columns(2)
        with col1:
            home_club = st.selectbox("Equipo Local", opciones, key="home_club")
        with col2:
            away_club = st.selectbox("Equipo Visitante", opciones, key="away_club")

        match_date = st.date_input("Fecha del Partido", value=pd.to_datetime('2025-06-13'))
        game_id = st.number_input("ID del Partido", min_value=1, value=999999, step=1)

        if st.button("Predecir Resultado", key="predict_match"):
            if 'match_model' not in st.session_state:
                st.error("Primero hay que entrenar el modelo")
            else:
                try:
                    home_club_id = [k for k, v in nombres.items() if v == home_club][0]
                    away_club_id = [k for k, v in nombres.items() if v == away_club][0]
                    result = predict_match_result(
                        model=st.session_state['match_model'],
                        le=st.session_state['match_le'],
                        home_club_id=home_club_id,
                        away_club_id=away_club_id,
                        game_id=game_id,
                        date=match_date,
                        df_games=st.session_state['df_games'],
                        df_clubs=st.session_state['df_clubs'],
                        df_appearances=st.session_state['df_appearances']
                    )

                    st.markdown(f"""
                    <div class="metric-card">
                        <p><strong>Resultado Predicho:</strong> {result}</p>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    st.error(f" Error en la predicción:\n{tb_str}")
    else:
        st.warning("Carga los datos primero.")

    if not train_match_button:
        st.markdown("""
        """)
