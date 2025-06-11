import pandas as pd
import uuid
from datetime import datetime

# File paths for input CSVs (adjust paths as needed)
data_dir = "./csv/"
competitions_file = f"{data_dir}competitions_limpio.csv"
clubs_file = f"{data_dir}clubs_limpio.csv"
players_file = f"{data_dir}players_limpio.csv"
games_file = f"{data_dir}games_limpio.csv"
appearances_file = f"{data_dir}appearances_limpio.csv"
player_valuations_file = f"{data_dir}players_valuations_limpio.csv"

# Output directory for star schema CSVs
output_dir = "./star_schema/"
import os
os.makedirs(output_dir, exist_ok=True)

# Load CSV files
try:
    competitions = pd.read_csv(competitions_file)
    clubs = pd.read_csv(clubs_file)
    players = pd.read_csv(players_file)
    games = pd.read_csv(games_file)
    appearances = pd.read_csv(appearances_file)
    player_valuations = pd.read_csv(player_valuations_file)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all CSV files are in the {data_dir} directory.")
    exit(1)

# Create Dim_Competition
dim_competition = competitions[['competition_id', 'name', 'type', 'country_name']].copy()
dim_competition.to_csv(f"{output_dir}dim_competition.csv", index=False)

# Create Dim_Club
dim_club = clubs[['club_id', 'name', 'domestic_competition_id', 'squad_size', 'average_age', 'total_market_value', 'foreigners_number', 'national_team_players', 'stadium_name']].copy()
dim_club.to_csv(f"{output_dir}dim_club.csv", index=False)

# Create Dim_Player
dim_player = players[['player_id', 'name', 'position', 'sub_position', 'date_of_birth', 'height_in_cm']].copy()
dim_player.to_csv(f"{output_dir}dim_player.csv", index=False)

# Create Dim_Game
dim_game = games[['game_id', 'date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals', 'round', 'referee']].copy()
dim_game['date'] = pd.to_datetime(dim_game['date'], errors='coerce')
dim_game.to_csv(f"{output_dir}dim_game.csv", index=False)

# Create Dim_Season
dim_season = pd.DataFrame({'season': games['season'].unique()})
dim_season['season_start_date'] = dim_season['season'].apply(lambda x: f"{x}-07-01")
dim_season['season_end_date'] = dim_season['season'].apply(lambda x: f"{x+1}-06-30")
dim_season.to_csv(f"{output_dir}dim_season.csv", index=False)

# Create Fact_Performance
fact_performance = appearances[['game_id', 'player_id', 'player_club_id', 'competition_id', 'goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards']].copy()
fact_performance.rename(columns={'player_club_id': 'club_id'}, inplace=True)
fact_performance['performance_id'] = [str(uuid.uuid4()) for _ in range(len(fact_performance))]

# Add season and date from games
fact_performance = fact_performance.merge(games[['game_id', 'season', 'date']], on='game_id', how='left')
fact_performance['date'] = pd.to_datetime(fact_performance['date'], errors='coerce')

# Add market_value_in_eur from player_valuations
player_valuations['date'] = pd.to_datetime(player_valuations['date'], errors='coerce')
fact_performance = fact_performance.merge(
    player_valuations[['player_id', 'date', 'market_value_in_eur']],
    on='player_id',
    how='left'
)

# Handle date differences
fact_performance['date_diff'] = (fact_performance['date_x'] - fact_performance['date_y']).abs()

# Fill NaN in market_value_in_eur with 0 for records with no valuation
fact_performance['market_value_in_eur'] = fact_performance['market_value_in_eur'].fillna(0)

# Filter out records with NaN date_diff to avoid idxmin errors
valid_valuations = fact_performance.dropna(subset=['date_diff'])

# Select valuation with closest date for valid records
if not valid_valuations.empty:
    valid_valuations = valid_valuations.loc[valid_valuations.groupby(['performance_id'])['date_diff'].idxmin()]
else:
    valid_valuations = pd.DataFrame(columns=fact_performance.columns)

# Combine valid valuations with records that had no valuations
no_valuations = fact_performance[fact_performance['date_diff'].isna()]
fact_performance = pd.concat([valid_valuations, no_valuations], ignore_index=True)

# Select final columns
fact_performance = fact_performance[['performance_id', 'game_id', 'player_id', 'club_id', 'competition_id', 'season', 'goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'market_value_in_eur']]

# Save Fact_Performance
fact_performance.to_csv(f"{output_dir}fact_performance.csv", index=False)

print("Star schema tables created successfully in", output_dir)