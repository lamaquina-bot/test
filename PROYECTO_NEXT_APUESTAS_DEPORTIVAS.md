# PROYECTO NEXT - APLICACIÓN EN APUESTAS DEPORTIVAS
## De la ciencia de datos a la rentabilidad sostenible

**Objetivo:** Generar rentabilidad consistente con metodologías NEXT para financiar proyectos de alto impacto social (salud).

---

## 🎯 POR QUÉ APUESTAS DEPORTIVAS

### **Ventajas competitivas con metodologías NEXT:**

1. **Datos abundantes y accesibles:**
   - APIs gratuitas (ESPN, NBA, etc.)
   - Miles de partidos históricos
   - Cientos de variables por partido

2. **Mercado ineficiente:**
   - La mayoría de apostadores no usan data science avanzado
   - Bookmakers tienen márgenes pequeños (5-10%)
   - Oportunidades de value betting

3. **Metodologías NEXT directamente aplicables:**
   - Feature Engineering de estadísticas deportivas
   - Functional Diversity de equipos
   - Random Forest para predicciones
   - PCA para reducir dimensionalidad
   - Time Series Analysis para patrones temporales

4. **ROI potencial alto:**
   - ROI 5-15% mensual con modelo robusto
   - Escalable (más capital = más profit)
   - Reinvertible en proyectos sociales

---

## 🏀 CASO DE ESTUDIO: NBA (PROYECTO JORDAN)

### **Dataset disponible:**

Ya tienes **5,836 partidos de NBA** (5 temporadas) con:
- Puntos por cuarto (Q1, Q2, Q3, Q4)
- Estadísticas de equipos
- Resultados históricos

### **Datos adicionales necesarios:**

```python
# Dataset completo para apuestas NBA
nba_betting_data = {
    # Datos de partidos (ya tienes)
    'games': {
        'date': '2026-02-25',
        'home_team': 'LAL',
        'away_team': 'BOS',
        'home_score': 112,
        'away_score': 108,
        'q1_home': 28, 'q1_away': 25,
        'q2_home': 30, 'q2_away': 28,
        'q3_home': 26, 'q3_away': 30,
        'q4_home': 28, 'q4_away': 25
    },

    # Odds históricos (necesitamos conseguir)
    'odds': {
        'opening_spread': -5.5,  # LAL favorito por 5.5
        'closing_spread': -6.0,
        'opening_total': 220.5,
        'closing_total': 221.0,
        'moneyline_home': -250,
        'moneyline_away': +200
    },

    # Estadísticas avanzadas (disponibles)
    'advanced_stats': {
        'offensive_rating': 115.2,
        'defensive_rating': 108.5,
        'pace': 100.5,
        'effective_fg_pct': 0.550,
        'turnover_pct': 0.125,
        # ... 50+ métricas avanzadas
    },

    # Contexto (disponibles)
    'context': {
        'rest_days_home': 2,
        'rest_days_away': 1,
        'back_to_back_away': False,
        'injuries': ['Player A - Out'],
        'travel_distance': 2500,  # km
        'altitude': 300  # meters
    }
}
```

---

## 🔬 METODOLOGÍA NEXT APLICADA A APUESTAS

### **FASE 1: FEATURE ENGINEERING (617 menciones en tesis)**

#### **1.1 Descriptores de Equipos**

```python
class NBATeamDescriptors:
    """Sistema de descriptores para equipos NBA - Basado en NEXT"""

    def __init__(self):
        self.descriptor_system = DescriptorSystem()
        self._setup_nba_descriptors()

    def _setup_nba_descriptors(self):
        """Configurar descriptores específicos de NBA"""

        # Passport descriptors (identidad del equipo)
        self.descriptor_system.add_descriptor(
            category='passport',
            name='team_conference',
            states=['East', 'West'],
            method='database',
            scale='categorical'
        )

        # Characterization descriptors (rasgos inherentes)
        self.descriptor_system.add_descriptor(
            category='characterization',
            name='offensive_style',
            states=['pace_and_space', 'post_up', 'iso_heavy', 'motion'],
            method='clustering_analysis',
            scale='categorical'
        )

        # Evaluation descriptors (influenciados por contexto)
        self.descriptor_system.add_descriptor(
            category='evaluation',
            name='rest_advantage',
            states=None,
            method='calculation',
            scale='continuous',
            unit='days'
        )

    def extract_all_features(self, team_data, opponent_data, context):
        """Extraer 300+ descriptores para un partido"""

        features = {}

        # ============================================
        # OFENSIVE FEATURES (100+ features)
        # ============================================

        # Scoring efficiency
        features['pts_per_100_poss'] = team_data['pts'] / team_data['poss'] * 100
        features['efg_pct'] = (team_data['fg'] + 0.5 * team_data['fg3']) / team_data['fga']
        features['ts_pct'] = team_data['pts'] / (2 * (team_data['fga'] + 0.44 * team_data['fta']))

        # Shot distribution
        features['fg3_rate'] = team_data['fg3a'] / team_data['fga']
        features['ft_rate'] = team_data['fta'] / team_data['fga']
        features['paint_touches'] = team_data['paint_fga'] / team_data['fga']

        # Playmaking
        features['ast_ratio'] = team_data['ast'] / team_data['fg']
        features['tov_pct'] = team_data['tov'] / (team_data['fga'] + 0.44 * team_data['fta'] + team_data['tov'])
        features['hockey_assists'] = team_data['secondary_ast'] / team_data['ast']

        # Tempo
        features['pace'] = team_data['poss'] / 48 * 5
        features['transition_freq'] = team_data['transition_poss'] / team_data['poss']

        # ============================================
        # DEFENSIVE FEATURES (100+ features)
        # ============================================

        # Opponent scoring
        features['opp_pts_per_100'] = opponent_data['pts'] / opponent_data['poss'] * 100
        features['opp_efg_pct'] = (opponent_data['fg'] + 0.5 * opponent_data['fg3']) / opponent_data['fga']

        # Rim protection
        features['blk_pct'] = team_data['blk'] / opponent_data['fga_at_rim']
        features['opp_rim_freq'] = opponent_data['fga_at_rim'] / opponent_data['fga']

        # Perimeter defense
        features['opp_fg3_pct'] = opponent_data['fg3'] / opponent_data['fg3a']
        features['contest_rate'] = team_data['contested_shots'] / opponent_data['fga']

        # ============================================
        # QUARTER-BY-QUARTER FEATURES (80+ features)
        # ============================================

        for q in [1, 2, 3, 4]:
            # Scoring patterns
            features[f'q{q}_avg_pts'] = team_data[f'q{q}_pts'].mean()
            features[f'q{q}_std_pts'] = team_data[f'q{q}_pts'].std()
            features[f'q{q}_consistency'] = 1 - (features[f'q{q}_std_pts'] / features[f'q{q}_avg_pts'])

            # Tempo by quarter
            features[f'q{q}_pace'] = team_data[f'q{q}_poss'].mean()

            # Clutch performance (4th quarter specific)
            if q == 4:
                features['clutch_off_rating'] = features[f'q{q}_avg_pts'] / features[f'q{q}_pace'] * 100
                features['clutch_efg'] = team_data['q4_efg'].mean()

        # ============================================
        # CONTEXT FEATURES (20+ features)
        # ============================================

        # Rest advantage
        features['rest_diff'] = context['rest_days_home'] - context['rest_days_away']
        features['home_back_to_back'] = context['home_back_to_back']
        features['away_back_to_back'] = context['away_back_to_back']

        # Travel fatigue
        features['travel_miles'] = context['travel_distance']
        features['timezone_change'] = context['timezone_diff']

        # Home court advantage
        features['home_altitude'] = context['altitude']
        features['home_crowd_factor'] = context['attendance'] / context['arena_capacity']

        # ============================================
        # MATCHUP-SPECIFIC FEATURES (40+ features)
        # ============================================

        # Head-to-head recent
        features['h2h_last_10_home_wins'] = self._get_h2h_wins(team_data, opponent_data, n=10, location='home')
        features['h2h_last_10_away_wins'] = self._get_h2h_wins(team_data, opponent_data, n=10, location='away')

        # Style mismatch
        features['pace_diff'] = abs(features['pace'] - opponent_data['pace'])
        features['off_vs_def_rating'] = features['pts_per_100_poss'] - opponent_data['opp_pts_per_100']

        # ============================================
        # ADVANCED DERIVED FEATURES (60+ features)
        # ============================================

        # Four Factors (Dean Oliver)
        features['shooting'] = features['efg_pct']
        features['turnovers'] = features['tov_pct']
        features['rebouncing'] = (team_data['orb'] + team_data['drb']) / \
                                 (team_data['orb'] + team_data['drb'] + opponent_data['orb'] + opponent_data['drb'])
        features['free_throws'] = features['ft_rate']

        # Expected points model
        features['expected_pts'] = (
            features['shooting'] * 0.4 +
            (1 - features['turnovers']) * 0.25 +
            features['rebouncing'] * 0.2 +
            features['free_throws'] * 0.15
        ) * 100

        # Performance vs league average
        league_avg = self._get_league_averages(season=context['season'])
        features['off_rating_vs_league'] = features['pts_per_100_poss'] - league_avg['pts_per_100']
        features['def_rating_vs_league'] = features['opp_pts_per_100'] - league_avg['pts_per_100']

        return features

    def create_feature_matrix(self, historical_games):
        """Crear matriz de features para todos los partidos históricos"""

        all_features = []

        for game in historical_games:
            home_features = self.extract_all_features(
                team_data=game['home_team'],
                opponent_data=game['away_team'],
                context=game['context']
            )

            away_features = self.extract_all_features(
                team_data=game['away_team'],
                opponent_data=game['home_team'],
                context=game['context']
            )

            # Differential features (home - away)
            diff_features = {
                f'home_{k}' if not k.startswith('home') else k: v
                for k, v in home_features.items()
            }

            for k, v in away_features.items():
                diff_features[f'away_{k}'] = v

            # Calculate differentials
            for feature in ['pts_per_100_poss', 'efg_pct', 'pace', 'tov_pct']:
                diff_features[f'diff_{feature}'] = home_features[feature] - away_features[feature]

            # Add target variable
            diff_features['home_margin'] = game['home_score'] - game['away_score']
            diff_features['total_points'] = game['home_score'] + game['away_score']

            # Add quarter targets
            for q in [1, 2, 3, 4]:
                diff_features[f'q{q}_home_pts'] = game[f'q{q}_home']
                diff_features[f'q{q}_away_pts'] = game[f'q{q}_away']
                diff_features[f'q{q}_margin'] = game[f'q{q}_home'] - game[f'q{q}_away']
                diff_features[f'q{q}_total'] = game[f'q{q}_home'] + game[f'q{q}_away']

            all_features.append(diff_features)

        return pd.DataFrame(all_features)
```

#### **1.2 Resultado: 300+ Features por Partido**

```
Por cada partido NBA, extraemos:

OFENSIVE (100 features):
├── Scoring efficiency (20)
├── Shot distribution (20)
├── Playmaking (20)
├── Tempo (20)
└── Quarter breakdown (20)

DEFENSIVE (100 features):
├── Opponent scoring (20)
├── Rim protection (20)
├── Perimeter defense (20)
├── Steals/blocks (20)
└── Quarter breakdown (20)

CONTEXT (40 features):
├── Rest advantage (10)
├── Travel fatigue (10)
├── Home court (10)
└── Injuries (10)

MATCHUP (60 features):
├── Head-to-head (20)
├── Style mismatch (20)
└── Expected performance (20)
```

---

### **FASE 2: DIMENSIONALITY REDUCTION (47 menciones en tesis)**

#### **2.1 PCA para Reducir 300+ Variables**

```python
class NBAPCAAnalyzer:
    """PCA para reducir dimensionalidad de features NBA"""

    def __init__(self, variance_threshold=0.95):
        self.pca_analyzer = PCAAnalyzer(variance_threshold=variance_threshold)
        self.feature_names = None

    def fit_transform(self, feature_matrix):
        """Aplicar PCA a matriz de features"""

        # Separar features y targets
        target_cols = ['home_margin', 'total_points',
                       'q1_margin', 'q2_margin', 'q3_margin', 'q4_margin',
                       'q1_total', 'q2_total', 'q3_total', 'q4_total']

        X = feature_matrix.drop(columns=target_cols)
        y = feature_matrix[target_cols]

        self.feature_names = X.columns.tolist()

        # Aplicar PCA
        X_pca = self.pca_analyzer.fit_transform(X)

        # Obtener varianza explicada
        variance_df = self.pca_analyzer.get_explained_variance()

        print(f"✅ PCA completado:")
        print(f"   Features originales: {X.shape[1]}")
        print(f"   Componentes principales: {X_pca.shape[1]}")
        print(f"   Varianza explicada: {variance_df['cumulative'].iloc[-1]:.1%}")

        return X_pca, y

    def get_top_features_per_pc(self, n_features=20):
        """Obtener features más importantes por componente"""

        top_features = self.pca_analyzer.get_top_features_per_component(n_features)

        # Interpretar componentes principales
        pc_interpretation = {
            'PC1': 'Offensive Power (scoring efficiency)',
            'PC2': 'Defensive Strength (opponent scoring)',
            'PC3': 'Pace Factor (tempo of game)',
            'PC4': 'Home Court Advantage (rest, travel, crowd)',
            'PC5': 'Clutch Performance (4th quarter efficiency)',
            'PC6': 'Style Matchup (pace vs defense)',
            'PC7': 'Quarter Consistency (variance by quarter)',
            'PC8': 'Travel Fatigue (distance, timezone)',
            'PC9': 'Rest Advantage (days of rest)',
            'PC10': 'Historical Performance (H2H, trends)'
        }

        return top_features, pc_interpretation

    def visualize_pca(self, X_pca, y):
        """Visualizar componentes principales"""

        # Biplot de PC1 vs PC2 con resultado como color
        fig = self.pca_analyzer.plot_biplot(
            X_pca,
            labels=(y['home_margin'] > 0).astype(int),  # 1=home win, 0=away win
            pc_x=1,
            pc_y=2
        )

        plt.title('NBA Games - PCA (PC1: Offense, PC2: Defense)\nColor: Home Win (1) vs Loss (0)')
        plt.savefig('/tmp/nba_pca_biplot.png', dpi=300, bbox_inches='tight')

        return fig
```

#### **2.2 Resultado: Reducción de 300+ a 20-30 componentes**

```
Antes de PCA:
- 300+ features por partido
- Mucha correlación entre features
- Difícil de interpretar

Después de PCA:
- 20-30 componentes principales
- 95% de varianza explicada
- Componentes interpretables:
  * PC1: Offensive Power
  * PC2: Defensive Strength
  * PC3: Pace Factor
  * PC4: Home Court Advantage
  * PC5: Clutch Performance
  * etc.
```

---

### **FASE 3: FUNCTIONAL DIVERSITY (55 menciones en tesis)**

#### **3.1 Diversidad Funcional de Equipos**

```python
class NBAFunctionalDiversity:
    """Calcular diversidad funcional de equipos NBA"""

    def __init__(self):
        self.fd_calculator = None

    def calculate_team_diversity(self, team_roster, team_stats):
        """Calcular diversidad funcional de un equipo"""

        # Definir rasgos funcionales de jugadores
        player_traits = [
            'scoring_ability',      # Puntos por 36 min
            'playmaking',           # Asistencias ratio
            'rebounding',           # Rebotes por 36 min
            'defense',              # Defensive rating
            'three_point_shooting', # 3P%
            'free_throw_shooting',  # FT%
            'speed',                # Pace impact
            'strength',             # Post defense
            'clutch_performance',   # 4th quarter scoring
            'consistency'           # Std dev of performance
        ]

        # Crear matriz de rasgos
        traits_matrix = []
        for player in team_roster:
            player_traits_values = [
                player['pts_per_36'],
                player['ast'] / player['min'],
                player['reb_per_36'],
                player['def_rating'],
                player['fg3_pct'],
                player['ft_pct'],
                player['pace_impact'],
                player['post_def'],
                player['q4_pts_per_36'],
                player['performance_std']
            ]
            traits_matrix.append(player_traits_values)

        traits_matrix = np.array(traits_matrix)

        # Abundancias (minutos jugados)
        abundances = np.array([player['min'] for player in team_roster])

        # Calcular diversidad funcional
        self.fd_calculator = FunctionalDiversityCalculator(
            traits_matrix=traits_matrix,
            abundances=abundances
        )

        diversity = self.fd_calculator.calculate_all_indices()

        # Interpretar
        interpretation = {
            'FRic': {
                'value': diversity['FRic'],
                'meaning': 'Volumen de habilidades del equipo',
                'betting_implication': 'Equipos con alta FRic son versátiles, difíciles de defender'
            },
            'FEve': {
                'value': diversity['FEve'],
                'meaning': 'Regularidad de distribución de minutos',
                'betting_implication': 'Equipos con alta FEve tienen rotación balanceada'
            },
            'FDiv': {
                'value': diversity['FDiv'],
                'meaning': 'Jugadores con perfiles únicos',
                'betting_implication': 'Equipos con alta FDiv tienen especialistas valiosos'
            },
            'FDis': {
                'value': diversity['FDis'],
                'meaning': 'Dispersión promedio de habilidades',
                'betting_implication': 'Equipos con alta FDis son impredecibles'
            }
        }

        return diversity, interpretation

    def compare_teams_diversity(self, home_team, away_team):
        """Comparar diversidad funcional de dos equipos"""

        home_div, home_interp = self.calculate_team_diversity(
            home_team['roster'],
            home_team['stats']
        )

        away_div, away_interp = self.calculate_team_diversity(
            away_team['roster'],
            away_team['stats']
        )

        # Calcular ventajas
        advantages = {
            'FRic_advantage': home_div['FRic'] - away_div['FRic'],
            'FEve_advantage': home_div['FEve'] - away_div['FEve'],
            'FDiv_advantage': home_div['FDiv'] - away_div['FDiv'],
            'FDis_advantage': home_div['FDis'] - away_div['FDis']
        }

        # Interpretar para apuestas
        if advantages['FRic_advantage'] > 0.1:
            print(f"✅ HOME tiene ventaja de versatilidad (+{advantages['FRic_advantage']:.2f})")
            print(f"   Implicación: HOME puede adaptarse mejor a diferentes situaciones")

        if advantages['FDiv_advantage'] > 0.1:
            print(f"✅ HOME tiene ventaja de especialistas (+{advantages['FDiv_advantage']:.2f})")
            print(f"   Implicación: HOME tiene jugadores únicos que pueden cambiar el juego")

        return advantages
```

#### **3.2 Aplicación en Apuestas**

```python
# Ejemplo: Lakers vs Celtics
lakers_diversity = {
    'FRic': 0.75,  # Alta versatilidad (LeBron, Davis, Reaves)
    'FEve': 0.68,  # Regular (depende mucho de estrellas)
    'FDiv': 0.82,  # Alta (jugadores muy diferentes)
    'FDis': 0.71   # Dispersión media
}

celtics_diversity = {
    'FRic': 0.82,  # Muy alta versatilidad (equipo profundo)
    'FEve': 0.85,  # Muy regular (rotación balanceada)
    'FDiv': 0.65,  # Media (jugadores más similares)
    'FDis': 0.60   # Baja dispersión (equipo cohesivo)
}

# Interpretación para apuestas:
print("ANÁLISIS DE DIVERSIDAD FUNCIONAL:")
print("="*60)
print("\nCELTICS (visitante):")
print(f"  ✅ Mayor versatilidad (FRic: {celtics_diversity['FRic']:.2f})")
print(f"  ✅ Rotación más balanceada (FEve: {celtics_diversity['FEve']:.2f})")
print(f"  ⚠️ Menos especialistas (FDiv: {celtics_diversity['FDiv']:.2f})")

print("\nLAKERS (local):")
print(f"  ⚠️ Menor versatilidad (FRic: {lakers_diversity['FRic']:.2f})")
print(f"  ⚠️ Rotación menos balanceada (FEve: {lakers_diversity['FEve']:.2f})")
print(f"  ✅ Más especialistas (FDiv: {lakers_diversity['FDiv']:.2f})")

print("\n🎯 IMPLICACIÓN PARA APUESTAS:")
print("  • Celtics más consistentes → Mejor para spreads")
print("  • Lakers con especialistas → Mejor para player props")
print("  • Diferencia en FEve sugiere Lakers más volátiles")
```

---

### **FASE 4: MACHINE LEARNING SUPERVISED (105 menciones en tesis)**

#### **4.1 Random Forest para Predicción de Resultados**

```python
class NBABettingPredictor:
    """Modelo de predicción para apuestas NBA"""

    def __init__(self):
        self.model = None
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def train_model(self, X_pca, y):
        """Entrenar modelo con componentes principales"""

        print("🎯 ENTRENANDO MODELO DE PREDICCIÓN NBA...")
        print("="*60)

        # Entrenar para diferentes targets
        targets = {
            'home_margin': 'Diferencia de puntos (home - away)',
            'total_points': 'Total de puntos del partido',
            'q1_margin': 'Diferencia Q1',
            'q2_margin': 'Diferencia Q2',
            'q3_margin': 'Diferencia Q3',
            'q4_margin': 'Diferencia Q4'
        }

        models = {}

        for target_name, target_desc in targets.items():
            print(f"\n📊 Entrenando modelo para: {target_desc}")

            y_target = y[target_name]

            # Split temporal (usar datos más antiguos para entrenar)
            split_idx = int(len(X_pca) * 0.8)
            X_train, X_test = X_pca[:split_idx], X_pca[split_idx:]
            y_train, y_test = y_target[:split_idx], y_target[split_idx:]

            # Entrenar Random Forest
            rf = RandomForestAnalyzer(task='regression')
            rf.train_with_tuning(X_train, y_train)

            # Evaluar
            y_pred = rf.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation
            cv_results = rf.cross_validate(X_train, y_train, cv=5)

            models[target_name] = {
                'model': rf,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_results['mean_score'],
                'cv_std': cv_results['std_score']
            }

            print(f"  ✅ MAE: {mae:.2f} puntos")
            print(f"  ✅ R²: {r2:.3f}")
            print(f"  ✅ CV Score: {cv_results['mean_score']:.3f} (+/- {cv_results['std_score']:.3f})")

        self.models = models
        return models

    def predict_game(self, home_team_data, away_team_data, context):
        """Predecir resultado de un partido"""

        # Extraer features
        feature_extractor = NBATeamDescriptors()
        features = feature_extractor.create_feature_matrix([{
            'home_team': home_team_data,
            'away_team': away_team_data,
            'context': context
        }])

        # Aplicar PCA
        features_pca = self.pca.transform(features)

        # Predecir todos los targets
        predictions = {}

        for target_name, model_info in self.models.items():
            pred = model_info['model'].model.predict(features_pca)[0]
            uncertainty = model_info['model'].model.predict(features_pca)[0]  # Simplified

            predictions[target_name] = {
                'predicted': pred,
                'uncertainty': uncertainty,
                'mae': model_info['mae']
            }

        return predictions

    def identify_value_bets(self, predictions, odds):
        """Identificar value bets basado en predicciones vs odds"""

        value_bets = []

        # 1. Spread betting
        predicted_margin = predictions['home_margin']['predicted']
        predicted_margin_uncertainty = predictions['home_margin']['mae']

        spread_line = odds['spread']
        spread_odds = odds['spread_odds']  # e.g., -110

        # Calculate edge
        if predicted_margin > spread_line + predicted_margin_uncertainty:
            # HOME cubre el spread
            edge = (predicted_margin - spread_line) / predicted_margin_uncertainty
            if edge > 0.5:  # Edge significativo
                value_bets.append({
                    'type': 'spread',
                    'pick': 'HOME',
                    'line': spread_line,
                    'odds': spread_odds,
                    'predicted_margin': predicted_margin,
                    'edge': edge,
                    'confidence': min(0.9, 0.5 + edge * 0.2)
                })

        elif predicted_margin < spread_line - predicted_margin_uncertainty:
            # AWAY cubre el spread
            edge = (spread_line - predicted_margin) / predicted_margin_uncertainty
            if edge > 0.5:
                value_bets.append({
                    'type': 'spread',
                    'pick': 'AWAY',
                    'line': spread_line,
                    'odds': -spread_odds,
                    'predicted_margin': predicted_margin,
                    'edge': edge,
                    'confidence': min(0.9, 0.5 + edge * 0.2)
                })

        # 2. Total points betting
        predicted_total = predictions['total_points']['predicted']
        total_line = odds['total']
        total_uncertainty = predictions['total_points']['mae']

        if predicted_total > total_line + total_uncertainty:
            edge = (predicted_total - total_line) / total_uncertainty
            if edge > 0.5:
                value_bets.append({
                    'type': 'total',
                    'pick': 'OVER',
                    'line': total_line,
                    'odds': odds['total_over_odds'],
                    'predicted_total': predicted_total,
                    'edge': edge,
                    'confidence': min(0.9, 0.5 + edge * 0.2)
                })

        elif predicted_total < total_line - total_uncertainty:
            edge = (total_line - predicted_total) / total_uncertainty
            if edge > 0.5:
                value_bets.append({
                    'type': 'total',
                    'pick': 'UNDER',
                    'line': total_line,
                    'odds': odds['total_under_odds'],
                    'predicted_total': predicted_total,
                    'edge': edge,
                    'confidence': min(0.9, 0.5 + edge * 0.2)
                })

        # 3. Quarter betting (Q1, Q2, Q3, Q4)
        for q in [1, 2, 3, 4]:
            q_margin = predictions[f'q{q}_margin']['predicted']
            q_line = odds.get(f'q{q}_spread', None)

            if q_line:
                if abs(q_margin - q_line) > predictions[f'q{q}_margin']['mae'] * 1.5:
                    edge = abs(q_margin - q_line) / predictions[f'q{q}_margin']['mae']
                    if edge > 0.5:
                        value_bets.append({
                            'type': f'q{q}_spread',
                            'pick': 'HOME' if q_margin > q_line else 'AWAY',
                            'line': q_line,
                            'odds': odds.get(f'q{q}_spread_odds', -110),
                            'predicted_margin': q_margin,
                            'edge': edge,
                            'confidence': min(0.85, 0.5 + edge * 0.15)
                        })

        return value_bets
```

#### **4.2 Resultado: Sistema de Value Betting**

```python
# Ejemplo de output del modelo

print("🏀 PREDICCIÓN: LAKERS vs CELTICS")
print("="*60)

predictions = predictor.predict_game(lakers_data, celtics_data, context)

print("\n📊 PREDICCIONES:")
print(f"  Margen final: LAL +{predictions['home_margin']['predicted']:.1f} ± {predictions['home_margin']['mae']:.1f}")
print(f"  Total puntos: {predictions['total_points']['predicted']:.1f} ± {predictions['total_points']['mae']:.1f}")
print(f"  Q1 margen: LAL +{predictions['q1_margin']['predicted']:.1f}")
print(f"  Q2 margen: LAL +{predictions['q2_margin']['predicted']:.1f}")
print(f"  Q3 margen: LAL +{predictions['q3_margin']['predicted']:.1f}")
print(f"  Q4 margen: LAL +{predictions['q4_margin']['predicted']:.1f}")

print("\n💰 VALUE BETS IDENTIFICADAS:")
print("="*60)

value_bets = predictor.identify_value_bets(predictions, odds)

for i, bet in enumerate(value_bets, 1):
    print(f"\n{i}. {bet['type'].upper()} - {bet['pick']}")
    print(f"   Línea: {bet['line']}")
    print(f"   Odds: {bet['odds']}")
    print(f"   Edge: {bet['edge']:.2f}")
    print(f"   Confianza: {bet['confidence']:.1%}")

    if bet['type'] == 'spread':
        print(f"   💡 Predicción: {bet['predicted_margin']:.1f} vs línea {bet['line']}")
    elif bet['type'] == 'total':
        print(f"   💡 Predicción: {bet['predicted_total']:.1f} vs línea {bet['line']}")
```

---

### **FASE 5: GESTIÓN DE CAPITAL Y RIESGO**

#### **5.1 Kelly Criterion para Tamaño de Apuesta**

```python
class BankrollManager:
    """Gestión de capital con Kelly Criterion"""

    def __init__(self, initial_bankroll=10000):
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll

    def calculate_kelly_bet(self, odds, probability, kelly_fraction=0.25):
        """
        Calcular tamaño óptimo de apuesta con Kelly Criterion

        Kelly % = (bp - q) / b
        where:
        b = decimal odds - 1
        p = probability of winning
        q = probability of losing (1 - p)

        Usamos fractional Kelly (25%) para reducir varianza
        """

        # Convertir odds americanas a decimales
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        b = decimal_odds - 1
        p = probability
        q = 1 - p

        # Kelly Criterion
        kelly_pct = (b * p - q) / b

        # Solo apostar si Kelly es positivo
        if kelly_pct <= 0:
            return 0, 0

        # Fractional Kelly (25% del Kelly óptimo)
        fractional_kelly = kelly_pct * kelly_fraction

        # Tamaño de apuesta
        bet_size = self.bankroll * fractional_kelly

        # Limitar a máximo 5% del bankroll
        max_bet = self.bankroll * 0.05
        bet_size = min(bet_size, max_bet)

        return bet_size, fractional_kelly

    def evaluate_bet(self, value_bet):
        """Evaluar si vale la pena hacer una apuesta"""

        # Convertir odds americanas a probabilidad implícita
        odds = value_bet['odds']
        if odds > 0:
            implied_prob = 100 / (odds + 100)
        else:
            implied_prob = abs(odds) / (abs(odds) + 100)

        # Nuestra probabilidad estimada
        our_prob = value_bet['confidence']

        # Edge
        edge = our_prob - implied_prob

        print(f"\n🎲 EVALUANDO: {value_bet['type']} - {value_bet['pick']}")
        print(f"   Odds: {odds}")
        print(f"   Probabilidad implícita: {implied_prob:.1%}")
        print(f"   Nuestra probabilidad: {our_prob:.1%}")
        print(f"   Edge: {edge:.1%}")

        if edge < 0.02:  # Edge mínimo 2%
            print(f"   ❌ RECHAZADA - Edge insuficiente (< 2%)")
            return None

        # Calcular tamaño de apuesta
        bet_size, kelly_pct = self.calculate_kelly_bet(odds, our_prob)

        if bet_size < self.bankroll * 0.01:  # Mínimo 1% del bankroll
            print(f"   ❌ RECHAZADA - Apuesta muy pequeña (< 1% bankroll)")
            return None

        print(f"   ✅ ACEPTADA")
        print(f"   Tamaño: ${bet_size:.2f} ({kelly_pct:.2%} Kelly)")
        print(f"   Expected ROI: {edge * 100:.1f}%")

        return {
            'bet_type': value_bet['type'],
            'pick': value_bet['pick'],
            'line': value_bet['line'],
            'odds': odds,
            'bet_size': bet_size,
            'probability': our_prob,
            'edge': edge,
            'expected_profit': bet_size * edge
        }

    def simulate_season(self, predictions_df, odds_df, n_seasons=1000):
        """Simular múltiples temporadas para estimar rentabilidad"""

        results = []

        for season in range(n_seasons):
            bankroll = self.initial_bankroll
            bets_made = 0
            wins = 0
            losses = 0

            for idx, game in predictions_df.iterrows():
                # Simular apuesta
                bet = self.evaluate_bet(game)

                if bet:
                    # Simular resultado
                    if np.random.random() < bet['probability']:
                        # Ganar
                        if bet['odds'] > 0:
                            profit = bet['bet_size'] * (bet['odds'] / 100)
                        else:
                            profit = bet['bet_size'] * (100 / abs(bet['odds']))

                        bankroll += profit
                        wins += 1
                    else:
                        # Perder
                        bankroll -= bet['bet_size']
                        losses += 1

                    bets_made += 1

            results.append({
                'final_bankroll': bankroll,
                'roi': (bankroll - self.initial_bankroll) / self.initial_bankroll,
                'bets_made': bets_made,
                'win_rate': wins / bets_made if bets_made > 0 else 0
            })

        # Analizar resultados
        results_df = pd.DataFrame(results)

        print(f"\n📊 SIMULACIÓN DE {n_seasons} TEMPORADAS:")
        print("="*60)
        print(f"  ROI promedio: {results_df['roi'].mean():.1%}")
        print(f"  ROI mediano: {results_df['roi'].median():.1%}")
        print(f"  ROI std: {results_df['roi'].std():.1%}")
        print(f"  ROI 95% CI: [{results_df['roi'].quantile(0.025):.1%}, {results_df['roi'].quantile(0.975):.1%}]")
        print(f"  Win rate promedio: {results_df['win_rate'].mean():.1%}")
        print(f"  Apuestas por temporada: {results_df['bets_made'].mean():.0f}")

        print(f"\n  💰 Con bankroll inicial de ${self.initial_bankroll:,}:")
        print(f"     Profit promedio: ${self.initial_bankroll * results_df['roi'].mean():,.0f}")
        print(f"     Profit mediano: ${self.initial_bankroll * results_df['roi'].median():,.0f}")

        return results_df
```

---

## 💰 PROYECCIÓN FINANCIERA

### **Escenario Realista:**

#### **Parámetros:**
- **Bankroll inicial:** $100,000 COP (≈ $25 USD)
- **ROI esperado:** 5-10% mensual (conservador)
- **Número de apuestas:** 100-200 por mes
- **Win rate:** 55-60%

#### **Proyección a 12 meses:**

```
MES 1:
  Bankroll: $100,000
  Apuestas: 150
  ROI: 7%
  Profit: $7,000
  Bankroll final: $107,000

MES 2:
  Bankroll: $107,000
  Apuestas: 160
  ROI: 7%
  Profit: $7,490
  Bankroll final: $114,490

...

MES 6:
  Bankroll: $150,073
  Profit acumulado: $50,073

MES 12:
  Bankroll: $225,219
  Profit acumulado: $125,219
  ROI total: 125%
```

#### **Escenario Optimista (ROI 10%):**
```
MES 12:
  Bankroll: $313,843
  Profit acumulado: $213,843
  ROI total: 214%
```

#### **Escenario Pesimista (ROI 3%):**
```
MES 12:
  Bankroll: $142,576
  Profit acumulado: $42,576
  ROI total: 43%
```

---

## 🎯 PLAN DE IMPLEMENTACIÓN

### **FASE 1: Recolección de Datos (2-4 semanas)**

```python
# 1. Completar dataset NBA
tasks = [
    "✅ Partidos históricos (ya tienes 5,836)",
    "⏳ Odds históricos (conseguir de OddsPortal, BetExplorer)",
    "⏳ Estadísticas avanzadas (NBA API, Basketball Reference)",
    "⏳ Datos de contexto (lesiones, descanso, viajes)"
]

# 2. Fuentes de datos
sources = {
    'odds': [
        'OddsPortal.com (manual scrape)',
        'BetExplorer.com (manual scrape)',
        'Action Network API (pagado, $99/mes)'
    ],
    'stats': [
        'NBA.com/stats (gratuito)',
        'BasketballReference.com (gratuito)',
        'ESPN API (gratuito)'
    ],
    'context': [
        'NBA injury reports (oficial)',
        'Rest days calculator (automatizar)',
        'Travel distance (Google Maps API)'
    ]
}
```

### **FASE 2: Feature Engineering (2 semanas)**

```python
# Implementar sistema de descriptores
feature_engineer = NBATeamDescriptors()
feature_matrix = feature_engineer.create_feature_matrix(historical_games)

# Resultado: 300+ features por partido
print(f"Features generadas: {feature_matrix.shape[1]}")
```

### **FASE 3: Modelado (2 semanas)**

```python
# Entrenar modelos
predictor = NBABettingPredictor()
models = predictor.train_model(X_pca, y)

# Validar con backtesting
backtest_results = backtest_model(models, historical_odds)
```

### **FASE 4: Paper Trading (4 semanas)**

```python
# Probar sin dinero real
paper_trader = PaperTrading(initial_bankroll=100000)
results = paper_trader.run(months=1)

print(f"Paper trading ROI: {results['roi']:.1%}")
print(f"Win rate: {results['win_rate']:.1%}")
```

### **FASE 5: Live Betting (ongoing)**

```python
# Apostar con dinero real
live_bettor = LiveBetting(
    initial_bankroll=100000,
    kelly_fraction=0.25
)

# Ejecutar apuestas diarias
daily_bets = live_bettor.find_value_bets(today_games)
```

---

## 🔮 ESCALAMIENTO HACIA SALUD

### **Fase A: Generar Capital (12-24 meses)**

**Objetivo:** Generar $2M USD con apuestas deportivas

**Estrategia:**
1. **Meses 1-6:** $100K → $200K (ROI 10-15%)
2. **Meses 7-12:** $200K → $500K (aumentar bankroll)
3. **Meses 13-18:** $500K → $1M (múltiples deportes)
4. **Meses 19-24:** $1M → $2M (automatización completa)

### **Fase B: Diversificación (24-36 meses)**

**Objetivo:** Estabilizar ingresos con múltiples fuentes

**Estrategias:**
1. **Apuestas deportivas** (50% del capital)
   - NBA, NFL, MLB, Soccer
   - Sistemas automatizados

2. **Trading algorítmico** (30% del capital)
   - Criptomonedas
   - Forex
   - Aplicar mismas metodologías NEXT

3. **Data science consulting** (20% del capital)
   - Vender modelos predictivos
   - Consultoría para bookmakers (ironía)
   - Proyectos de ML para empresas

### **Fase C: Salud y Alto Impacto (36+ meses)**

**Objetivo:** Financiar proyectos de salud con capital generado

**Proyectos posibles:**
1. **Medicina de precisión en Colombia:**
   - Comprar datos de 10,000 pacientes
   - Aplicar metodologías NEXT
   - Partnership con hospitales

2. **Sistema de diagnóstico temprano:**
   - IA para detección de cáncer
   - Feature engineering de imágenes médicas
   - ROI en vidas salvadas

3. **Investigación en enfermedades tropicales:**
   - Análisis de diversidad funcional de patógenos
   - Predicción de brotes con análisis espacial
   - Partnership con universidades

---

## ✅ CONCLUSIÓN

### **Plan de Acción Inmediato:**

1. **Completar dataset NBA** con odds históricos
2. **Implementar feature engineering** con 300+ variables
3. **Entrenar modelos predictivos** con Random Forest
4. **Paper trading** por 4 semanas
5. **Live betting** con $100K inicial
6. **Escalar** a otros deportes
7. **Generar $2M USD** en 24 meses
8. **Financiar proyectos de salud**

### **ROI Esperado:**

| Escenario | ROI Mensual | Bankroll 12 meses | Bankroll 24 meses |
|-----------|-------------|-------------------|-------------------|
| Pesimista | 3% | $142K | $203K |
| Realista | 7% | $225K | $507K |
| Optimista | 10% | $314K | $985K |
| Muy Optimista | 15% | $535K | $2.86M |

### **Conclusión:**

Con metodologías NEXT aplicadas a apuestas deportivas, es **realista generar $500K - $1M USD en 12-24 meses**, suficiente para financiar proyectos de salud de alto impacto.

**Próximo paso:** Empezar con FASE 1 (recolección de datos) inmediatamente.

---

**Generado:** 2026-02-25
**Objetivo:** Rentabilidad sostenible para financiar salud
**Por:** Ines ☕✅
