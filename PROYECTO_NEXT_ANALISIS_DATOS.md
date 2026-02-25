# PROYECTO NEXT - ANÁLISIS PROFUNDO DE METODOLOGÍAS DE DATOS

**Basado en:** Tesis Doctoral - "Evolutionary adaptation of the tepary bean"
**Autor:** Diego Felipe Conejo Rodríguez (2024)
**Total fuentes analizadas:** 159
**Enfoque:** Metodologías de datos transferibles

---

## 📊 RESUMEN EJECUTIVO

Este documento presenta un análisis profundo de **15 categorías de metodologías de datos** extraídas de la tesis doctoral, con énfasis en técnicas transferibles a múltiples disciplinas.

### **TOP 5 Metodologías por Menciones:**

1. **Feature Engineering & Extraction** - 617 menciones
2. **Spatial & Geographic Analysis** - 213 menciones
3. **Data Visualization** - 186 menciones
4. **Machine Learning - Supervised** - 105 menciones
5. **Diversity & Similarity Indices** - 55 menciones

---

## 🔬 METODOLOGÍA 1: FEATURE ENGINEERING & EXTRACTION (617 menciones)

### **Definición:**
Proceso de extraer, seleccionar y transformar variables (features/descriptores) de datos crudos para crear representaciones significativas que mejoren el rendimiento de modelos analíticos.

### **Marco Teórico:**

La tesis distingue tres tipos de descriptores:

#### **1.1 Classical Descriptors (Descriptores Clásicos)**
- **Definición:** Rasgos medidos manualmente con protocolos estandarizados
- **Características:**
  - Altamente heredables
  - Bajo costo de medición
  - Fácil replicabilidad
  - Limitados en número (20-50 típicamente)

- **Tipos:**
  - **Métros:** Longitud, ancho, peso (continuos)
  - **Escalares:** Intensidad de color, textura (ordinales)
  - **Estados:** Forma, tipo (categóricos)
  - **Presencia/Ausencia:** Rasgos binarios

- **Ejemplo de la tesis:**
  ```
  Descriptor: Seed color
  States: [white, cream, yellow, brown, black]
  Method: Visual assessment
  Scale: Categorical
  ```

#### **1.2 Digital Descriptors (Descriptores Digitales)**
- **Definición:** Rasgos extraídos automáticamente mediante análisis de imágenes
- **Características:**
  - Alta precisión y reproducibilidad
  - Alto throughput (miles de muestras)
  - Objetividad (sin sesgo humano)
  - Generan cientos de variables

- **Técnicas de extracción:**
  1. **Color features:**
     - Histogramas de color (RGB, HSV, LAB)
     - Momentos de color (media, desviación estándar, skewness)
     - Entropía de color

  2. **Texture features:**
     - GLCM (Gray-Level Co-occurrence Matrix)
     - Haralick features (contraste, correlación, energía, homogeneidad)
     - LBP (Local Binary Patterns)

  3. **Shape features:**
     - Área, perímetro, circularidad
     - Aspect ratio, eccentricity
     - Bounding box, convex hull
     - Momentos de Hu

  4. **Morphometric features:**
     - Landmarks (puntos de referencia anatómicos)
     - Procrustes analysis
     - Fourier descriptors

- **Pipeline de extracción:**
  ```
  1. Image acquisition (standardized lighting, background)
  2. Preprocessing (resize, normalize, denoise)
  3. Segmentation (separate object from background)
  4. Feature extraction (calculate metrics)
  5. Feature selection (remove redundant)
  6. Normalization (scale to 0-1)
  ```

#### **1.3 Phenomic Descriptors (Descriptores Fenómicos)**
- **Definición:** Rasgos fisiológicos y bioquímicos medidos con sensores
- **Características:**
  - Relacionados con función/rendimiento
  - Medición no destructiva
  - Alta complejidad técnica
  - Requieren equipamiento especializado

- **Técnicas:**
  1. **Espectroscopía:**
     - Reflectancia espectral (400-2500 nm)
     - Índices de vegetación (NDVI, EVI)
     - Contenido de pigmentos

  2. **Fluorescencia:**
     - Chlorophyll fluorescence (Fv/Fm)
     - Sun-induced fluorescence (SIF)
     - Eficiencia fotosintética

  3. **Termografía:**
     - Temperatura de canopia
     - Stress hídrico
     - Transpiración

  4. **Tomografía:**
     - Estructura 3D
     - Biomasa
     - Arquitectura radicular

### **Fuentes Bibliográficas Clave:**

#### **Fuente 1.1: IBPGR/IPGRI Descriptors (1985-2001)**
- **Título:** "Descriptors for Phaseolus"
- **Institución:** International Plant Genetic Resources Institute
- **Contribución:** Sistema estandarizado de descriptores clásicos
- **Metodología:**
  - Definición de rasgos cualitativos y cuantitativos
  - Escalas de medición estandarizadas
  - Protocolos de evaluación

- **Aplicación en NEXT:**
  ```
  Sistema de descriptores para cualquier dominio:

  class DescriptorSystem:
      def __init__(self):
          self.passport_descriptors = {}  # Origen, procedencia
          self.characterization_descriptors = {}  # Rasgos inherentes
          self.evaluation_descriptors = {}  # Rasgos ambientales

      def add_descriptor(self, category, name, states, method, scale):
          descriptor = {
              'name': name,
              'states': states,  # [value1, value2, ...] or None
              'method': method,  # 'measurement', 'visual', 'automated'
              'scale': scale,    # 'categorical', 'ordinal', 'continuous'
              'unit': None       # 'cm', 'kg', '%', etc.
          }

          if category == 'passport':
              self.passport_descriptors[name] = descriptor
          elif category == 'characterization':
              self.characterization_descriptors[name] = descriptor
          else:
              self.evaluation_descriptors[name] = descriptor
  ```

#### **Fuente 1.2: Araus et al. (2014)**
- **Título:** "Field high-throughput phenotyping: the new crop breeding frontier"
- **Journal:** Trends in Plant Science, 19(1), 52-61
- **DOI:** 10.1016/j.tplants.2013.09.008

- **Metodología:**
  - Fenotipado de campo con sensores proximales
  - Plataformas móviles y aéreas (drones)
  - Imágenes multiespectrales e hiperespectrales
  - Fluorescencia inducida por sol (SIF)

- **Técnicas específicas:**
  1. **Multispectral imaging:**
     - Bandas: RGB + NIR + Red Edge
     - Resolución: 1-10 cm/pixel
     - Cobertura: hectáreas por vuelo

  2. **Proximal sensing:**
     - Sensores de fluorescencia
     - Espectroradiómetros
     - Cámaras térmicas

  3. **Data fusion:**
     - Integración de múltiples sensores
     - Corrección atmosférica
     - Calibración radiométrica

- **Pipeline de datos:**
  ```python
  class PhenomicDataPipeline:
      def __init__(self):
          self.sensors = []
          self.calibration_data = None

      def acquire_data(self, samples):
          """Acquire phenomic data from samples"""
          data = {
              'spectral': self._acquire_spectral(samples),
              'fluorescence': self._acquire_fluorescence(samples),
              'thermal': self._acquire_thermal(samples)
          }
          return data

      def extract_features(self, raw_data):
          """Extract phenomic descriptors"""
          features = {}

          # Spectral indices
          features['NDVI'] = (raw_data['spectral']['NIR'] - raw_data['spectral']['Red']) / \
                             (raw_data['spectral']['NIR'] + raw_data['spectral']['Red'])

          features['EVI'] = 2.5 * ((raw_data['spectral']['NIR'] - raw_data['spectral']['Red']) /
                                   (raw_data['spectral']['NIR'] + 6*raw_data['spectral']['Red'] - \
                                    7.5*raw_data['spectral']['Blue'] + 1))

          # Fluorescence parameters
          features['Fv_Fm'] = (raw_data['fluorescence']['Fm'] - raw_data['fluorescence']['Fo']) / \
                              raw_data['fluorescence']['Fm']

          # Thermal stress
          features['CTD'] = raw_data['thermal']['canopy_temp'] - raw_data['thermal']['air_temp']

          return features
  ```

- **Transferibilidad:**
  | Disciplina | Aplicación |
  |------------|------------|
  | **Medicina** | Monitoreo de pacientes con sensores wearables |
  | **Manufactura** | Control de calidad con visión artificial |
  | **Construcción** | Monitoreo estructural con sensores |
  | **Transporte** | Inspección de infraestructura |
  | **Energía** | Monitoreo de paneles solares |

#### **Fuente 1.3: Singh et al. (2016)**
- **Título:** "Machine learning for high-throughput stress phenotyping in plants"
- **Journal:** Trends in Plant Science, 21(2), 110-124
- **DOI:** 10.1016/j.tplants.2015.10.015

- **Metodología:**
  - Pipeline de ML para feature extraction y selección
  - Automated trait discovery
  - Stress detection y clasificación

- **Feature Engineering Pipeline:**
  ```python
  class AutomatedFeatureEngineering:
      def __init__(self):
          self.extractors = {
              'color': self._extract_color_features,
              'texture': self._extract_texture_features,
              'shape': self._extract_shape_features,
              'spectral': self._extract_spectral_features
          }

      def extract_all_features(self, image):
          """Extract comprehensive feature set from image"""
          features = {}

          for feature_type, extractor in self.extractors.items():
              features.update(extractor(image))

          return features

      def _extract_color_features(self, image):
          """Extract color-based features"""
          import cv2
          import numpy as np

          features = {}

          # Convert to different color spaces
          hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
          lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

          # RGB features
          for i, channel in enumerate(['R', 'G', 'B']):
              features[f'{channel}_mean'] = image[:,:,i].mean()
              features[f'{channel}_std'] = image[:,:,i].std()
              features[f'{channel}_skew'] = self._skewness(image[:,:,i])

          # HSV features
          for i, channel in enumerate(['H', 'S', 'V']):
              features[f'{channel}_mean'] = hsv[:,:,i].mean()
              features[f'{channel}_std'] = hsv[:,:,i].std()

          # Color histograms
          hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
          features['color_entropy'] = self._entropy(hist_r)

          return features

      def _extract_texture_features(self, image):
          """Extract texture features using GLCM"""
          from skimage.feature import graycomatrix, graycoprops

          features = {}

          # Convert to grayscale
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

          # Calculate GLCM
          glcm = graycomatrix(gray, distances=[1, 3, 5], angles=[0, 45, 90, 135],
                             levels=256, symmetric=True, normed=True)

          # Haralick features
          for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
              features[f'texture_{prop}'] = graycoprops(glcm, prop).mean()

          return features

      def _extract_shape_features(self, image):
          """Extract shape features"""
          import cv2

          features = {}

          # Segment object
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
          contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          if contours:
              cnt = max(contours, key=cv2.contourArea)

              # Basic shape features
              features['area'] = cv2.contourArea(cnt)
              features['perimeter'] = cv2.arcLength(cnt, True)

              # Circularity
              features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)

              # Bounding rectangle
              x, y, w, h = cv2.boundingRect(cnt)
              features['aspect_ratio'] = float(w) / h
              features['extent'] = features['area'] / (w * h)

              # Convex hull
              hull = cv2.convexHull(cnt)
              features['solidity'] = features['area'] / cv2.contourArea(hull)

              # Moments
              M = cv2.moments(cnt)
              features['eccentricity'] = self._calculate_eccentricity(M)

              # Hu moments
              hu_moments = cv2.HuMoments(M)
              for i, hu in enumerate(hu_moments):
                  features[f'hu_moment_{i+1}'] = hu[0]

          return features

      def select_features(self, features, target, n_features=50):
          """Select most informative features"""
          from sklearn.ensemble import RandomForestClassifier
          from sklearn.feature_selection import SelectFromModel

          # Train Random Forest for feature importance
          rf = RandomForestClassifier(n_estimators=100, random_state=42)
          rf.fit(features, target)

          # Select top features
          selector = SelectFromModel(rf, max_features=n_features, prefit=True)
          selected_features = selector.transform(features)

          # Get selected feature names
          feature_names = features.columns
          selected_names = feature_names[selector.get_support()]

          return selected_features, selected_names, rf.feature_importances_
  ```

- **Feature Selection Techniques:**
  1. **Filter methods:**
     - Variance threshold
     - Correlation filter
     - Mutual information
     - Chi-square test

  2. **Wrapper methods:**
     - Recursive Feature Elimination (RFE)
     - Forward selection
     - Backward elimination

  3. **Embedded methods:**
     - LASSO (L1 regularization)
     - Ridge (L2 regularization)
     - Random Forest importance
     - Gradient Boosting importance

  ```python
  def feature_selection_pipeline(X, y, method='random_forest'):
      """Comprehensive feature selection pipeline"""

      from sklearn.feature_selection import (
          VarianceThreshold, SelectKBest, mutual_info_classif,
          RFE, SelectFromModel
      )
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.linear_model import LassoCV

      selected_features = {}

      # 1. Variance threshold (remove constant features)
      var_selector = VarianceThreshold(threshold=0.01)
      X_var = var_selector.fit_transform(X)
      selected_features['variance'] = X.columns[var_selector.get_support()]

      # 2. Correlation filter (remove highly correlated)
      corr_matrix = X.corr().abs()
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
      to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
      selected_features['correlation'] = [col for col in X.columns if col not in to_drop]

      # 3. Mutual information
      mi_selector = SelectKBest(mutual_info_classif, k=50)
      X_mi = mi_selector.fit_transform(X, y)
      selected_features['mutual_info'] = X.columns[mi_selector.get_support()]

      # 4. Random Forest importance
      rf = RandomForestClassifier(n_estimators=100, random_state=42)
      rf.fit(X, y)
      rf_selector = SelectFromModel(rf, max_features=50)
      X_rf = rf_selector.fit_transform(X, y)
      selected_features['random_forest'] = X.columns[rf_selector.get_support()]

      # 5. RFE with Random Forest
      rfe = RFE(estimator=RandomForestClassifier(n_estimators=50), n_features_to_select=30)
      rfe.fit(X, y)
      selected_features['rfe'] = X.columns[rfe.support_]

      return selected_features
  ```

- **Transferibilidad:**
  | Disciplina | Aplicación |
  |------------|------------|
  | **E-commerce** | Descriptores de productos (color, forma, textura) |
  | **Medicina** | Descriptores de imágenes médicas (radiomics) |
  | **Seguridad** | Descriptores faciales y de comportamiento |
  | **Calidad** | Detección de defectos en manufactura |
  | **Deportes** | Análisis biomecánico de movimientos |

---

## 🔬 METODOLOGÍA 2: SPATIAL & GEOGRAPHIC ANALYSIS (213 menciones)

### **Definición:**
Análisis de datos georreferenciados para identificar patrones espaciales, correlaciones ambientales y distribuciones geográficas.

### **Marco Teórico:**

#### **2.1 Geographic Data Types**

1. **Point data:**
   - Coordenadas de colecciones
   - Ubicaciones de muestreo
   - Eventos georreferenciados

2. **Raster data:**
   - Datos climáticos (WorldClim)
   - Imágenes satelitales
   - Modelos de elevación (DEM)

3. **Vector data:**
   - Polígonos de distribución
   - Líneas de conexión
   - Regiones administrativas

#### **2.2 Climate Data Sources**

**WorldClim (Fick & Hijmans, 2017):**
- **Variables:** 19 bioclim variables
- **Resolución:** 30 arc-seconds (~1 km²)
- **Cobertura:** Global

**Bioclimatic variables:**
```
BIO1  = Annual Mean Temperature
BIO2  = Mean Diurnal Range (Mean of monthly (max temp - min temp))
BIO3  = Isothermality (BIO2/BIO7) (* 100)
BIO4  = Temperature Seasonality (standard deviation *100)
BIO5  = Max Temperature of Warmest Month
BIO6  = Min Temperature of Coldest Month
BIO7  = Temperature Annual Range (BIO5-BIO6)
BIO8  = Mean Temperature of Wettest Quarter
BIO9  = Mean Temperature of Driest Quarter
BIO10 = Mean Temperature of Warmest Quarter
BIO11 = Mean Temperature of Coldest Quarter
BIO12 = Annual Precipitation
BIO13 = Precipitation of Wettest Month
BIO14 = Precipitation of Driest Month
BIO15 = Precipitation Seasonality (Coefficient of Variation)
BIO16 = Precipitation of Wettest Quarter
BIO17 = Precipitation of Driest Quarter
BIO18 = Precipitation of Warmest Quarter
BIO19 = Precipitation of Coldest Quarter
```

### **Fuentes Bibliográficas Clave:**

#### **Fuente 2.1: Fick & Hijmans (2017)**
- **Título:** "WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas"
- **Journal:** International Journal of Climatology, 37(12), 4302-4315
- **DOI:** 10.1002/joc.5086

- **Metodología:**
  - Interpolación de datos climáticos
  - Modelado espacial
  - Corrección topográfica

- **Implementación:**
  ```python
  import rasterio
  import numpy as np
  from rasterio.plot import show
  import matplotlib.pyplot as plt

  class ClimateDataAnalyzer:
      def __init__(self, bioclim_dir):
          self.bioclim_dir = bioclim_dir
          self.layers = {}

      def load_bioclim_layers(self):
          """Load all 19 bioclimatic variables"""
          for i in range(1, 20):
              file_path = f"{self.bioclim_dir}/wc2.1_30s_bio_{i}.tif"
              with rasterio.open(file_path) as src:
                  self.layers[f'BIO{i}'] = {
                      'data': src.read(1),
                      'transform': src.transform,
                      'crs': src.crs
                  }
          return self.layers

      def extract_climate_at_points(self, points):
          """Extract climate values at specific coordinates"""
          from rasterio.transform import rowcol

          climate_data = []

          for lat, lon in points:
              point_climate = {'lat': lat, 'lon': lon}

              for bio_name, layer_info in self.layers.items():
                  # Convert coordinates to row/col
                  row, col = rowcol(layer_info['transform'], lon, lat)

                  # Extract value
                  if 0 <= row < layer_info['data'].shape[0] and \
                     0 <= col < layer_info['data'].shape[1]:
                      value = layer_info['data'][row, col]
                      point_climate[bio_name] = value
                  else:
                      point_climate[bio_name] = np.nan

              climate_data.append(point_climate)

          return pd.DataFrame(climate_data)

      def calculate_climate_pca(self, climate_df):
          """Reduce climate dimensions using PCA"""
          from sklearn.decomposition import PCA
          from sklearn.preprocessing import StandardScaler

          # Select bioclim variables
          bio_cols = [col for col in climate_df.columns if col.startswith('BIO')]
          X = climate_df[bio_cols].dropna()

          # Normalize
          scaler = StandardScaler()
          X_scaled = scaler.fit_transform(X)

          # PCA
          pca = PCA(n_components=0.95)  # Retain 95% variance
          X_pca = pca.fit_transform(X_scaled)

          # Results
          results = {
              'pca_scores': X_pca,
              'explained_variance': pca.explained_variance_ratio_,
              'loadings': pd.DataFrame(
                  pca.components_.T,
                  index=bio_cols,
                  columns=[f'PC{i+1}' for i in range(pca.n_components_)]
              )
          }

          return results

      def climate_clustering(self, climate_df, n_clusters=5):
          """Cluster locations by climate similarity"""
          from sklearn.cluster import KMeans
          from sklearn.preprocessing import StandardScaler

          # Prepare data
          bio_cols = [col for col in climate_df.columns if col.startswith('BIO')]
          X = climate_df[bio_cols].dropna()

          # Normalize
          scaler = StandardScaler()
          X_scaled = scaler.fit_transform(X)

          # K-means clustering
          kmeans = KMeans(n_clusters=n_clusters, random_state=42)
          clusters = kmeans.fit_predict(X_scaled)

          # Add cluster labels
          climate_df['climate_cluster'] = np.nan
          climate_df.loc[X.index, 'climate_cluster'] = clusters

          # Calculate cluster centroids
          centroids = pd.DataFrame(
              scaler.inverse_transform(kmeans.cluster_centers_),
              columns=bio_cols
          )

          return climate_df, centroids
  ```

#### **Fuente 2.2: Ecogeographic Land Characterization**

- **Metodología:**
  - Clustering de variables ecogeográficas
  - Identificación de grupos ambientales
  - Análisis de representatividad

- **Pipeline completo:**
  ```python
  class EcogeographicAnalyzer:
      def __init__(self):
          self.climate_vars = None
          self.soil_vars = None
          self.topography_vars = None

      def load_all_environmental_data(self, points):
          """Load comprehensive environmental data for points"""

          environmental_data = pd.DataFrame()

          # 1. Climate data (WorldClim)
          climate_analyzer = ClimateDataAnalyzer('/path/to/worldclim')
          climate_df = climate_analyzer.extract_climate_at_points(points)
          environmental_data = pd.concat([environmental_data, climate_df], axis=1)

          # 2. Soil data (SoilGrids)
          soil_data = self._extract_soil_data(points)
          environmental_data = pd.concat([environmental_data, soil_data], axis=1)

          # 3. Topography (SRTM DEM)
          topo_data = self._extract_topography(points)
          environmental_data = pd.concat([environmental_data, topo_data], axis=1)

          return environmental_data

      def _extract_soil_data(self, points):
          """Extract soil properties from SoilGrids"""
          import requests

          soil_data = []

          for lat, lon in points:
              # Query SoilGrids API
              url = f"https://rest.soilgrids.org/soilgrids/rest/v1.0/properties/query?lon={lon}&lat={lat}"

              try:
                  response = requests.get(url)
                  if response.status_code == 200:
                      data = response.json()

                      # Extract soil properties
                      soil_props = {
                          'lat': lat,
                          'lon': lon,
                          'clay_content': data['properties']['layers'][0]['depths'][0]['values']['mean'],
                          'sand_content': data['properties']['layers'][1]['depths'][0]['values']['mean'],
                          'silt_content': data['properties']['layers'][2]['depths'][0]['values']['mean'],
                          'organic_carbon': data['properties']['layers'][3]['depths'][0]['values']['mean'],
                          'pH': data['properties']['layers'][4]['depths'][0]['values']['mean']
                      }
                      soil_data.append(soil_props)
              except:
                  pass

          return pd.DataFrame(soil_data)

      def _extract_topography(self, points):
          """Extract topographic variables from DEM"""
          import elevation

          topo_data = []

          for lat, lon in points:
              # Calculate topographic metrics
              topo = {
                  'lat': lat,
                  'lon': lon,
                  'elevation': self._get_elevation(lat, lon),
                  'slope': self._calculate_slope(lat, lon),
                  'aspect': self._calculate_aspect(lat, lon)
              }
              topo_data.append(topo)

          return pd.DataFrame(topo_data)

      def dimensionality_reduction(self, env_data, method='pca'):
          """Reduce environmental dimensions"""

          from sklearn.decomposition import PCA
          from sklearn.preprocessing import StandardScaler
          import umap

          # Select environmental variables
          env_cols = [col for col in env_data.columns if col not in ['lat', 'lon']]
          X = env_data[env_cols].dropna()

          # Normalize
          scaler = StandardScaler()
          X_scaled = scaler.fit_transform(X)

          if method == 'pca':
              reducer = PCA(n_components=0.95)
          elif method == 'umap':
              reducer = umap.UMAP(n_components=3, random_state=42)

          # Reduce dimensions
          X_reduced = reducer.fit_transform(X_scaled)

          # Add reduced dimensions to dataframe
          env_data['ECO1'] = np.nan
          env_data['ECO2'] = np.nan
          env_data['ECO3'] = np.nan

          env_data.loc[X.index, 'ECO1'] = X_reduced[:, 0]
          env_data.loc[X.index, 'ECO2'] = X_reduced[:, 1]
          if X_reduced.shape[1] > 2:
              env_data.loc[X.index, 'ECO3'] = X_reduced[:, 2]

          return env_data, reducer

      def ecogeographic_clustering(self, env_data, n_clusters=10):
          """Cluster locations by ecogeographic similarity"""

          from sklearn.cluster import AgglomerativeClustering
          from scipy.spatial.distance import pdist, squareform

          # Use reduced dimensions
          X = env_data[['ECO1', 'ECO2', 'ECO3']].dropna()

          # Hierarchical clustering
          clustering = AgglomerativeClustering(
              n_clusters=n_clusters,
              affinity='euclidean',
              linkage='ward'
          )

          clusters = clustering.fit_predict(X)

          # Add cluster labels
          env_data['ecogeo_group'] = np.nan
          env_data.loc[X.index, 'ecogeo_group'] = clusters

          return env_data

      def calculate_representativeness(self, env_data, collection_points):
          """Calculate how well collection represents environmental diversity"""

          from scipy.spatial.distance import cdist

          # Get environmental space
          env_space = env_data[['ECO1', 'ECO2', 'ECO3']].dropna()

          # Calculate environmental coverage
          # Distance from each point to nearest collection point
          collection_coords = env_space.loc[collection_points]
          all_coords = env_space

          distances = cdist(all_coords, collection_coords, metric='euclidean')
          min_distances = distances.min(axis=1)

          # Representativeness metrics
          representativeness = {
              'mean_distance': min_distances.mean(),
              'max_distance': min_distances.max(),
              'coverage_%': (min_distances < min_distances.mean()).sum() / len(min_distances) * 100
          }

          return representativeness
  ```

- **Transferibilidad:**
  | Disciplina | Aplicación |
  |------------|------------|
  | **Retail** | Análisis de ubicaciones óptimas de tiendas |
  | **Bienes Raíces** | Predicción de valor por ubicación |
  | **Salud** | Mapeo de enfermedades por región |
  | **Turismo** | Identificación de destinos óptimos |
  | **Logística** | Optimización de rutas y centros de distribución |

---

## 🔬 METODOLOGÍA 3: MACHINE LEARNING - SUPERVISED (105 menciones)

### **Definición:**
Técnicas de aprendizaje automático donde el modelo aprende de datos etiquetados para hacer predicciones sobre nuevos datos.

### **Algoritmos Identificados:**

#### **3.1 Random Forest (26 menciones)**

**Ventajas:**
- Robusto a overfitting
- Maneja datos no lineales
- Proporciona importancia de variables
- Funciona con datos faltantes

**Implementación:**
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np

class RandomForestAnalyzer:
    def __init__(self, task='classification'):
        self.task = task
        self.model = None
        self.best_params = None

    def train_with_tuning(self, X, y):
        """Train Random Forest with hyperparameter tuning"""

        # Define parameter grid
        if self.task == 'classification':
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            rf = RandomForestClassifier(random_state=42)
        else:
            rf = RandomForestRegressor(random_state=42)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy' if self.task == 'classification' else 'r2',
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X, y)

        # Save best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        return self.model

    def get_feature_importance(self, feature_names):
        """Extract and visualize feature importance"""

        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def predict_with_uncertainty(self, X):
        """Make predictions with uncertainty estimates"""

        # Get predictions from all trees
        if self.task == 'classification':
            predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            # Calculate uncertainty as disagreement between trees
            uncertainty = 1 - (predictions == predictions[0]).mean(axis=0)
            final_prediction = self.model.predict(X)
        else:
            predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            uncertainty = predictions.std(axis=0)
            final_prediction = predictions.mean(axis=0)

        return final_prediction, uncertainty

    def cross_validate(self, X, y, cv=10):
        """Perform cross-validation"""

        scores = cross_val_score(
            self.model,
            X, y,
            cv=cv,
            scoring='accuracy' if self.task == 'classification' else 'r2'
        )

        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }
```

#### **3.2 Support Vector Machines (SVM)**

**Implementación:**
```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SVMAnalyzer:
    def __init__(self, kernel='rbf', task='classification'):
        self.kernel = kernel
        self.task = task
        self.scaler = StandardScaler()
        self.model = None

    def train(self, X, y):
        """Train SVM with proper scaling"""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features (important for SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if self.task == 'classification':
            self.model = SVC(kernel=self.kernel, random_state=42)
        else:
            self.model = SVR(kernel=self.kernel)

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        return {
            'train_score': train_score,
            'test_score': test_score,
            'model': self.model
        }

    def predict(self, X):
        """Make predictions on new data"""

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
```

#### **3.3 Neural Networks**

**Implementación:**
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

class NeuralNetworkAnalyzer:
    def __init__(self, input_dim, task='classification'):
        self.input_dim = input_dim
        self.task = task
        self.model = None
        self.history = None

    def build_model(self, hidden_layers=[64, 32], dropout=0.3):
        """Build neural network architecture"""

        model = models.Sequential()

        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu',
                              input_dim=self.input_dim))
        model.add(layers.Dropout(dropout))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout))

        # Output layer
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(layers.Dense(1, activation='linear'))
            model.compile(optimizer='adam',
                         loss='mse',
                         metrics=['mae'])

        self.model = model
        return model

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """Train neural network"""

        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        return self.history

    def predict(self, X):
        """Make predictions"""

        predictions = self.model.predict(X)

        if self.task == 'classification':
            return (predictions > 0.5).astype(int)
        else:
            return predictions

    def plot_training_history(self):
        """Plot training and validation metrics"""

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Metrics
        if self.task == 'classification':
            axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
            axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].set_ylabel('Accuracy')
        else:
            axes[1].plot(self.history.history['mae'], label='Training MAE')
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
            axes[1].set_title('Model MAE')
            axes[1].set_ylabel('MAE')

        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        plt.tight_layout()
        return fig
```

### **Pipeline Integrado de ML:**

```python
class IntegratedMLPipeline:
    def __init__(self, task='classification'):
        self.task = task
        self.models = {}
        self.results = {}

    def train_all_models(self, X, y):
        """Train multiple models and compare performance"""

        from sklearn.model_selection import cross_val_score

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42) if self.task == 'classification'
                            else RandomForestRegressor(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42) if self.task == 'classification'
                   else SVR(kernel='rbf'),
            'Neural Network': self._build_nn(X.shape[1])
        }

        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")

            if name == 'Neural Network':
                # Special handling for NN
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )

                train_score = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
                test_score = model.evaluate(X_test_scaled, y_test, verbose=0)[1]

            else:
                # Train traditional ML models
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)

            self.models[name] = model
            self.results[name] = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"  Train Score: {train_score:.4f}")
            print(f"  Test Score: {test_score:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.results

    def _build_nn(self, input_dim):
        """Build simple neural network"""

        model = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid' if self.task == 'classification' else 'linear')
        ])

        if self.task == 'classification':
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

    def get_best_model(self):
        """Select best model based on test score"""

        best_model_name = max(self.results, key=lambda x: self.results[x]['test_score'])
        return best_model_name, self.models[best_model_name], self.results[best_model_name]
```

---

## 🔬 METODOLOGÍA 4: DIVERSITY & SIMILARITY INDICES (55 menciones)

### **Definición:**
Métricas para cuantificar la diversidad, similitud y distancia entre elementos en un sistema.

### **Índices Principales:**

#### **4.1 Functional Diversity Indices**

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull

class FunctionalDiversityCalculator:
    def __init__(self, traits_matrix, abundances=None):
        """
        Initialize functional diversity calculator

        Parameters:
        -----------
        traits_matrix : np.array
            Matrix of traits (n_samples x n_traits)
        abundances : np.array, optional
            Relative abundances of each sample
        """
        self.traits = traits_matrix
        self.abundances = abundances if abundances is not None else np.ones(len(traits_matrix)) / len(traits_matrix)

    def functional_richness(self):
        """
        Calculate Functional Richness (FRic)
        Volume of convex hull in trait space
        """

        try:
            hull = ConvexHull(self.traits)
            fric = hull.volume
        except:
            fric = 0  # Not enough points for convex hull

        return fric

    def functional_evenness(self):
        """
        Calculate Functional Evenness (FEve)
        Regularity of trait distribution
        """

        # Calculate pairwise distances
        distances = squareform(pdist(self.traits, metric='euclidean'))

        # Normalize distances
        max_dist = distances.max()
        if max_dist > 0:
            distances_norm = distances / max_dist
        else:
            return 1.0

        # Calculate weighted distances
        weighted_dist = []
        for i in range(len(self.traits)):
            for j in range(i+1, len(self.traits)):
                weighted_dist.append(self.abundances[i] * self.abundances[j] * distances_norm[i, j])

        # FEve formula
        EW = np.array(weighted_dist) / np.sum(weighted_dist) if np.sum(weighted_dist) > 0 else np.array(weighted_dist)

        S = len(self.traits)
        FEve = (np.sum(np.minimum(EW, 1/(S-1))) - 1/(S-1)) / (1 - 1/(S-1))

        return max(0, FEve)

    def functional_divergence(self):
        """
        Calculate Functional Divergence (FDiv)
        Distribution of abundances in trait space
        """

        # Calculate centroid
        centroid = np.average(self.traits, axis=0, weights=self.abundances)

        # Distances to centroid
        distances = np.linalg.norm(self.traits - centroid, axis=1)

        # Weighted mean distance
        d_mean = np.average(distances, weights=self.abundances)

        # Grand mean distance
        d_grand = np.mean(distances)

        # FDiv formula
        if d_mean + d_grand > 0:
            FDiv = (d_mean - d_grand) / (d_mean + d_grand)
        else:
            FDiv = 0

        return FDiv

    def functional_dispersion(self):
        """
        Calculate Functional Dispersion (FDis)
        Weighted mean distance to centroid
        """

        # Calculate weighted centroid
        centroid = np.average(self.traits, axis=0, weights=self.abundances)

        # Weighted distances to centroid
        distances = np.linalg.norm(self.traits - centroid, axis=1)
        FDis = np.average(distances, weights=self.abundances)

        return FDis

    def calculate_all_indices(self):
        """Calculate all functional diversity indices"""

        return {
            'FRic': self.functional_richness(),
            'FEve': self.functional_evenness(),
            'FDiv': self.functional_divergence(),
            'FDis': self.functional_dispersion()
        }
```

#### **4.2 Similarity Indices**

```python
class SimilarityIndices:
    def __init__(self):
        pass

    @staticmethod
    def jaccard_index(set1, set2):
        """
        Jaccard similarity index
        J(A,B) = |A ∩ B| / |A ∪ B|
        """

        intersection = len(set(set1) & set(set2))
        union = len(set(set1) | set(set2))

        return intersection / union if union > 0 else 0

    @staticmethod
    def sorensen_index(set1, set2):
        """
        Sørensen-Dice similarity index
        QS = 2|A ∩ B| / (|A| + |B|)
        """

        intersection = len(set(set1) & set(set2))
        sum_sizes = len(set1) + len(set2)

        return 2 * intersection / sum_sizes if sum_sizes > 0 else 0

    @staticmethod
    def bray_curtis_dissimilarity(abundance1, abundance2):
        """
        Bray-Curtis dissimilarity
        BC = Σ|xi - yi| / Σ(xi + yi)
        """

        abundance1 = np.array(abundance1)
        abundance2 = np.array(abundance2)

        numerator = np.sum(np.abs(abundance1 - abundance2))
        denominator = np.sum(abundance1 + abundance2)

        return numerator / denominator if denominator > 0 else 0

    @staticmethod
    def euclidean_distance(point1, point2):
        """Euclidean distance"""

        return np.linalg.norm(np.array(point1) - np.array(point2))

    @staticmethod
    def manhattan_distance(point1, point2):
        """Manhattan distance"""

        return np.sum(np.abs(np.array(point1) - np.array(point2)))

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Cosine similarity"""

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        return dot_product / norms if norms > 0 else 0
```

#### **4.3 Diversity Indices**

```python
class DiversityIndices:
    def __init__(self, abundances):
        """
        Initialize diversity calculator

        Parameters:
        -----------
        abundances : array-like
            Abundances of each species/category
        """
        self.abundances = np.array(abundances)
        self.proportions = self.abundances / self.abundances.sum()

    def shannon_index(self):
        """
        Shannon diversity index
        H' = -Σ(pi * ln(pi))
        """

        # Remove zero proportions
        props = self.proportions[self.proportions > 0]

        return -np.sum(props * np.log(props))

    def simpson_index(self):
        """
        Simpson diversity index
        D = Σ(pi²)
        Returns 1-D (higher = more diverse)
        """

        return 1 - np.sum(self.proportions ** 2)

    def fisher_alpha(self):
        """
        Fisher's alpha diversity index
        Requires iterative solution
        """

        S = np.sum(self.abundances > 0)  # Number of species
        N = self.abundances.sum()  # Total abundance

        # Solve iteratively
        alpha = 1.0
        for _ in range(100):
            alpha_new = (N - S) / np.log(1 + alpha)
            if abs(alpha_new - alpha) < 1e-6:
                break
            alpha = alpha_new

        return alpha

    def species_richness(self):
        """Number of species"""

        return np.sum(self.abundances > 0)

    def pielou_evenness(self):
        """
        Pielou's evenness index
        J = H' / ln(S)
        """

        H = self.shannon_index()
        S = self.species_richness()

        return H / np.log(S) if S > 1 else 0

    def calculate_all_indices(self):
        """Calculate all diversity indices"""

        return {
            'shannon': self.shannon_index(),
            'simpson': self.simpson_index(),
            'fisher_alpha': self.fisher_alpha(),
            'richness': self.species_richness(),
            'evenness': self.pielou_evenness()
        }
```

---

## 🔬 METODOLOGÍA 5: DIMENSIONALITY REDUCTION (47 menciones)

### **Técnicas Principales:**

#### **5.1 Principal Component Analysis (PCA)**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class PCAAnalyzer:
    def __init__(self, n_components=None, variance_threshold=0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.loadings = None

    def fit_transform(self, X):
        """Fit PCA and transform data"""

        # Normalize data
        X_scaled = self.scaler.fit_transform(X)

        # Determine number of components
        if self.n_components is None:
            # Start with all components
            pca_full = PCA()
            pca_full.fit(X_scaled)

            # Find number of components for variance threshold
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
        else:
            n_components = self.n_components

        # Fit final PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        # Calculate loadings
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=X.columns,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        return X_pca

    def get_explained_variance(self):
        """Get explained variance by component"""

        return pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(self.pca.explained_variance_ratio_))],
            'variance': self.pca.explained_variance_ratio_,
            'cumulative': np.cumsum(self.pca.explained_variance_ratio_)
        })

    def plot_variance_explained(self):
        """Plot scree plot and cumulative variance"""

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Scree plot
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_) + 1),
                   self.pca.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Scree Plot')

        # Cumulative variance
        axes[1].plot(range(1, len(self.pca.explained_variance_ratio_) + 1),
                    np.cumsum(self.pca.explained_variance_ratio_), 'bo-')
        axes[1].axhline(y=self.variance_threshold, color='r', linestyle='--',
                       label=f'{self.variance_threshold*100}% Variance')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance Explained')
        axes[1].legend()

        plt.tight_layout()
        return fig

    def plot_biplot(self, X_pca, labels=None, pc_x=1, pc_y=2):
        """Create biplot showing samples and loadings"""

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot samples
        scatter = ax.scatter(X_pca[:, pc_x-1], X_pca[:, pc_y-1],
                           c=labels if labels is not None else 'blue',
                           alpha=0.6)

        # Plot loadings (feature vectors)
        scale = 2  # Scale factor for arrows
        for i, (feature, loading) in enumerate(self.loadings.iterrows()):
            ax.arrow(0, 0,
                    loading[f'PC{pc_x}'] * scale,
                    loading[f'PC{pc_y}'] * scale,
                    color='red', alpha=0.5)
            ax.text(loading[f'PC{pc_x}'] * scale * 1.1,
                   loading[f'PC{pc_y}'] * scale * 1.1,
                   feature, color='red', ha='center', va='center')

        ax.set_xlabel(f'PC{pc_x} ({self.pca.explained_variance_ratio_[pc_x-1]:.1%} variance)')
        ax.set_ylabel(f'PC{pc_y} ({self.pca.explained_variance_ratio_[pc_y-1]:.1%} variance)')
        ax.set_title('PCA Biplot')
        ax.grid(True, alpha=0.3)

        return fig

    def get_top_features_per_component(self, n_features=10):
        """Get most important features for each PC"""

        top_features = {}

        for pc in self.loadings.columns:
            # Sort by absolute loading
            sorted_loadings = self.loadings[pc].abs().sort_values(ascending=False)
            top_features[pc] = sorted_loadings.head(n_features).index.tolist()

        return top_features
```

#### **5.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)**

```python
from sklearn.manifold import TSNE

class TSNEAnalyzer:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.tsne = None

    def fit_transform(self, X):
        """Fit t-SNE and transform data"""

        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            random_state=42
        )

        X_tsne = self.tsne.fit_transform(X)

        return X_tsne

    def plot_embedding(self, X_tsne, labels=None):
        """Plot t-SNE embedding"""

        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                               c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Label')
        else:
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)

        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE Embedding')
        ax.grid(True, alpha=0.3)

        return fig
```

#### **5.3 UMAP (Uniform Manifold Approximation and Projection)**

```python
import umap

class UMAPAnalyzer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_model = None

    def fit_transform(self, X):
        """Fit UMAP and transform data"""

        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=42
        )

        X_umap = self.umap_model.fit_transform(X)

        return X_umap

    def plot_embedding(self, X_umap, labels=None):
        """Plot UMAP embedding"""

        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                               c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Label')
        else:
            ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Embedding')
        ax.grid(True, alpha=0.3)

        return fig
```

---

## 📊 RESUMEN DE HERRAMIENTAS IDENTIFICADAS

### **Lenguajes de Programación:**
- **R** (62 menciones) - Principal herramienta estadística
- **Python** - Machine Learning y automatización

### **Paquetes de R Identificados:**
- **sp** (579 menciones) - Análisis espacial
- **cluster** (86 menciones) - Clustering
- **sf** (19 menciones) - Simple features
- **FD** (16 menciones) - Functional diversity
- **leaflet** (13 menciones) - Mapas interactivos
- **randomForest** (6 menciones) - Random Forest
- **mFD** (4 menciones) - Multidimensional FD
- **caret** (2 menciones) - Machine Learning
- **factoextra** (2 menciones) - Visualización PCA
- **nnet** (1 menciones) - Neural networks

### **Métodos Estadísticos:**
- **PCA** (14 menciones)
- **ANOVA** (12 menciones)
- **Cluster Analysis** (10 menciones)
- **MANOVA** (9 menciones)
- **Correlation** (2 menciones)
- **Discriminant Analysis** (2 menciones)
- **Logistic Regression** (1 menciones)

---

## 🎯 APLICACIONES DEL PROYECTO NEXT

### **Aplicación 1: Sistema de Descriptores Digitales**
Crear sistema estandarizado de caracterización para cualquier dominio.

### **Aplicación 2: Pipeline de Machine Learning**
Framework completo de ML con feature engineering automático.

### **Aplicación 3: Análisis de Diversidad Funcional**
Calcular índices de diversidad para sistemas multidimensionales.

### **Aplicación 4: Reducción de Dimensionalidad**
Implementar PCA, t-SNE, UMAP para visualización de datos complejos.

### **Aplicación 5: Análisis Espacial**
Integrar datos geográficos y ambientales en análisis.

---

**Generado:** 2026-02-25
**Proyecto:** NEXT
**Por:** Ines ☕✅
