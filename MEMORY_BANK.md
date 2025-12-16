# Memory Bank - PyGeo Project

## Project Overview
**Project Name:** PyGeo - Geological Data Visualization System  
**Purpose:** Visualization and analysis of well data for geological exploration  
**Technology Stack:** Python, Streamlit, Plotly, NumPy, Pandas, LASio  
**Target Users:** Geologists, petroleum engineers, data analysts

---

## Project Structure

```
pygeo/
├── app.py                          # Main Streamlit application
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── backend/                        # Backend modules (placeholder)
│   └── __init__.py
├── frontend/                       # Frontend modules
│   ├── __init__.py
│   └── modules/
│       ├── __init__.py
│       ├── data_loader.py         # Data loading utilities
│       ├── preprocess.py          # Data preprocessing functions
│       └── visualizer.py          # Visualization functions
└── src_data/                      # Source data directory
    ├── WELL_*.las                 # LAS files (67 wells)
    ├── dot_dtv/
    │   ├── H                      # Formation thickness data
    │   └── EFF_H                  # Effective thickness data
    └── INKL/
        └── траектории             # Well trajectories file
```

---

## Core Modules

### 1. [`app.py`](app.py:1) - Main Application
**Purpose:** Streamlit web application entry point

**Key Features:**
- Multi-mode visualization interface
- Session state management
- Data loading orchestration
- Interactive controls and filters

**View Modes:**
1. **Карта (Map)** - 2D well location map with collector ratio
2. **3D траектории (3D Trajectories)** - 3D well path visualization
3. **3D пласты коллекторов (3D Reservoir Layers)** - 3D reservoir visualization with well logs
4. **2D проекция скважины (2D Well Projection)** - 2D well cross-section with layers
5. **Разрезы (Cross-sections)** - Geophysical well logs
6. **Анализ (Analysis)** - Statistical analysis and data export

**Session State Variables:**
- `data_loaded`: Boolean flag for data loading status
- `trajectories`: Dictionary of well trajectories
- `well_data`: Combined DataFrame with H and EFF_H data
- `las_data`: Dictionary of LAS file data

**Key Functions:**
- Data loading button handler (lines 56-74)
- View mode switching (lines 42-46)
- Filter controls (lines 80-89)

---

### 2. [`frontend/modules/data_loader.py`](frontend/modules/data_loader.py:1) - Data Loading
**Purpose:** Load and parse various geological data formats

**Key Functions:**

#### [`load_welltrajectories(filepath)`](frontend/modules/data_loader.py:12)
- **Input:** Path to trajectory file
- **Output:** Dict[well_name, np.ndarray[X, Y, Z, MD]]
- **Format:** Parses custom trajectory format with `welltrack 'WELL_XXX'` headers
- **Data Structure:** Each trajectory is [X, Y, Z, MD] where MD = Measured Depth

#### [`load_h_data(filepath)`](frontend/modules/data_loader.py:63)
- **Input:** Path to H (thickness) data file
- **Output:** DataFrame with columns [X, Y, Z, Well, H]
- **Format:** Space-separated values with # comments

#### [`load_eff_h_data(filepath)`](frontend/modules/data_loader.py:80)
- **Input:** Path to EFF_H (effective thickness) data file
- **Output:** DataFrame with columns [X, Y, Z, Well, EFF_H]
- **Purpose:** Loads effective thickness (collector zones only)

#### [`load_las_file(filepath)`](frontend/modules/data_loader.py:97)
- **Input:** Path to LAS file
- **Output:** Dict with keys: `well_name`, `depth`, `curve`, `null_value`
- **Library:** Uses `lasio` library
- **Curve Name:** Looks for 'КриваяГИС1' or uses first curve
- **Null Value:** -999.25 (standard missing data marker)

#### [`load_all_las_files(folder_path)`](frontend/modules/data_loader.py:125)
- **Input:** Folder path containing LAS files
- **Output:** Dict[well_name, las_data]
- **Supported Extensions:** .las, .txt

#### [`combine_all_data(h_path, eff_h_path)`](frontend/modules/data_loader.py:151)
- **Input:** Paths to H and EFF_H files
- **Output:** Merged DataFrame with calculated collector ratio
- **Calculated Field:** `Доля_коллектора = EFF_H / H` (collector ratio)
- **Data Cleaning:** Removes NaN values, handles division by zero

---

### 3. [`frontend/modules/preprocess.py`](frontend/modules/preprocess.py:1) - Data Preprocessing
**Purpose:** Data cleaning, interpolation, and preparation for ML

**Key Functions:**

#### [`clean_las_data(las_data)`](frontend/modules/preprocess.py:10)
- **Purpose:** Remove null values (-999.25) from LAS data
- **Returns:** Cleaned dictionary with valid data only

#### [`interpolate_trajectory(trajectory, step=1.0)`](frontend/modules/preprocess.py:34)
- **Purpose:** Interpolate well trajectory with uniform step
- **Input:** Trajectory array [X, Y, Z, MD]
- **Output:** Interpolated trajectory with specified MD step
- **Method:** Linear interpolation using `np.interp`

#### [`create_grid_from_points(df, grid_size=100)`](frontend/modules/preprocess.py:64)
- **Purpose:** Create regular grid for spatial interpolation
- **Returns:** X_grid, Y_grid meshgrid arrays
- **Padding:** Adds 10% padding around data bounds

#### [`prepare_ml_data(df, las_dict)`](frontend/modules/preprocess.py:99)
- **Purpose:** Prepare data structure for ML model integration
- **Output:** Dictionary with coordinates, labels, well names, and LAS data
- **Use Case:** Ready for ML model from 1st year team

#### [`filter_by_depth(las_data, min_depth, max_depth)`](frontend/modules/preprocess.py:129)
- **Purpose:** Filter LAS data by depth range
- **Returns:** Filtered LAS data dictionary

---

### 4. [`frontend/modules/visualizer.py`](frontend/modules/visualizer.py:1) - Visualization
**Purpose:** Create interactive Plotly visualizations

**Key Functions:**

#### [`create_2d_map(df, show_well_names=True)`](frontend/modules/visualizer.py:12)
- **Purpose:** 2D map view of well locations
- **Color Scale:** Viridis colormap based on collector ratio
- **Features:** Well names, hover info, equal aspect ratio
- **Returns:** Plotly Figure

#### [`create_3d_trajectories(trajectories)`](frontend/modules/visualizer.py:61)
- **Purpose:** 3D visualization of well paths
- **Features:** 
  - Colored lines for each well
  - Start/end markers
  - Hover information with depth
- **Aspect Ratio:** x=1, y=1, z=0.7

#### [`create_las_cross_section(las_data, well_name)`](frontend/modules/visualizer.py:131)
- **Purpose:** Display well log cross-section
- **Color Coding:**
  - Yellow: Collector (value = 1)
  - Gray: Non-collector (value = 0)
  - Light blue: Other values
- **Y-axis:** Reversed (depth increases downward)

#### [`create_prediction_heatmap(X_grid, Y_grid, Z_pred)`](frontend/modules/visualizer.py:200)
- **Purpose:** Display ML prediction results as heatmap
- **Color Scale:** RdBu_r (red-blue reversed)
- **Use Case:** Placeholder for ML model integration

#### [`create_well_comparison(df)`](frontend/modules/visualizer.py:230)
- **Purpose:** Comparative bar chart of well characteristics
- **Displays:**
  - H (formation thickness) - light blue bars
  - EFF_H (effective thickness) - orange bars
  - Collector ratio - red line on secondary axis

#### [`create_3d_reservoir_layers(well_data, trajectories, las_data, ...)`](frontend/modules/visualizer.py:286)
- **Purpose:** 3D visualization combining well trajectories with collector layers
- **Approach:** Uses trajectory-based rendering (like create_3d_trajectories) with layer visualization (like create_2d_well_projection)
- **Features:**
  - Well trajectories with colored lines
  - Collector/non-collector layers rendered along trajectories
  - Accurate MD to X,Y,Z mapping using trajectory interpolation
  - Start/end markers for each well
  - Legend for layer types
- **Layer Rendering:**
  - Green thick lines (width=10): Collector intervals (value=1)
  - Gray lines (width=8): Non-collector intervals (value=0)
  - Layers follow actual well trajectory in 3D space
- **MD Mapping:**
  - Uses np.interp to map LAS MD values to trajectory X,Y,Z coordinates
  - Handles both vertical and deviated wells correctly
  - Validates MD range compatibility between LAS and trajectory data
- **Aspect Ratio:** x=1, y=1, z=0.7 (same as 3D trajectories view)

#### [`create_2d_well_projection(well_data, las_data, selected_well)`](frontend/modules/visualizer.py:556)
- **Purpose:** 2D vertical cross-section of single well
- **Features:**
  - Layer visualization (green=collector, gray=non-collector)
  - Red wiggle trace for log curve
  - Black vertical line for wellbore
  - Top/bottom markers (blue/red triangles)
  - Information annotations
- **Scaling:** Normalized depth to well thickness range

---

## Data Formats

### 1. Well Trajectories File (`src_data/INKL/траектории`)
```
welltrack 'WELL_034'
7131.91 75939.70 71.62 0.00
7131.88 75939.60 61.62 10.00
...
```
- **Format:** Custom text format
- **Fields:** X, Y, Z, MD (Measured Depth)
- **Encoding:** UTF-8 (supports Cyrillic)

### 2. H Data File (`src_data/dot_dtv/H`)
```
# Comment line
X Y Z Well H
6681.46 74209.62 1086.29 WELL_067 8.62
```
- **Format:** Space-separated values
- **Fields:** X, Y, Z (coordinates), Well (name), H (thickness in meters)

### 3. EFF_H Data File (`src_data/dot_dtv/EFF_H`)
```
# Comment line
X Y Z Well EFF_H
6681.46 74209.62 1086.29 WELL_067 5.23
```
- **Format:** Space-separated values
- **Fields:** X, Y, Z (coordinates), Well (name), EFF_H (effective thickness)

### 4. LAS Files (`src_data/WELL_*.las`)
- **Format:** Standard LAS 2.0 format
- **Key Curve:** КриваяГИС1 (Geophysical log curve)
- **Values:**
  - 1 = Effective collector
  - 0 = Non-collector or ineffective collector
  - -999.25 = Missing data
- **Index:** DEPT (depth)

---

## Key Concepts

### Geological Terms
- **H (Мощность пласта):** Total formation thickness in meters
- **EFF_H (Эффективная мощность):** Effective thickness (collector zones only)
- **Доля коллектора (Collector Ratio):** EFF_H / H (percentage of formation that is collector)
- **Коллектор (Collector):** Reservoir rock with good porosity/permeability
- **Неколлектор (Non-collector):** Rock with poor reservoir properties
- **Кровля (Top):** Top boundary of formation
- **Подошва (Bottom):** Bottom boundary of formation
- **ГИС (Geophysical Investigation of Wells):** Well logging data

### Coordinate System
- **X, Y:** Horizontal coordinates (meters)
- **Z:** Vertical depth (meters, positive downward)
- **MD:** Measured Depth along wellbore (meters)

---

## Dependencies (requirements.txt)
```
streamlit          # Web application framework
plotly             # Interactive plotting
pandas             # Data manipulation
numpy              # Numerical computing
lasio              # LAS file reading
scipy              # Scientific computing (interpolation)
```

---

## Current State & Known Issues

### Implemented Features ✅
- Multi-view visualization system
- LAS file loading and parsing
- 3D trajectory visualization
- 3D reservoir layer visualization along trajectories
- Accurate MD to Z coordinate mapping
- Well log display with color coding
- Statistical analysis tools
- Data export functionality
- Interactive filtering

### Placeholders / TODO 🚧
- ML model integration (line 169-193 in app.py)
- Backend module implementation
- Advanced interpolation methods (Kriging, IDW, RBF)
- Performance optimization for large datasets

### Technical Notes
- **Browser Compatibility:** Requires modern browser with WebGL support
- **Data Size:** 67 wells with full trajectory and log data
- **Performance:** Grid resolution set to 40x40 for balance of quality/speed
- **Encoding:** UTF-8 required for Cyrillic characters in data files

---

## Usage Patterns

### Typical Workflow
1. Click "Загрузить все данные" (Load all data) in sidebar
2. Wait for data loading confirmation
3. Select view mode from radio buttons
4. Adjust filters and settings as needed
5. Interact with visualizations (zoom, rotate, hover)
6. Export data if needed (Analysis tab)

### View Mode Selection Guide
- **Карта:** Quick overview of well locations and collector distribution
- **3D траектории:** Understand well paths and drilling patterns
- **3D пласты коллекторов:** Comprehensive 3D reservoir visualization
- **2D проекция скважины:** Detailed single-well analysis
- **Разрезы:** Examine raw geophysical log data
- **Анализ:** Statistical analysis and data export

---

## Integration Points

### ML Model Integration
- **Data Preparation:** Use [`prepare_ml_data()`](frontend/modules/preprocess.py:99)
- **Prediction Display:** Use [`create_prediction_heatmap()`](frontend/modules/visualizer.py:200)
- **Input Features:** X, Y, Z coordinates + LAS curve data
- **Target Variable:** Collector ratio (0-1)
- **Location:** Lines 169-193 in [`app.py`](app.py:169)

### Future Enhancements
- Real-time data updates
- Advanced geostatistical methods
- 4D visualization (time dimension)
- Multi-user collaboration features
- Cloud deployment

---

## File Naming Conventions
- Wells: `WELL_XXX` where XXX is 3-digit number (001-076)
- LAS files: `WELL_XXX.las`
- Data files: Descriptive names in Russian (H, EFF_H, траектории)

---

## Color Schemes

### Collector Visualization
- **Green:** Collector zones (value = 1)
- **Gray:** Non-collector zones (value = 0)
- **Yellow/Orange:** High collector ratio surfaces
- **Red:** Log curves
- **Black:** Wellbore traces

### Map Visualization
- **Viridis:** Collector ratio gradient (purple to yellow)
- **Qualitative.Plotly:** Well trajectory colors

---

## Performance Considerations
- Grid resolution: 40x40 (adjustable in code)
- Interpolation method: Cubic (balance of smoothness/speed)
- LAS data filtering: Remove null values before processing
- Session state: Prevents redundant data loading

---

## Error Handling
- Missing files: Graceful degradation with warnings
- Invalid data: NaN filtering and error messages
- Empty wells: Skip in visualizations
- Division by zero: Protected in collector ratio calculation

---

## Contact & Support
- **Team:** 2nd year course team
- **Event:** Hackathon: Software Development Technologies
- **Repository:** Git-based version control (main, dev, feature/* branches)

---

## Recent Changes (2025-12-16)

### MD to Z Mapping Fix
- **Issue:** Incorrect mapping of Measured Depth (MD) to Z coordinates in 3D visualization
- **Root Cause:** Linear approximation assumed MD = Z, which is incorrect for deviated wells
- **Solution:**
  - Implemented proper MD interpolation using trajectory data
  - Added validation of MD range compatibility
  - Added fallback handling for vertical wells
  - Files modified: [`frontend/modules/visualizer.py`](frontend/modules/visualizer.py:286)

### 3D Reservoir Layers Redesign
- **Change:** Complete rewrite of [`create_3d_reservoir_layers()`](frontend/modules/visualizer.py:286)
- **Old Approach:** Surface interpolation with griddata (top/bottom surfaces)
- **New Approach:** Trajectory-based layer rendering
  - Renders collector/non-collector segments along actual well trajectories
  - Uses same visualization style as [`create_3d_trajectories()`](frontend/modules/visualizer.py:61)
  - Applies layer logic from [`create_2d_well_projection()`](frontend/modules/visualizer.py:551)
- **Benefits:**
  - More accurate representation of geology along wellbore
  - Correct handling of deviated wells
  - Better visual clarity
  - Consistent with other 3D views

### Key Technical Details
- **MD (Measured Depth):** Length along wellbore from surface
- **Z Coordinate:** Vertical depth (can differ from MD in deviated wells)
- **Mapping Formula:** `Z = np.interp(MD_las, MD_trajectory, Z_trajectory)`
- **Layer Segments:** Grouped by consecutive collector/non-collector values

---

*Last Updated: 2025-12-16*
*Version: 1.1*