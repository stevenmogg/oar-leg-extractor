# OAR Leg Extractor

A comprehensive Streamlit web application for analyzing Out and Return (OAR) flight legs from AVPlan GPX files. This tool provides detailed flight analysis including waypoint tracking, closest approach calculations, and interactive mapping.

## 🚀 Features

- **GPX File Processing**: Upload and analyze AVPlan GPX flight files
- **Leg Extraction**: Automatically extract flight legs and waypoints
- **Closest Approach Analysis**: Calculate the closest approach to each waypoint
- **Interactive Maps**: Visualize flight tracks and waypoints on interactive maps
- **Time Analysis**: Track time intervals and flight duration
- **Statistics**: Comprehensive distance and altitude statistics
- **Raw Data View**: Examine raw GPS track data with detailed popups

## 📋 Requirements

- Python 3.8+
- Streamlit
- GPX processing libraries
- Mapping and geospatial tools

## 🛠️ Local Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd OAR_Leg_Project
```

### 2. Create Virtual Environment
```bash
python3 -m venv oar_leg_env
source oar_leg_env/bin/activate  # On Windows: oar_leg_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy automatically

### Other Options
- **Heroku**: Add `Procfile` and `runtime.txt`
- **Railway**: Connect GitHub repository
- **Google Cloud Run**: Containerize the application

## 📁 Project Structure

```
OAR_Leg_Project/
├── app.py                 # Main Streamlit application
├── utils/
│   └── extract_legs.py   # Core GPX processing logic
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── pages/               # Additional Streamlit pages
├── mgl_rec/            # Flight data files (gitignored)
└── README.md           # This file
```

## 🔧 Configuration

The application can be configured through:
- `.streamlit/config.toml` - Streamlit settings
- `requirements.txt` - Python dependencies
- Environment variables for deployment

## 📊 Usage

1. **Upload GPX File**: Use the file uploader to select an AVPlan GPX file
2. **View Legs**: Examine extracted flight legs in the data table
3. **Analyze Waypoints**: Review waypoint analysis with closest approach data
4. **Interactive Maps**: Click "View Comprehensive Map" for flight visualization
5. **Raw Data**: Use "View Raw Track Data on Map" for detailed GPS analysis

## 🎯 Key Features Explained

- **Closest Approach**: Finds the three closest GPS track points to each waypoint
- **Time Intervals**: Calculates time between consecutive closest approach points
- **Altitude Interpolation**: Estimates altitude at closest approach points
- **Distance Statistics**: Provides comprehensive distance analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in the code comments
- Review the Streamlit logs for debugging information

---

**Note**: This application is designed for aviation flight analysis and requires AVPlan GPX files for optimal functionality. 