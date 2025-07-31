import streamlit as st
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import folium_static
from utils.extract_legs import (
    load_gpx_file, extract_legs, extract_routes, 
    extract_gate_points, is_in_or_out_with_number, SHOW_MATCHING_NAMES,
    calculate_closest_approach, format_time_interval, format_time_from_wheels_off,
    extract_gps_track, calculate_waypoint_time_matrix, create_comprehensive_map
)

# Initialize session state
if 'show_track_map' not in st.session_state:
    st.session_state.show_track_map = False
if 'track_data' not in st.session_state:
    st.session_state.track_data = None

# Check if we should show the track map page
if st.session_state.show_track_map and st.session_state.track_data is not None:
    st.title("Raw Track Data Map")
    
    # Add back button
    if st.button("← Back to Main App"):
        st.session_state.show_track_map = False
        st.session_state.track_data = None
        st.rerun()
    
    track_df = st.session_state.track_data
    
    # Create map centered on the track
    center_lat = track_df['latitude'].mean()
    center_lon = track_df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add track points with popups
    for idx, row in track_df.iterrows():
        # Create popup content with all data
        popup_content = f"""
        <div style="width: 300px;">
            <h4>Track Point {idx}</h4>
            <table style="width: 100%; font-size: 12px;">
        """
        
        for col in track_df.columns:
            value = row[col]
            if pd.isna(value):
                value = "N/A"
            elif isinstance(value, (int, float)):
                value = f"{value:.6f}" if isinstance(value, float) else str(value)
            elif isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                value = str(value)
            
            popup_content += f"""
                <tr>
                    <td style="font-weight: bold; padding: 2px;">{col}:</td>
                    <td style="padding: 2px;">{value}</td>
                </tr>
            """
        
        popup_content += """
            </table>
        </div>
        """
        
        # Add marker to map
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=350),
            tooltip=f"Point {idx}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # Add track line
    track_coords = track_df[['latitude', 'longitude']].values.tolist()
    folium.PolyLine(
        track_coords,
        color='blue',
        weight=2,
        opacity=0.8
    ).add_to(m)
    
    # Display the map
    folium_static(m, width=800, height=600)
    
    # Show track statistics
    st.subheader("Track Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Points", len(track_df))
        if 'time' in track_df.columns:
            time_range = track_df['time'].max() - track_df['time'].min()
            st.metric("Duration", str(time_range).split('.')[0])
    
    with col2:
        if 'elevation' in track_df.columns:
            st.metric("Min Elevation", f"{track_df['elevation'].min():.0f} m")
            st.metric("Max Elevation", f"{track_df['elevation'].max():.0f} m")
    
    with col3:
        if 'speed' in track_df.columns:
            st.metric("Avg Speed", f"{track_df['speed'].mean():.1f} m/s")
            st.metric("Max Speed", f"{track_df['speed'].max():.1f} m/s")
    
    # Show raw data table
    st.subheader("Raw Track Data Table")
    st.dataframe(track_df, use_container_width=True)
    
else:
    # Main app content
    st.title("OAR Leg Extractor")

# Add toggle for testing all waypoints vs just in/out waypoints
show_all_waypoints = st.checkbox("Show all waypoints (for testing)", value=True, 
                                help="When checked, shows all waypoints. When unchecked, shows only 'in'/'out' waypoints.")

uploaded_file = st.file_uploader("Upload AVPlan GPX file", type=["gpx"])

if uploaded_file is not None:
    try:
        gpx, root = load_gpx_file(uploaded_file)
        legs_df = extract_legs(root)
        
        if not legs_df.empty:
            st.subheader("Extracted Legs")
            st.dataframe(legs_df)
            
            # Get wheels_off time from the first leg and convert to datetime
            wheels_off_time_str = legs_df.iloc[0]['wheels_off']
            if wheels_off_time_str:
                try:
                    wheels_off_time = pd.to_datetime(wheels_off_time_str)
                except:
                    wheels_off_time = None
            else:
                wheels_off_time = None
            
            # Extract GPS track data and show its structure
            track_df = extract_gps_track(gpx)
            if not track_df.empty:
                # Add button to view raw track data on map
                if st.button("View Raw Track Data on Map", type="primary"):
                    st.session_state.show_track_map = True
                    st.session_state.track_data = track_df
                    st.rerun()
            else:
                st.warning("No GPS track data found in the GPX file for closest approach analysis.")
            
            # Extract waypoints
            waypoints_df = extract_routes(root, legs_df.iloc[0]['leg_guid'] if not legs_df.empty else None)
            
            if not waypoints_df.empty:
                # Filter waypoints if needed
                if not show_all_waypoints:
                    waypoints_df = waypoints_df[waypoints_df['name'].apply(is_in_or_out_with_number)]
                
                # Calculate closest approach analysis
                waypoints_df = calculate_closest_approach(waypoints_df, gpx, wheels_off_time)
                
                # Format the time columns
                if 'time_interval_sec' in waypoints_df.columns:
                    waypoints_df['Interval'] = waypoints_df['time_interval_sec'].apply(format_time_interval)
                
                if 'closest_approach_time' in waypoints_df.columns and wheels_off_time is not None:
                    waypoints_df['Time from Start'] = waypoints_df['closest_approach_time'].apply(
                        lambda x: format_time_from_wheels_off(x, wheels_off_time)
                    )
                
                # Round altitude to 1 decimal place
                if 'closest_approach_altitude' in waypoints_df.columns:
                    waypoints_df['Closest Approach Altitude'] = waypoints_df['closest_approach_altitude'].apply(
                        lambda x: round(x, 1) if x is not None else None
                    )
                
                # Select and rename columns for display
                display_columns = {
                    'name': 'Name',
                    'latitude': 'Latitude', 
                    'longitude': 'Longitude',
                    'altitude': 'Altitude',
                    'closest_approach_distance_m': 'Closest Approach',
                    'Interval': 'Interval',
                    'Time from Start': 'Time from Start',
                    'closest_approach_altitude': 'Closest Approach Altitude'
                }
                
                # Only include columns that exist in the dataframe
                available_columns = {k: v for k, v in display_columns.items() if k in waypoints_df.columns}
                display_df = waypoints_df[list(available_columns.keys())].rename(columns=available_columns)
                
                st.subheader("All Waypoints with Closest Approach Analysis")
                st.dataframe(display_df)

                # Show waypoint time interval matrix
                try:
                    matrix_df = calculate_waypoint_time_matrix(waypoints_df, format_intervals=True)
                    st.subheader("Waypoint Time Interval Matrix (H:M:S)")
                    st.dataframe(matrix_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute waypoint time interval matrix: {e}")

                # Add button to view comprehensive map
                if st.button("View Comprehensive Map", type="primary"):
                    try:
                        comprehensive_map = create_comprehensive_map(track_df, waypoints_df, gpx)
                        if comprehensive_map is not None:
                            st.subheader("Comprehensive Flight Analysis Map")
                            st.write("**Legend:** Blue = Track Points, Yellow = Waypoints, Red = Closest Approach Points")
                            folium_static(comprehensive_map, width=800, height=600)
                        else:
                            st.warning("Could not create comprehensive map - no track data available.")
                    except Exception as e:
                        st.error(f"Error creating comprehensive map: {str(e)}")

                # Show statistics
                if 'closest_approach_distance_m' in waypoints_df.columns:
                    distances = waypoints_df['closest_approach_distance_m'].dropna()
                    if not distances.empty:
                        st.subheader("Distance Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average", f"{distances.mean():.1f} m")
                        with col2:
                            st.metric("Min", f"{distances.min():.1f} m")
                        with col3:
                            st.metric("Max", f"{distances.max():.1f} m")
                        with col4:
                            st.metric("Std Dev", f"{distances.std():.1f} m")
            else:
                st.warning("No waypoints found in the GPX file.")
            
            # Note about GPS track requirement
            st.warning("""
            **Note:** The closest approach analysis uses GPS track data extracted from the same GPX file.
            If no track data is found, the distance and time columns will show as empty.
            
            **Current Status:** Waypoints are detected and closest approach analysis is performed automatically.
            The analysis finds the three closest track points and calculates the closest approach distance.
            Time intervals show the time between consecutive closest approach points.
            Altitude is interpolated from the closest track points.
            Start time is based on wheels_off time from the leg data.
            """)
        else:
            st.warning("No legs found in the GPX file.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)
else:
    st.info("Please upload a GPX file to begin.") 