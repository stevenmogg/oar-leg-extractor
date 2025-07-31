import gpxpy
import pandas as pd
import xml.etree.ElementTree as ET
import uuid
from geopy.distance import geodesic
import re
from datetime import datetime
import math
import folium

# Conversion factors
METERS_TO_FEET = 3.28084
KMH_TO_KNOTS = 0.539956803455724
MPS_TO_KMH = 3.6
MPS_TO_KNOTS = 1.944
FLIGHT_SPEED = 25  # Knots - Speed at which assume in flight
POINT_SOURCE = ["AV_Plan", "MGL_Odyssey"]
FLIGHT_STATUS = ["BF", "IF", "AF"]

# Define the namespace for GPX and AvPlan extensions
namespace = {'gpx': 'http://www.topografix.com/GPX/1/1', 'avplan': 'avplan'}

# --- Toggle this flag to return a list of all matching names ---
SHOW_MATCHING_NAMES = False  # Set to True to filter names strictly

# Function to load the GPX file and parse it using gpxpy
# Accepts a file-like object or file path

def load_gpx_file(gpx_file):
    if hasattr(gpx_file, 'read'):
        gpx_file.seek(0)
        gpx = gpxpy.parse(gpx_file)
        gpx_file.seek(0)
        tree = ET.parse(gpx_file)
        root = tree.getroot()
    else:
        with open(gpx_file, 'r') as f:
            gpx = gpxpy.parse(f)
        tree = ET.parse(gpx_file)
        root = tree.getroot()
    return gpx, root

# Function to extract GPS track data from GPX file
def extract_gps_track(gpx):
    """Extract GPS track points from the GPX file"""
    track_data = []
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                track_point = {
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time,
                    'speed': getattr(point, 'speed', None),
                    'course': getattr(point, 'course', None)
                }
                track_data.append(track_point)
    
    return pd.DataFrame(track_data)

# Function to find the two closest points on the track (from original code)
def find_closest_points(gate_point, track_df):
    """Find the three closest points on the track to a gate point"""
    if track_df.empty:
        return None, None, None
    
    gate_lat, gate_lon = gate_point['latitude'], gate_point['longitude']
    
    # Calculate distances from the gate_point to all points in the track
    distances = track_df.apply(lambda row: geodesic((gate_lat, gate_lon), (row['latitude'], row['longitude'])).meters, axis=1)

    # Find the index of the closest point
    closest_index = distances.idxmin()

    # Find the second & third closest point (before and after the closest)
    if closest_index == 0:
        closest_index = 1
        second_closest_index = 0
        third_closest_index = 2
    elif closest_index == len(track_df) - 1:
        closest_index = len(track_df) - 2
        second_closest_index = len(track_df) - 3
        third_closest_index = len(track_df) - 1
    else:
        second_closest_index = closest_index - 1
        third_closest_index = closest_index + 1
    
    return closest_index, second_closest_index, third_closest_index

# Function to interpolate between points using geodesic calculations (improved version)
def interpolate_between_points(gate_point, track_df, idx1, idx2, idx3):
    """
    Interpolate the closest point on the geodesic line between track points to the gate point.
    Improved version that finds the actual closest point on the line segment.
    """
    if track_df.empty or idx1 is None:
        return None
    
    # Extract the three closest points from the track
    point1 = track_df.iloc[idx1]
    point2 = track_df.iloc[idx2]
    point3 = track_df.iloc[idx3]
    
    # Find the closest point on the line segment between point2 and point3
    # (this is the main segment we're interested in)
    gate_pos = (gate_point['latitude'], gate_point['longitude'])
    p2_pos = (point2['latitude'], point2['longitude'])
    p3_pos = (point3['latitude'], point3['longitude'])
    
    # Find closest point on the geodesic line between p2 and p3
    closest_point = find_closest_point_on_geodesic_line(gate_pos, p2_pos, p3_pos)
    
    if closest_point is None:
        return None
    
    closest_lat, closest_lon, fraction = closest_point
    
    # Calculate distance from gate to closest point
    closest_distance = geodesic(gate_pos, (closest_lat, closest_lon)).meters
    
    # Interpolate time at the closest point
    time_diff = (point3['time'] - point2['time']).total_seconds()
    interpolated_time = point2['time'] + pd.Timedelta(seconds=time_diff * fraction)
    
    # Interpolate altitude at the closest point
    if 'elevation' in point3 and 'elevation' in point2 and point3['elevation'] is not None and point2['elevation'] is not None:
        altitude_diff = point3['elevation'] - point2['elevation']
        interpolated_altitude = point2['elevation'] + (altitude_diff * fraction)
    else:
        interpolated_altitude = None
    
    return closest_distance, interpolated_time, interpolated_altitude

def find_closest_point_on_geodesic_line(gate_pos, p1_pos, p2_pos):
    """
    Find the closest point on the geodesic line between p1 and p2 to the gate point.
    Uses iterative approach to find the minimum distance point along the line.
    """
    # Use geopy for accurate distance calculations
    min_distance = float('inf')
    best_fraction = 0.5  # Start at middle
    best_point = p1_pos
    
    # Search with finer resolution
    for fraction in range(0, 101, 1):  # 0 to 100 in steps of 1%
        fraction = fraction / 100.0
        
        # Interpolate point along the geodesic
        interpolated_lat, interpolated_lon = interpolate_along_geodesic(
            p1_pos[0], p1_pos[1], p2_pos[0], p2_pos[1], fraction
        )
        
        # Calculate distance from gate to this interpolated point
        distance = geodesic(gate_pos, (interpolated_lat, interpolated_lon)).meters
        
        if distance < min_distance:
            min_distance = distance
            best_fraction = fraction
            best_point = (interpolated_lat, interpolated_lon)
    
    return best_point[0], best_point[1], best_fraction

def interpolate_along_geodesic(lat1, lon1, lat2, lon2, fraction):
    """
    Interpolate a point along a geodesic line between two points.
    Uses spherical trigonometry for accurate interpolation.
    """
    # Convert to radians for calculations
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate the angular distance between points
    angular_distance = geodesic_distance_radians(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
    
    if angular_distance == 0:
        return lat1, lon1
    
    # Calculate the bearing from point 1 to point 2
    bearing = math.atan2(
        math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad),
        math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
    )
    
    # Calculate the interpolated angular distance
    interpolated_angular_distance = angular_distance * fraction
    
    # Calculate the interpolated point using spherical trigonometry
    interpolated_lat_rad = math.asin(
        math.sin(lat1_rad) * math.cos(interpolated_angular_distance) +
        math.cos(lat1_rad) * math.sin(interpolated_angular_distance) * math.cos(bearing)
    )
    
    interpolated_lon_rad = lon1_rad + math.atan2(
        math.sin(bearing) * math.sin(interpolated_angular_distance) * math.cos(lat1_rad),
        math.cos(interpolated_angular_distance) - math.sin(lat1_rad) * math.sin(interpolated_lat_rad)
    )
    
    # Convert back to degrees
    return math.degrees(interpolated_lat_rad), math.degrees(interpolated_lon_rad)

def geodesic_distance_radians(lat1, lon1, lat2, lon2):
    """
    Calculate geodesic distance between two points in radians.
    Uses the haversine formula for spherical trigonometry.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return c

# Function to calculate closest approach for all waypoints
def calculate_closest_approach(waypoints_df, gpx, wheels_off_time=None):
    """Calculate closest approach distance and time intervals for all waypoints using the original approach"""
    track_df = extract_gps_track(gpx)
    
    if track_df.empty:
        # If no track data, return waypoints with None values
        waypoints_df['closest_approach_distance_m'] = None
        waypoints_df['time_interval_sec'] = None
        waypoints_df['closest_approach_time'] = None
        waypoints_df['closest_approach_altitude'] = None
        return waypoints_df
    
    results = []
    previous_closest_time = None
    
    for idx, waypoint in waypoints_df.iterrows():
        # Find closest points using the original method
        idx1, idx2, idx3 = find_closest_points(waypoint, track_df)
        
        if idx1 is None:
            # No track data available
            results.append({
                'closest_approach_distance_m': None,
                'time_interval_sec': None,
                'closest_approach_time': None,
                'closest_approach_altitude': None
            })
            continue
        
        # Interpolate between points to find closest approach
        result = interpolate_between_points(waypoint, track_df, idx1, idx2, idx3)
        
        if result is None:
            results.append({
                'closest_approach_distance_m': None,
                'time_interval_sec': None,
                'closest_approach_time': None,
                'closest_approach_altitude': None
            })
            continue
        
        closest_distance, closest_time, closest_altitude = result
        
        # Calculate time interval from previous closest approach point
        if previous_closest_time is not None:
            time_interval = (closest_time - previous_closest_time).total_seconds()
        else:
            time_interval = 0  # First waypoint
        
        previous_closest_time = closest_time
        
        results.append({
            'closest_approach_distance_m': closest_distance,
            'time_interval_sec': time_interval,
            'closest_approach_time': closest_time,
            'closest_approach_altitude': closest_altitude
        })
    
    # Add results to dataframe
    waypoints_df['closest_approach_distance_m'] = [r['closest_approach_distance_m'] for r in results]
    waypoints_df['time_interval_sec'] = [r['time_interval_sec'] for r in results]
    waypoints_df['closest_approach_time'] = [r['closest_approach_time'] for r in results]
    waypoints_df['closest_approach_altitude'] = [r['closest_approach_altitude'] for r in results]
    
    return waypoints_df

def interpolate_altitude_at_point(target_lat, target_lon, track_df, idx1, idx2, idx3):
    """
    Interpolate altitude at a specific point using the three closest track points.
    Uses linear interpolation based on distance weights.
    """
    if track_df.empty or idx1 is None:
        return None
    
    # Get the three closest points
    point1 = track_df.iloc[idx1]
    point2 = track_df.iloc[idx2]
    point3 = track_df.iloc[idx3]
    
    # Calculate distances from target to each point
    target_pos = (target_lat, target_lon)
    dist1 = geodesic(target_pos, (point1['latitude'], point1['longitude'])).meters
    dist2 = geodesic(target_pos, (point2['latitude'], point2['longitude'])).meters
    dist3 = geodesic(target_pos, (point3['latitude'], point3['longitude'])).meters
    
    # Use inverse distance weighting for interpolation
    total_weight = 0
    weighted_altitude = 0
    
    for point, dist in [(point1, dist1), (point2, dist2), (point3, dist3)]:
        if dist > 0 and point['elevation'] is not None:
            weight = 1.0 / dist
            total_weight += weight
            weighted_altitude += point['elevation'] * weight
    
    if total_weight > 0:
        return weighted_altitude / total_weight
    else:
        return None

def format_time_interval(seconds):
    """Format seconds as HH:MM:SS"""
    if seconds is None:
        return "00:00:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_time_from_wheels_off(time_obj, wheels_off_time):
    """Format time as HH:MM:SS from wheels_off time"""
    if time_obj is None or wheels_off_time is None:
        return "00:00:00"
    
    time_diff = (time_obj - wheels_off_time).total_seconds()
    return format_time_interval(time_diff)

# Function to extract Leg elements from AVPlan gpx file using the <rte> element
def extract_legs(root, gpx_file_path=None):
    leg_data = {
        'leg_guid': [],
        'leg_name': [],
        'leg_number': [],
        'src': [],
        'desc': [],
        'departure_time': [],
        'holding_time': [],
        'pob': [],
        'taxi_time': [],
        'block_off': [],
        'wheels_on': [],
        'wheels_off': [],
        'taxi_fuel': [],
        'approach_fuel': [],
        'load_ids': [],
        'load_values': [],
        'fuel_load_ids': [],
        'fuel_load_values': [],
        'gpx_file_path': []
    }
    for rte in root.findall('gpx:rte', namespace):
        leg_guid = str(uuid.uuid4())
        leg_name = rte.find('gpx:name', namespace).text if rte.find('gpx:name', namespace) is not None else None
        leg_number = rte.find('gpx:number', namespace).text if rte.find('gpx:number', namespace) is not None else None
        leg_details = rte.find('gpx:extensions/avplan:AvPlanLegDetails', namespace)
        if leg_details is not None:
            src = leg_details.find('avplan:src', namespace).text if leg_details.find('avplan:src', namespace) is not None else None
            desc = leg_details.find('avplan:desc', namespace).text if leg_details.find('avplan:desc', namespace) is not None else None
            departure_time = leg_details.find('avplan:AvPlanDepartureTime', namespace).text if leg_details.find('avplan:AvPlanDepartureTime', namespace) is not None else None
            holding_time = leg_details.find('avplan:AvPlanHoldingTime', namespace).text if leg_details.find('avplan:AvPlanHoldingTime', namespace) is not None else None
            pob = leg_details.find('avplan:AvPlanPOB', namespace).text if leg_details.find('avplan:AvPlanPOB', namespace) is not None else None
            taxi_time = leg_details.find('avplan:AvPlanTaxiTime', namespace).text if leg_details.find('avplan:AvPlanTaxiTime', namespace) is not None else None
            block_off = leg_details.find('avplan:AvPlanBlockOff', namespace).text if leg_details.find('avplan:AvPlanBlockOff', namespace) is not None else None
            wheels_on = leg_details.find('avplan:AvPlanWheelsOn', namespace).text if leg_details.find('avplan:AvPlanWheelsOn', namespace) is not None else None
            wheels_off = leg_details.find('avplan:AvPlanWheelsOff', namespace).text if leg_details.find('avplan:AvPlanWheelsOff', namespace) is not None else None
            taxi_fuel = leg_details.find('avplan:AvPlanTaxiFuel', namespace).text if leg_details.find('avplan:AvPlanTaxiFuel', namespace) is not None else None
            approach_fuel = leg_details.find('avplan:AvPlanApproachFuel', namespace).text if leg_details.find('avplan:AvPlanApproachFuel', namespace) is not None else None
            load_ids = []
            load_values = []
            for load in leg_details.findall('avplan:AvPlanLoading/avplan:AvPlanLoad', namespace):
                load_id = load.find('avplan:AvPlanLoadID', namespace).text if load.find('avplan:AvPlanLoadID', namespace) is not None else None
                load_value = load.find('avplan:AvPlanLoadValue', namespace).text if load.find('avplan:AvPlanLoadValue', namespace) is not None else None
                load_ids.append(load_id)
                load_values.append(load_value)
            fuel_load_ids = []
            fuel_load_values = []
            for fuel_load in leg_details.findall('avplan:AvPlanFuelLoading/avplan:AvPlanFuelLoad', namespace):
                fuel_load_id = fuel_load.find('avplan:AvPlanFuelLoadID', namespace).text if fuel_load.find('avplan:AvPlanFuelLoadID', namespace) is not None else None
                fuel_load_value = fuel_load.find('avplan:AvPlanFuelLoadValue', namespace).text if fuel_load.find('avplan:AvPlanFuelLoadValue', namespace) is not None else None
                fuel_load_ids.append(fuel_load_id)
                fuel_load_values.append(fuel_load_value)
            leg_data['leg_guid'].append(leg_guid)
            leg_data['leg_name'].append(leg_name)
            leg_data['leg_number'].append(leg_number)
            leg_data['src'].append(src)
            leg_data['desc'].append(desc)
            leg_data['departure_time'].append(departure_time)
            leg_data['holding_time'].append(holding_time)
            leg_data['pob'].append(pob)
            leg_data['taxi_time'].append(taxi_time)
            leg_data['block_off'].append(block_off)
            leg_data['wheels_on'].append(wheels_on)
            leg_data['wheels_off'].append(wheels_off)
            leg_data['taxi_fuel'].append(taxi_fuel)
            leg_data['approach_fuel'].append(approach_fuel)
            leg_data['load_ids'].append(load_ids)
            leg_data['load_values'].append(load_values)
            leg_data['fuel_load_ids'].append(fuel_load_ids)
            leg_data['fuel_load_values'].append(fuel_load_values)
            leg_data['gpx_file_path'].append(gpx_file_path)
    leg_df = pd.DataFrame(leg_data)
    return leg_df

# Function to extract route points from an AVPlan gpx file
def extract_routes(root, leg_guid):
    route_data = {
        'leg_guid': [],
        'latitude': [],
        'longitude': [],
        'name': [],
        'type': [],
        'altitude': [],
        'description': [],
        'magvar': [],
        'delay': [],
        'alternate': [],
        'lsalt': [],
        'segment_rules': [],
        'track': [],
        'distance': [],
        'distance_remaining': [],
        'eta': [],
        'ata': [],
        'actual_fob': [],
        'distance_error': []
    }
    prev_lat = None
    prev_lon = None
    prev_planned_distance = None
    for rtept in root.findall('gpx:rte/gpx:rtept', namespace):
        lat = rtept.get('lat')
        lon = rtept.get('lon')
        name = rtept.find('gpx:name', namespace).text if rtept.find('gpx:name', namespace) is not None else None
        wpt_type = rtept.find('gpx:type', namespace).text if rtept.find('gpx:type', namespace) is not None else None
        waypoint_details = rtept.find('gpx:extensions/avplan:AvPlanWaypointDetails', namespace)
        altitude = description = magvar = delay = alternate = lsalt = segment_rules = track = distance = distance_remaining = eta = ata = actual_fob = None
        if waypoint_details is not None:
            altitude = waypoint_details.find('avplan:AvPlanAltitude', namespace).text if waypoint_details.find('avplan:AvPlanAltitude', namespace) is not None else None
            description = waypoint_details.find('avplan:desc', namespace).text if waypoint_details.find('avplan:desc', namespace) is not None else None
            magvar = waypoint_details.find('avplan:magvar', namespace).text if waypoint_details.find('avplan:magvar', namespace) is not None else None
            delay = waypoint_details.find('avplan:AvPlanDelay', namespace).text if waypoint_details.find('avplan:AvPlanDelay', namespace) is not None else None
            alternate = waypoint_details.find('avplan:AvPlanAlternate', namespace).text if waypoint_details.find('avplan:AvPlanAlternate', namespace) is not None else None
            lsalt = waypoint_details.find('avplan:AvPlanLSALT', namespace).text if waypoint_details.find('avplan:AvPlanLSALT', namespace) is not None else None
            segment_rules = waypoint_details.find('avplan:AvPlanSegRules', namespace).text if waypoint_details.find('avplan:AvPlanSegRules', namespace) is not None else None
            track = waypoint_details.find('avplan:AvPlanTrack', namespace).text if waypoint_details.find('avplan:AvPlanTrack', namespace) is not None else None
            distance = waypoint_details.find('avplan:AvPlanDistance', namespace).text if waypoint_details.find('avplan:AvPlanDistance', namespace) is not None else None
            distance_remaining = waypoint_details.find('avplan:AvPlanDistanceRemain', namespace).text if waypoint_details.find('avplan:AvPlanDistanceRemain', namespace) is not None else None
            eta = waypoint_details.find('avplan:AvPlanETA', namespace).text if waypoint_details.find('avplan:AvPlanETA', namespace) is not None else None
            ata = waypoint_details.find('avplan:AvPlanATA', namespace).text if waypoint_details.find('avplan:AvPlanATA', namespace) is not None else None
            actual_fob = waypoint_details.find('avplan:AvPlanActualFOB', namespace).text if waypoint_details.find('avplan:AvPlanActualFOB', namespace) is not None else None
        # Calculate distance error
        if prev_lat is not None and prev_lon is not None:
            actual_distance = geodesic((prev_lat, prev_lon), (float(lat), float(lon))).meters
            planned_distance = float(distance) if distance is not None else None
            if planned_distance is not None:
                distance_error = planned_distance - actual_distance
            else:
                distance_error = None
        else:
            distance_error = None
        route_data['leg_guid'].append(leg_guid)
        route_data['latitude'].append(float(lat))
        route_data['longitude'].append(float(lon))
        route_data['name'].append(name)
        route_data['type'].append(wpt_type)
        route_data['altitude'].append(altitude)
        route_data['description'].append(description)
        route_data['magvar'].append(magvar)
        route_data['delay'].append(delay)
        route_data['alternate'].append(alternate)
        route_data['lsalt'].append(lsalt)
        route_data['segment_rules'].append(segment_rules)
        route_data['track'].append(track)
        route_data['distance'].append(distance)
        route_data['distance_remaining'].append(distance_remaining)
        route_data['eta'].append(eta)
        route_data['ata'].append(ata)
        route_data['actual_fob'].append(actual_fob)
        route_data['distance_error'].append(distance_error)
        prev_lat = float(lat)
        prev_lon = float(lon)
        prev_planned_distance = float(distance) if distance is not None else None
    route_df = pd.DataFrame(route_data)
    return route_df

# Function to check if a name starts with "in" or "out" followed by a number
def is_in_or_out_with_number(name):
    # Use regex to match "in" or "out" followed by a number
    if SHOW_MATCHING_NAMES:
        return bool(re.match(r'^(in|out)\d+', name.lower()))
    else:
        return True  # Accept all names if filtering is turned off

# Function to extract gate points from route data
def extract_gate_points(route_df):
    gate_points = []
    for index, row in route_df.iterrows():
        if is_in_or_out_with_number(row['name']):
            # Create a dictionary with the latitude, longitude, and elevation
            gate_point = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'elevation': row['altitude']  # Assuming 'altitude' is the correct field for elevation
            }
            # Add the gate point to the list
            gate_points.append(gate_point)
    return gate_points

def calculate_waypoint_time_matrix(waypoints_df, format_intervals=False):
    """
    Returns a DataFrame where entry (i, j) is the absolute time interval between
    the closest_approach_time of waypoint i and waypoint j.
    The index and columns are waypoint names.
    If format_intervals is True, intervals are formatted as H:M:S.
    """
    # Ensure required columns exist
    if 'name' not in waypoints_df.columns or 'closest_approach_time' not in waypoints_df.columns:
        raise ValueError("Waypoints DataFrame must have 'name' and 'closest_approach_time' columns.")
    
    # Drop waypoints with missing times
    df = waypoints_df.dropna(subset=['name', 'closest_approach_time']).copy()
    df = df.reset_index(drop=True)
    names = df['name'].tolist()
    times = df['closest_approach_time'].tolist()
    
    # Build the matrix
    matrix = []
    for t1 in times:
        row = []
        for t2 in times:
            if pd.isnull(t1) or pd.isnull(t2):
                row.append(None)
            else:
                interval = abs((t2 - t1).total_seconds())
                if format_intervals:
                    row.append(format_time_interval(interval))
                else:
                    row.append(interval)
        matrix.append(row)
    
    # Create DataFrame
    matrix_df = pd.DataFrame(matrix, index=names, columns=names)
    return matrix_df 

def create_comprehensive_map(track_df, waypoints_df, gpx):
    """
    Create a comprehensive folium map showing:
    - Blue markers for track points with hover details
    - Yellow markers for waypoints with hover details  
    - Red markers for closest approach points with analysis data
    """
    if track_df.empty:
        return None
    
    # Calculate center of the map
    center_lat = track_df['latitude'].mean()
    center_lon = track_df['longitude'].mean()
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add track points as blue markers
    for idx, row in track_df.iterrows():
        # Create popup content for track points
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
        
        # Add blue marker for track point
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=350),
            tooltip=f"Track Point {idx}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Add track line
    track_coords = track_df[['latitude', 'longitude']].values.tolist()
    folium.PolyLine(
        track_coords,
        color='blue',
        weight=2,
        opacity=0.8
    ).add_to(m)
    
    # Add waypoints as yellow markers
    if not waypoints_df.empty:
        for idx, waypoint in waypoints_df.iterrows():
            # Create popup content for waypoints
            popup_content = f"""
            <div style="width: 300px;">
                <h4>Waypoint: {waypoint['name']}</h4>
                <table style="width: 100%; font-size: 12px;">
                    <tr><td style="font-weight: bold;">Latitude:</td><td>{waypoint['latitude']:.6f}</td></tr>
                    <tr><td style="font-weight: bold;">Longitude:</td><td>{waypoint['longitude']:.6f}</td></tr>
            """
            
            if 'altitude' in waypoint and pd.notna(waypoint['altitude']):
                popup_content += f"<tr><td style='font-weight: bold;'>Altitude:</td><td>{waypoint['altitude']:.1f} m</td></tr>"
            
            popup_content += """
                </table>
            </div>
            """
            
            # Add yellow marker for waypoint
            folium.Marker(
                location=[waypoint['latitude'], waypoint['longitude']],
                popup=folium.Popup(popup_content, max_width=350),
                tooltip=f"Waypoint: {waypoint['name']}",
                icon=folium.Icon(color='yellow', icon='info-sign')
            ).add_to(m)
    
    # Add closest approach points as red markers
    if not waypoints_df.empty and 'closest_approach_distance_m' in waypoints_df.columns:
        for idx, waypoint in waypoints_df.iterrows():
            if pd.notna(waypoint['closest_approach_distance_m']):
                # Create popup content for closest approach analysis
                popup_content = f"""
                <div style="width: 350px;">
                    <h4>Closest Approach: {waypoint['name']}</h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td style="font-weight: bold;">Distance:</td><td>{waypoint['closest_approach_distance_m']:.1f} m</td></tr>
                """
                
                if 'closest_approach_altitude' in waypoint and pd.notna(waypoint['closest_approach_altitude']):
                    popup_content += f"<tr><td style='font-weight: bold;'>Altitude:</td><td>{waypoint['closest_approach_altitude']:.1f} m</td></tr>"
                
                if 'time_interval_sec' in waypoint and pd.notna(waypoint['time_interval_sec']):
                    interval_formatted = format_time_interval(waypoint['time_interval_sec'])
                    popup_content += f"<tr><td style='font-weight: bold;'>Interval:</td><td>{interval_formatted}</td></tr>"
                
                if 'closest_approach_time' in waypoint and pd.notna(waypoint['closest_approach_time']):
                    time_str = waypoint['closest_approach_time'].strftime("%H:%M:%S")
                    popup_content += f"<tr><td style='font-weight: bold;'>Time:</td><td>{time_str}</td></tr>"
                
                popup_content += """
                    </table>
                </div>
                """
                
                # Add red marker for closest approach point
                folium.Marker(
                    location=[waypoint['latitude'], waypoint['longitude']],
                    popup=folium.Popup(popup_content, max_width=400),
                    tooltip=f"Closest Approach: {waypoint['name']}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
    
    return m 