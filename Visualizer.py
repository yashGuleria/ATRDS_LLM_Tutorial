import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2, degrees
import warnings
warnings.filterwarnings('ignore')

class MultiAircraftVisualizer:
    def __init__(self, waypoints_csv_path, sector_boundary_csv_path=None, time_step_seconds=10):
        self.time_step_seconds = time_step_seconds
        
        # Load waypoints data
        self.waypoints_df = pd.read_csv(waypoints_csv_path)
        self.waypoints_dict = {
            row['AWaypoint']: (row['Lat'], row['Lon']) 
            for _, row in self.waypoints_df.iterrows()
        }
        
        # Load sector boundary
        self.sector_boundary = None
        if sector_boundary_csv_path:
            self.load_sector_boundary(sector_boundary_csv_path)
        
        # Aircraft specifications
        self.aircraft_specs = {
            'A320': 420, 'A321': 430, 'A330': 450, 'A333': 450, 'A340': 460,
            'A350': 470, 'A380': 480, 'A388': 480, 'B737': 410, 'B738': 415,
            'B777': 470, 'B787': 465, 'B747': 475, 'B744': 475
        }
        
        self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    def load_sector_boundary(self, sector_csv_path):
        try:
            # Read CSV without header since first row contains null values
            sector_df = pd.read_csv(sector_csv_path, header=None)
            
            # Remove first row (header with null values) and get clean data
            clean_data = sector_df.iloc[1:].dropna()
            
            if len(clean_data.columns) >= 2:
                col1_vals = clean_data.iloc[:, 0]
                col2_vals = clean_data.iloc[:, 1]
                
                # First column is latitude (smaller values ~2-4), second is longitude (larger values ~104-107)
                if col1_vals.mean() > col2_vals.mean():
                    lons, lats = col1_vals.tolist(), col2_vals.tolist()
                else:
                    lats, lons = col1_vals.tolist(), col2_vals.tolist()
                
                self.sector_boundary = list(zip(lats, lons))
                print(f"Sector boundary loaded: {len(self.sector_boundary)} points")
                print(f"Lat range: {min(lats):.3f} to {max(lats):.3f}, Lon range: {min(lons):.3f} to {max(lons):.3f}")
            
        except Exception as e:
            print(f"Warning: Could not load sector boundary: {e}")
            self.sector_boundary = None
    
    def parse_coordinate(self, coord_str):
        direction = coord_str[-1]
        coord_num = coord_str[:-1]
        
        if len(coord_num) == 9:  # DDMMSS.SS
            degrees_part = int(coord_num[:2])
            minutes_part = int(coord_num[2:4])
            seconds_part = float(coord_num[4:])
        else:  # DDDMMSS.SS
            degrees_part = int(coord_num[:3])
            minutes_part = int(coord_num[3:5])
            seconds_part = float(coord_num[5:])
        
        decimal_degrees = degrees_part + minutes_part/60.0 + seconds_part/3600.0
        
        if direction in ['S', 'W']:
            decimal_degrees = -decimal_degrees
        
        return decimal_degrees
    
    def parse_xml_flight_plans(self, xml_content):
        root = ET.fromstring(xml_content)
        flight_plans = []
        
        for initial_fp in root.findall('.//initial-flightplans'):
            flight_plan = {
                'callsign': initial_fp.find('callsign').text,
                'aircraft_type': initial_fp.find('type').text,
                'time': int(initial_fp.find('time').text),
                'route': [route.text for route in initial_fp.findall('air_route')]
            }
            
            # Parse initial position
            init_pos = initial_fp.find('.//init/pos')
            flight_plan['initial_lat'] = self.parse_coordinate(init_pos.find('lat').text)
            flight_plan['initial_lon'] = self.parse_coordinate(init_pos.find('lon').text)
            
            # Get aircraft speed
            aircraft_type = flight_plan['aircraft_type']
            flight_plan['speed_knots'] = self.aircraft_specs.get(aircraft_type, 450)
            
            flight_plans.append(flight_plan)
        
        return flight_plans
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 3440.065 * 2 * atan2(sqrt(a), sqrt(1-a))
    
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        return (degrees(atan2(y, x)) + 360) % 360
    
    def interpolate_position(self, lat1, lon1, lat2, lon2, fraction):
        if fraction <= 0: return lat1, lon1
        if fraction >= 1: return lat2, lon2
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        d = 2 * np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + 
                                 np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2))
        
        a = np.sin((1-fraction) * d) / np.sin(d)
        b = np.sin(fraction * d) / np.sin(d)
        
        x = a * np.cos(lat1) * np.cos(lon1) + b * np.cos(lat2) * np.cos(lon2)
        y = a * np.cos(lat1) * np.sin(lon1) + b * np.cos(lat2) * np.sin(lon2)
        z = a * np.sin(lat1) + b * np.sin(lat2)
        
        lat_interp = np.arctan2(z, np.sqrt(x**2 + y**2))
        lon_interp = np.arctan2(y, x)
        
        return degrees(lat_interp), degrees(lon_interp)
    
    def calculate_flight_path(self, flight_plan):
        # Build route coordinates
        route_coords = [(flight_plan['initial_lat'], flight_plan['initial_lon'])]
        for waypoint in flight_plan['route']:
            if waypoint in self.waypoints_dict:
                route_coords.append(self.waypoints_dict[waypoint])
        
        # Calculate segments
        path_segments = []
        total_distance = 0
        
        for i in range(len(route_coords) - 1):
            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i + 1]
            
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            bearing = self.calculate_bearing(lat1, lon1, lat2, lon2)
            
            path_segments.append({
                'start_lat': lat1, 'start_lon': lon1,
                'end_lat': lat2, 'end_lon': lon2,
                'distance_nm': distance, 'bearing': bearing,
                'start_distance': total_distance,
                'end_distance': total_distance + distance
            })
            total_distance += distance
        
        # Calculate time steps
        speed_nm_per_second = flight_plan['speed_knots'] / 3600.0
        distance_per_step = speed_nm_per_second * self.time_step_seconds
        
        time_steps = []
        current_distance = 0
        time_step = 0
        
        while current_distance < total_distance:
            # Find current segment
            current_segment = None
            for segment in path_segments:
                if segment['start_distance'] <= current_distance <= segment['end_distance']:
                    current_segment = segment
                    break
            
            if current_segment is None:
                break
            
            # Calculate position
            segment_distance = current_distance - current_segment['start_distance']
            segment_fraction = segment_distance / current_segment['distance_nm']
            
            lat, lon = self.interpolate_position(
                current_segment['start_lat'], current_segment['start_lon'],
                current_segment['end_lat'], current_segment['end_lon'],
                segment_fraction
            )
            
            time_steps.append({
                'time_seconds': time_step * self.time_step_seconds + flight_plan['time'],
                'latitude': lat, 'longitude': lon,
                'distance_covered': current_distance,
                'bearing': current_segment['bearing']
            })
            
            current_distance += distance_per_step
            time_step += 1
        
        return {
            'flight_plan': flight_plan,
            'route_coords': route_coords,
            'time_steps': time_steps,
            'total_distance_nm': total_distance
        }
    
    def calculate_all_flight_paths(self, flight_plans):
        all_flight_paths = []
        for flight_plan in flight_plans:
            flight_path = self.calculate_flight_path(flight_plan)
            all_flight_paths.append(flight_path)
            print(f"{flight_plan['callsign']}: {flight_path['total_distance_nm']:.1f} NM, "
                  f"{len(flight_path['time_steps']) * self.time_step_seconds / 60:.1f} min")
        return all_flight_paths
    
    def create_static_plot(self, all_flight_paths, save_path=None):
        plt.figure(figsize=(14, 10))
        
        # Plot sector boundary first (so it appears behind everything else)
        if self.sector_boundary:
            boundary_lats = [coord[0] for coord in self.sector_boundary]
            boundary_lons = [coord[1] for coord in self.sector_boundary]
            
            print(f"Plotting sector boundary with {len(boundary_lats)} points")
            print(f"Boundary coords: Lat {min(boundary_lats):.3f}-{max(boundary_lats):.3f}, Lon {min(boundary_lons):.3f}-{max(boundary_lons):.3f}")
            
            # Fill the sector boundary area
            plt.fill(boundary_lons, boundary_lats, alpha=0.3, color='lightblue', 
                    label='Sector 6 Boundary', edgecolor='navy', linewidth=3, zorder=1)
            
            # Also plot the boundary line for better visibility
            boundary_lons_closed = boundary_lons + [boundary_lons[0]]  # Close the polygon
            boundary_lats_closed = boundary_lats + [boundary_lats[0]]
            plt.plot(boundary_lons_closed, boundary_lats_closed, color='navy', 
                    linewidth=3, alpha=0.8, zorder=2)
        else:
            print("No sector boundary data loaded!")
        
        # Plot aircraft paths
        for i, flight_path in enumerate(all_flight_paths):
            color = self.colors[i % len(self.colors)]
            callsign = flight_path['flight_plan']['callsign']
            
            # Plot route
            waypoint_lats = [coord[0] for coord in flight_path['route_coords']]
            waypoint_lons = [coord[1] for coord in flight_path['route_coords']]
            
            plt.plot(waypoint_lons, waypoint_lats, color=color, linewidth=3, 
                    alpha=0.9, label=f'{callsign}', zorder=4)
            plt.scatter(waypoint_lons, waypoint_lats, color=color, s=100, 
                       zorder=6, marker='o', edgecolors='white', linewidth=2)
            
            # Plot aircraft positions
            aircraft_lats = [step['latitude'] for step in flight_path['time_steps']]
            aircraft_lons = [step['longitude'] for step in flight_path['time_steps']]
            plt.scatter(aircraft_lons, aircraft_lats, color=color, s=20, alpha=0.5, 
                       marker='.', zorder=3)
            
            # Mark start position
            start_lat, start_lon = flight_path['route_coords'][0]
            plt.scatter(start_lon, start_lat, color=color, s=250, 
                       marker='^', zorder=7, edgecolors='black', linewidth=2)
            
            # Add callsign label
            plt.annotate(f'{callsign}\n{flight_path["flight_plan"]["aircraft_type"]}', 
                        (start_lon, start_lat), xytext=(15, 15), 
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8, 
                                edgecolor='white'), zorder=8)
        
        plt.xlabel('Longitude (degrees)', fontsize=12)
        plt.ylabel('Latitude (degrees)', fontsize=12)
        plt.title('Scenario visualization', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Static plot saved to: {save_path}")
        plt.show()
    
    def create_animation(self, all_flight_paths, save_path=None, interval=100):
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot sector boundary first
        if self.sector_boundary:
            boundary_lats = [coord[0] for coord in self.sector_boundary]
            boundary_lons = [coord[1] for coord in self.sector_boundary]
            
            
            # Fill the sector boundary area
            ax.fill(boundary_lons, boundary_lats, alpha=0.3, color='lightblue', 
                   label='Sector 6 Boundary', edgecolor='navy', linewidth=3, zorder=1)
            
            # Also plot the boundary line
            boundary_lons_closed = boundary_lons + [boundary_lons[0]]
            boundary_lats_closed = boundary_lats + [boundary_lats[0]]
            ax.plot(boundary_lons_closed, boundary_lats_closed, color='navy', 
                   linewidth=3, alpha=0.8, zorder=2)
        else:
            print("No sector boundary data for animation!")
        
        # Setup static elements and animation objects
        aircraft_elements = []
        for i, flight_path in enumerate(all_flight_paths):
            color = self.colors[i % len(self.colors)]
            callsign = flight_path['flight_plan']['callsign']
            
            # Plot route
            waypoint_lats = [coord[0] for coord in flight_path['route_coords']]
            waypoint_lons = [coord[1] for coord in flight_path['route_coords']]
            
            ax.plot(waypoint_lons, waypoint_lats, color=color, linewidth=2, 
                   alpha=0.6, linestyle='--', zorder=3)
            ax.scatter(waypoint_lons, waypoint_lats, color=color, s=80, 
                      alpha=0.8, marker='o', edgecolors='white', zorder=5)
            
            # Add waypoint labels
            for j, (lat, lon) in enumerate(flight_path['route_coords']):
                if j == 0:
                    label = f'{callsign}_START'
                else:
                    label = flight_path['flight_plan']['route'][j-1]
                
                ax.annotate(label, (lon, lat), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                           alpha=0.7, edgecolor='white'), zorder=6)
            
            # Initialize animated elements
            aircraft_marker, = ax.plot([], [], 'o', color=color, markersize=15, 
                                     zorder=10, label=callsign, markeredgecolor='black', 
                                     markeredgewidth=2)
            trail_line, = ax.plot([], [], '-', color=color, linewidth=3, alpha=0.8, zorder=9)
            
            aircraft_elements.append({
                'flight_path': flight_path,
                'aircraft_marker': aircraft_marker,
                'trail_line': trail_line,
                'callsign': callsign
            })
        
        # Find max steps
        max_steps = max(len(fp['time_steps']) for fp in all_flight_paths)
        
        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), zorder=11)
        
        ax.set_xlabel('Longitude (degrees)', fontsize=12)
        ax.set_ylabel('Latitude (degrees)', fontsize=12)
        ax.set_title('Scenario visualization', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        def animate(frame):
            info_lines = []
            
            for aircraft in aircraft_elements:
                time_steps = aircraft['flight_path']['time_steps']
                
                if frame < len(time_steps):
                    step = time_steps[frame]
                    
                    # Update aircraft position
                    aircraft['aircraft_marker'].set_data([step['longitude']], [step['latitude']])
                    
                    # Update trail
                    trail_start = max(0, frame - 15)
                    trail_lons = [time_steps[i]['longitude'] for i in range(trail_start, frame + 1)]
                    trail_lats = [time_steps[i]['latitude'] for i in range(trail_start, frame + 1)]
                    aircraft['trail_line'].set_data(trail_lons, trail_lats)
                    
                    info_lines.append(f"{aircraft['callsign']}: {step['latitude']:.3f}°N, {step['longitude']:.3f}°E")
                else:
                    info_lines.append(f"{aircraft['callsign']}: COMPLETED")
            
            current_time = frame * self.time_step_seconds
            info_text.set_text(f"⏰ Time: {current_time//60:02.0f}:{current_time%60:02.0f}\n" + 
                              "\n".join(info_lines))
            
            return [aircraft['aircraft_marker'] for aircraft in aircraft_elements] + \
                   [aircraft['trail_line'] for aircraft in aircraft_elements] + [info_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=max_steps, 
                                     interval=interval, blit=True, repeat=True)
        
        if save_path:
            if save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=10, bitrate=1800)
            elif save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to: {save_path}")
        
        plt.show()
        return anim


def create_visualization(xml_content, waypoints_csv='SingaporeWaypoints.csv', 
                        sector_csv='sector6coords 1.csv', output_name='aircraft'):
    """
    Create visualization from XML content.
    
    Args:
        xml_content: XML string with aircraft data
        waypoints_csv: Path to waypoints CSV file
        sector_csv: Path to sector boundary CSV file
        output_name: Base name for output files
    """
    print(f"Creating visualization for {output_name}...")
    
    # Initialize visualizer
    visualizer = MultiAircraftVisualizer(waypoints_csv, sector_csv)
    
    # Parse and calculate
    flight_plans = visualizer.parse_xml_flight_plans(xml_content)
    print(f"Found {len(flight_plans)} aircraft")
    
    all_flight_paths = visualizer.calculate_all_flight_paths(flight_plans)
    
    # Create outputs
    visualizer.create_static_plot(all_flight_paths, f'{output_name}_static.png')
    visualizer.create_animation(all_flight_paths, f'{output_name}_animation.gif')
    
    print(f"Files created: {output_name}_static.png, {output_name}_animation.gif")
    return visualizer, all_flight_paths


