import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

class TwoAircraftScenarioChecker:
    """
    Validates 2-aircraft XML scenarios against specific prompt requirements.
    """
    
    def __init__(self):
        # Define updated airways
        self.airways = {
            'M758': ['IDSEL', 'URIGO', 'VISAT', 'MABAL', 'ELGOR', 'OPULA', 'LUSMO'],
            'N884': ['VMR', 'LENDA', 'LIPRO', 'LEBIN', 'ONAPO'],
            'M761': ['SABIP', 'BOBOB', 'OMBAP', 'VERIN', 'BUNTO', 'LIPRO', 'KILOT', 'OTLON', 'KETOD', 'OBDAB', 'VPK'],
            'M771': ['VMR', 'RAXIM', 'OTLON', 'VISAT', 'DUBSA', 'DAMOG', 'DOLOX']
        }
    
    def parse_aircraft_from_xml(self, xml_content):
        """Extract aircraft information from XML."""
        root = ET.fromstring(xml_content)
        aircraft_list = []
        
        for initial_fp in root.findall('.//initial-flightplans'):
            aircraft = {
                'callsign': initial_fp.find('callsign').text,
                'start_time': int(initial_fp.find('time').text),
                'aircraft_type': initial_fp.find('type').text,
                'route': [route.text for route in initial_fp.findall('air_route')]
            }
            aircraft_list.append(aircraft)
        
        # Sort by start time
        aircraft_list.sort(key=lambda x: x['start_time'])
        return aircraft_list
    
    def identify_airway(self, route):
        """Identify which airway a route belongs to."""
        route_set = set(route)
        
        for airway_name, airway_waypoints in self.airways.items():
            airway_set = set(airway_waypoints)
            # Check if route has significant overlap with airway waypoints
            overlap = len(route_set.intersection(airway_set))
            if overlap >= 2:  # At least 2 waypoints match
                return airway_name
        
        return 'UNKNOWN'
    
    def check_aircraft_count(self, aircraft_list, expected_count=2):
        """Check if correct number of aircraft are present."""
        actual_count = len(aircraft_list)
        if actual_count == expected_count:
            return True, f" PASSED: Exactly {expected_count} aircraft found"
        else:
            return False, f" FAILED: Expected {expected_count} aircraft, found {actual_count}"
    
    def check_airway_assignments(self, aircraft_list, expected_airways=['M758', 'N884']):
        """Check if aircraft are assigned to correct airways."""
        results = []
        all_correct = True
        
        for i, aircraft in enumerate(aircraft_list):
            actual_airway = self.identify_airway(aircraft['route'])
            expected_airway = expected_airways[i] if i < len(expected_airways) else 'UNKNOWN'
            
            if actual_airway == expected_airway:
                results.append(f" Aircraft {i+1} ({aircraft['callsign']}): Correctly assigned to {actual_airway}")
            else:
                results.append(f" Aircraft {i+1} ({aircraft['callsign']}): Expected {expected_airway}, but on {actual_airway}")
                results.append(f"   Route: {' → '.join(aircraft['route'])}")
                results.append(f"   {expected_airway} waypoints: {', '.join(self.airways.get(expected_airway, []))}")
                all_correct = False
        
        return all_correct, results
    
    def check_timing(self, aircraft_list, first_start_time=100, second_start_time=300):
        """Check timing requirements for specific start times."""
        results = []
        all_correct = True
        
        # Check first aircraft start time
        first_start = aircraft_list[0]['start_time']
        if first_start == first_start_time:
            results.append(f" First aircraft starts at {first_start_time} seconds: {aircraft_list[0]['callsign']} at T+{first_start}s")
        else:
            results.append(f" Expected first aircraft to start at {first_start_time}s, but {aircraft_list[0]['callsign']} starts at T+{first_start}s")
            all_correct = False
        
        # Check second aircraft start time
        if len(aircraft_list) >= 2:
            actual_second_start = aircraft_list[1]['start_time']
            
            if actual_second_start == second_start_time:
                results.append(f" Second aircraft starts at {second_start_time} seconds: {aircraft_list[1]['callsign']} at T+{actual_second_start}s")
            else:
                results.append(f" Expected second aircraft to start at {second_start_time}s, but {aircraft_list[1]['callsign']} starts at T+{actual_second_start}s")
                all_correct = False
            
            # Show separation info
            actual_separation = actual_second_start - first_start
            results.append(f"  Separation: {actual_separation} seconds ({aircraft_list[0]['callsign']} → {aircraft_list[1]['callsign']})")
        
        return all_correct, results
    
    def validate_scenario(self, xml_content, prompt, expected_airways=['M758', 'N884'], 
                         first_start_time=100, second_start_time=300):
        """
        Validate a 2-aircraft XML scenario against requirements.
        
        Args:
            xml_content: XML string to validate
            prompt: Original user prompt
            expected_airways: List of expected airways for aircraft 1, 2
            first_start_time: Expected start time for first aircraft
            second_start_time: Expected start time for second aircraft
        """
        print("TWO AIRCRAFT SCENARIO VALIDATION")
        print("=" * 60)
        
        print("=" * 60)
        
        # Parse XML
        aircraft_list = self.parse_aircraft_from_xml(xml_content)
        
        print(f"\n AIRCRAFT FOUND:")
        for i, aircraft in enumerate(aircraft_list, 1):
            route_str = ' → '.join(aircraft['route'])
            print(f"{i}. {aircraft['callsign']} ({aircraft['aircraft_type']}) starts at T+{aircraft['start_time']}s")
            print(f"   Route: {route_str}")
        
        # Perform checks
        print(f"\n CHECK 1: AIRCRAFT COUNT")
        count_pass, count_msg = self.check_aircraft_count(aircraft_list, 2)
        print(count_msg)
        
        print(f"\n CHECK 2: AIRWAY ASSIGNMENTS")
        airway_pass, airway_results = self.check_airway_assignments(aircraft_list, expected_airways)
        for result in airway_results:
            print(result)
        
        print(f"\n CHECK 3: TIMING REQUIREMENTS")
        timing_pass, timing_results = self.check_timing(aircraft_list, first_start_time, second_start_time)
        for result in timing_results:
            print(result)
        
        # Overall Result
        print(f"\n{'='*60}")
        print(f" SUMMARY:")
        print(f"• Aircraft Count: {' PASS' if count_pass else ' FAIL'}")
        print(f"• Airway Assignment: {'PASS' if airway_pass else ' FAIL'}")
        print(f"• Timing: {' PASS' if timing_pass else ' FAIL'}")
        
        overall_pass = count_pass and airway_pass and timing_pass
        
        if overall_pass:
            print(f"\n OVERALL RESULT: ALL REQUIREMENTS MET! ")
        else:
            print(f"\n  OVERALL RESULT: SOME REQUIREMENTS NOT MET ")
        
        print("=" * 60)
        
        # Detailed Analysis
        self._print_detailed_analysis(aircraft_list, count_pass, airway_pass, timing_pass, 
                                    expected_airways, first_start_time, second_start_time)
        
        return overall_pass, {
            'count_pass': count_pass,
            'airway_pass': airway_pass,
            'timing_pass': timing_pass,
            'aircraft_count': len(aircraft_list),
            'aircraft_airways': [self.identify_airway(ac['route']) for ac in aircraft_list],
            'start_times': [ac['start_time'] for ac in aircraft_list],
            'separation': aircraft_list[1]['start_time'] - aircraft_list[0]['start_time'] if len(aircraft_list) >= 2 else 0
        }
    
    def _print_detailed_analysis(self, aircraft_list, count_pass, airway_pass, timing_pass, 
                               expected_airways, first_start_time, second_start_time):
        """Print detailed analysis of what's working and what needs fixing."""
        print(f"\n DETAILED ANALYSIS:")
        
        # What's working
        working = []
        if count_pass:
            working.append("Correct number of aircraft (2)")
        if len(aircraft_list) >= 1 and aircraft_list[0]['start_time'] == first_start_time:
            working.append(f"First aircraft starts at correct time ({first_start_time}s)")
        if len(aircraft_list) >= 2 and aircraft_list[1]['start_time'] == second_start_time:
            working.append(f"Second aircraft starts at correct time ({second_start_time}s)")
        
        for i, aircraft in enumerate(aircraft_list):
            actual_airway = self.identify_airway(aircraft['route'])
            expected_airway = expected_airways[i] if i < len(expected_airways) else 'UNKNOWN'
            if actual_airway == expected_airway:
                working.append(f"Aircraft {i+1} correctly on {actual_airway} airway")
        
        if working:
            print(f"\n WHAT'S WORKING:")
            for item in working:
                print(f"• {item}")
        
        # What needs fixing
        issues = []
        if not count_pass:
            issues.append("Aircraft count incorrect")
        
        for i, aircraft in enumerate(aircraft_list):
            actual_airway = self.identify_airway(aircraft['route'])
            expected_airway = expected_airways[i] if i < len(expected_airways) else 'UNKNOWN'
            if actual_airway != expected_airway:
                issues.append(f"Aircraft {i+1} should be on {expected_airway}, but is on {actual_airway}")
        
        if len(aircraft_list) >= 1 and aircraft_list[0]['start_time'] != first_start_time:
            issues.append(f"First aircraft should start at {first_start_time}s, but starts at {aircraft_list[0]['start_time']}s")
        
        if len(aircraft_list) >= 2 and aircraft_list[1]['start_time'] != second_start_time:
            issues.append(f"Second aircraft should start at {second_start_time}s, but starts at {aircraft_list[1]['start_time']}s")
        
        if issues:
            print(f"\n ISSUES TO FIX:")
            for issue in issues:
                print(f"• {issue}")


def check_two_aircraft_xml(xml_content, prompt):
    
    checker = TwoAircraftScenarioChecker()
    return checker.validate_scenario(
        xml_content, 
        prompt, 
        expected_airways=['M758', 'N884'],
        first_start_time=100,
        second_start_time=300
    )

