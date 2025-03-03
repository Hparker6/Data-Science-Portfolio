import overpy
import requests
import os

from typing import List, Dict


def get_locations(lat:float, lon: float, dist: int = 1000, location_type: str = 'restaurant') -> List[Dict]:
    """
    Get all locations of a certain type that lie near to a certain point, using OpenStreetMap.

    :param lat: Latitude of the point
    :param lon: Longitude of the point
    :param dist: (optional) Maximum distance of the locations from the selected point in m. Default: 1000
    :param location_type: (optional) Type of the location that is returned.
        Currently supported: restaurant (default), park, sports, socialize
    :return: List of Dictionaries which contain three entries for each location:
                                - name: Name of the location
                                - lon: Latitude
                                - lat: Longitude
    """
    api = overpy.Overpass()

    if location_type == 'restaurant':
        tags = '["amenity"~"restaurant|fast_food"]'
    elif location_type == 'park':
        tags = '["leisure"~"park|garden|nature_reserve"]'
    elif location_type == 'sports':
        tags = ('["leisure"~"disc_golf_course|bowling_alley|disc_golf_course|fitness_centre|golf_course|ice_rink|'
                'sports_centre|sports_hall|miniature_golf|pitch|stadium|track|swimming_pool"]')
    elif location_type == 'socialize':
        tags = '["amenity"~"bar|biergarten|cafe|pub"]'

    else:
        print('Location type not supported. Supported types: restaurant, park')
        return []

    result = api.query(f"""
        node
        {tags}
        (around:{dist}, {lat}, {lon});
        out;
        """)

    ret = []

    for node in result.nodes:
        ret.append({'name' : node.tags['name'], 'lon' : node.lon, 'lat' : node.lat})

    return ret


def get_way(lat_start, lon_start, lat_target, lon_target, mobility_mode: str = None) -> List[Dict]:
    """
    Maps a way between a start point and an endpoint and returns the distance and time between two points.

    :param lon_start: Longitude of start point
    :param lat_start: Latitude of start point
    :param lon_target: Longitude of target
    :param lat_target: Latitude of target
    :param mobility_mode: (optional) Can be used to select the current vehicle the user has.
        When not given computes values for walking, cycle and car.
        Possible values: foot, cycle, car,
                        (also all profiles from https://giscience.github.io/openrouteservice-r/reference/ors_profile.html)
    :return: List of dictionaries containing the distance and time of the way between the two locations
    """

    try:
        params = {'api_key' : os.environ['OPRS_API_KEY'],
                  'start' : f'{lon_start},{lat_start}',
                  'end': f'{lon_target},{lat_target}'}
    except KeyError:
        print('You must first set the Environment variable "OPRS_API_KEY"'
              ' with your Openrouteservice API key. Aborting')
        return []

    profile = ["foot-walking", "cycling-regular", "driving-car"]
    mapping = {'foot': 0, 'cycle': 1, 'car': 2}

    if mobility_mode is not None:
        try:
            profile = [profile[mapping[mobility_mode]],]
        except KeyError:
            profile = [mobility_mode,]
    res = []


    for p in profile:
        url = f'https://api.openrouteservice.org/v2/directions/{p}'
        response = requests.get(url, params)
        status_code = response.status_code
        data = response.json()
        if status_code == 200:
            res.append(data['features'][0]['properties']['summary'])
        else:
            print(f'Request failed. Status Code: {status_code}')
            return [dict()]

    if len(res) == 0:
        res = [dict()]

    return res


def mapping_call(lat: float, lon: float, radius: int = 1000, location_type: str = 'restaurant',mobility_mode : str = 'foot'):
    """
    Function that can be called from the front-end and returns a list of dictionaries with all information for
    the selected locations around the desired point. The number of locations is limited to 40, due to the API key usage limit.

    :param lat: Latitude of the point
    :param lon: Longitude of the point
    :param radius: Maximum distance of the locations from the selected point in m. Default: 1000
    :param location_type: (optional) Type of the location that is returned.
        Currently supported: restaurant (default), park, sports, socialize
    :param mobility_mode: (optionl) Can be used to select the current vehicle the user has.
        When not given computes values for walking, cycle and car.
        Possible values: foot, cycle, car,
                        (also all profiles from https://giscience.github.io/openrouteservice-r/reference/ors_profile.html)
    :return: Dictionaries with five keys: name, lon, lat, distance and duration.
    """
    locs = get_locations(lat, lon, radius, location_type)[:20]
    return [{**l, **(get_way(lat, lon, l['lat'], l['lon'], mobility_mode)[0])} for l in locs]
