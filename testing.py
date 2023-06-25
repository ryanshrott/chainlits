import googlemaps
import requests
import os
from dotenv import load_dotenv

load_dotenv()
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_KEY"))


def geocoords(place_id):
    place_details = gmaps.place(place_id)
    latitude = place_details['result']['geometry']['location']['lat']
    longitude = place_details['result']['geometry']['location']['lng']
    return {'lat': latitude, 'lon': longitude}


def get_gmaps_address(query):
    return gmaps.places_autocomplete(query, types=["geocode"], components={"country": "ca"})


def get_address_info(top_suggestion, unit_number=None):
    coords = geocoords(gmaps.find_place(top_suggestion, 'textquery')['candidates'][0]['place_id'])
    return {
        'description': top_suggestion['description'],
        'street_num': top_suggestion['terms'][0]['value'],
        'street_name': top_suggestion['terms'][1]['value'].split(' ')[0],
        'city': top_suggestion['terms'][2]['value'],
        'unit_number': unit_number,
        'lat': coords['lat'],
        'lon': coords['lon']
    }

def price_and_find_comparables(street_number: str, street_name: str, city: str, house_category: str,  unit_number:str='', beds:str='', baths:str='', house_sqft:str='', land_sqft:str='', use_comparables:str="False"):
    """
    Price a home and optionaly find comparable properties based on the street number, street name, city, house type, and optional parameters.
    :param street_number: The street number (e.g 123)
    :param street_name: The street name (e.g. Harvie Ave)
    :param city: The city of the property (e.g. Toronto)
    :param house_category: The type of the house <<Condo, Detached, Semi, Link, Multiplex, Vacant Land, Condo Townhouse, Freehold Townhouse, Other>>
    :param unit_number: The unit number like 402 (optional)
    :param beds: The number of bedrooms (optional)
    :param baths: The number of bathrooms (optional)
    :param house_sqft: The square footage of the house (optional)
    :param land_sqft: The square footage of the land (optional)
    :param use_comparables: Whether to use comparables or not (optional)

    :return: Pricing information for the property
    """
    print(street_number, street_name, city, house_category,  unit_number, beds, baths, house_sqft, land_sqft, bool(use_comparables))
    suggestion = get_gmaps_address(f"{street_number} {street_name} {city}")
    info = get_address_info(suggestion[0], unit_number)
    headers = {
        'Authorization': os.getenv("HUGGING_FACE_API_KEY"),
        'Content-Type': 'application/json'
    }
    data = [
            info['street_num'],
            info['street_name'],
            info['city'],
            unit_number,
            info['lat'],
            info['lon'],
            house_category,
            bool(use_comparables),
            True,
            beds,
            baths,
            house_sqft,
            land_sqft,
            '',
            {'headers': ['None'], 'data': [['None']]}
        ]
    print(data)
    #print({'headers': ['None'], 'data': [['None']]} if modifications is None else modifications)
    response = requests.post("https://rshrott-smartbids.hf.space/run/price_from_info", json={
        "data": data
    }, headers=headers)


    if response.status_code == 200:
        data = response.json()["data"]
        return str(data)
    else:
        return str(response.status_code) + ' ' + str(response.json())
    
def price_home(street_number: str, street_name: str, city: str, house_category: str,  unit_number:str='', beds:str='', baths:str='', house_sqft:str='', land_sqft:str=''):
    """
    Price a property based on the street number, street name, city, house type, and optional parameters.
    :param street_number: The street number (e.g 123)
    :param street_name: The street name (e.g. Harvie Ave)
    :param city: The city of the property (e.g. Toronto)
    :param house_category: The type of the house <<Condo, Detached, Semi, Link, Multiplex, Vacant Land, Condo Townhouse, Freehold Townhouse, Other>>
    :param unit_number: The unit number like 402 (optional)
    :param beds: The number of bedrooms (optional)
    :param baths: The number of bathrooms (optional)
    :param house_sqft: The square footage of the house (optional)
    :param land_sqft: The square footage of the land (optional)

    :return: Pricing information for the property
    """
    suggestion = get_gmaps_address(f"{street_number} {street_name} {city}")
    info = get_address_info(suggestion[0], unit_number)
    headers = {
        'Authorization': os.getenv("HUGGING_FACE_API_KEY"),
        'Content-Type': 'application/json'
    }
    data = [
        info['street_num'],
        info['street_name'],
        info['city'],
        unit_number,
        info['lat'],
        info['lon'],
        house_category,
        beds,
        baths,
        house_sqft,
        land_sqft
    ]
    print(data)

    response = requests.post("https://rshrott-smartbids.hf.space/run/smartbids_pricer", json={
        "data": data
    }, headers=headers).json()

    if response.get("data"):
        return str(response["data"])
    else:
        return 'The pricing API returned a server error. Did you enter the correct address? Did you specifiy the property type like Detached or Condo?' 


street_number = '71'
street_name = 'Charles St E'
city = 'Toronto'
unit_number = '402'
house_type = 'Condo'
beds = '2'
baths = '2'
sqft = '1315'
land_sqft = ''
result = price_and_find_comparables('218', 'Harvie Ave', 'Toronto','Detached', '',  '', '', '', '', "False")
print(result)
#result = price_and_find_comparables(street_number, street_name, city, house_type, unit_number, beds, baths, sqft, land_sqft)
#print(result)

#result = price_and_find_comparables('218', 'Harvie Ave', 'Toronto', 'Detached', '', '10', '10', '10000', '10000', 'False')
#print(result)