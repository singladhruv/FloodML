import csv
import datetime
import pickle
import requests

API_KEY = "3JEF3KYEGWP83K5DWW37CKDJ8"

def _build_url(lat, lon):
    return (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/"
        f"forecast?locations={lat}%2C%20{lon}&aggregateHours=24&unitGroup=us&shortColumnNames=false"
        f"&contentType=json&key={API_KEY}"
    )

def get_data(lat, lon):
    
    
    k = _build_url(lat, lon)
    response = requests.get(k)
    payload = response.json()
    if 'locations' not in payload:
        message = payload.get('message', 'Unknown error from Visual Crossing API')
        raise RuntimeError(f"Visual Crossing API error: {message}")
    x = payload['locations']
    for i in x:
        y = x[i]['values']

    final = [0, 0, 0, 0, 0, 0]
    if not y:
        raise RuntimeError("No forecast values returned from Visual Crossing API")
    count = len(y)

    for j in y:
        final[0] += j['temp']
        if j['maxt'] > final[1]:
            final[1] = j['maxt']
        final[2] += j['wspd']
        final[3] += j['cloudcover']
        final[4] += j['precip']
        final[5] += j['humidity']
    final[0] /= count
    final[2] /= count
    final[3] /= count
    final[5] /= count

    return final

def testConnection():
    return "yo"
