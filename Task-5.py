#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import folium

def get_ip_geolocation(ip):
    url = f"http://ip-api.com/json/{ip}"
    response = requests.get(url)
    data = response.json()
    return data

def display_map(latitude, longitude):
# Create a map centered around the provided coordinates
    map_obj = folium.Map(location=[latitude, longitude], zoom_start=10)
# Add a marker for the provided coordinates
    folium.Marker([latitude, longitude]).add_to(map_obj)
# Save the map to an HTML file
    map_obj.save("geolocation_map.html")
    print("Map saved as geolocation_map.html")

def main():
    ip_address = input("Enter the IP address to track: ")
    location_data = get_ip_geolocation(ip_address)
    
    if location_data["status"] == "success":
        latitude = location_data["lat"]
        longitude = location_data["lon"]
        print(f"Latitude: {latitude}, Longitude: {longitude}")
        display_map(latitude, longitude)
    else:
        print("Failed to fetch geolocation data for the provided IP address.")

if __name__ == "__main__":
    main()


# In[ ]: