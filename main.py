import requests
from pybaseball import statcast, statcast_pitcher, playerid_lookup, pitching_stats

def main():
    response = requests.get("https://httpbin.org/get")
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
